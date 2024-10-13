import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Function
from models.utils import *


class SelfExpressionLayer(nn.Module):
    def __init__(self, output_dim, domain_dim, batch_size, d_model, weight_c, w_c=1, w_s=0.5, w_b=1, w_sum=0):
        super(SelfExpressionLayer, self).__init__()
        self.weight_c = weight_c
        self.w_c = w_c
        self.w_s = w_s
        self.w_b = w_b
        self.w_sum = w_sum
        self.m = d_model
        self.class_n = output_dim * batch_size * (domain_dim-1)
        self.class_list = [batch_size*(domain_dim-1) for _ in range(output_dim)]
        self.class_block = self.change_class_list(self.class_list)
        self.class_self_expression = SelfExpression(self.class_n, self.weight_c, self.class_block)
        self.class_zero1 = torch.zeros((self.class_n, self.class_n), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_zero2 = torch.zeros((self.class_n, self.m), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.domain_n = output_dim * batch_size * domain_dim
        self.domain_list = [batch_size*output_dim for _ in range(domain_dim)]
        self.domain_block = self.change_domain_list(self.domain_list)
        self.domain_self_expression = SelfExpression(self.domain_n, self.weight_c, self.domain_block)
        self.domain_zero1 = torch.zeros((self.domain_n, self.domain_n),
                                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.domain_zero2 = torch.zeros((self.domain_n, self.m),
                                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.x1 = []
        self.x2 = []
        self.zx = []
        self.dx = []

    def forward(self, X, Y):
        return self.class_forward(X, Y)

    def class_forward(self, X, Y):
        # 根据 y 对 X 进行重新排序
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        indices = torch.argsort(Y.squeeze(), dim=0, descending=False)
        X_sorted = X[indices]
        end_event.record()
        # 计算耗费时间
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # 时间单位是毫秒
        print(f'Time taken: {elapsed_time} ms')

        # 注意要使用 torch.matmul 或者 @ 进行矩阵乘法
        ZX = self.class_self_expression(X_sorted)

        # 将 ZX 变回原始输入 X 的排序
        X_restored = ZX[torch.argsort(indices)]

        self.x1 = X
        self.zx = X_restored
        return X_restored

    def domain_forward(self, X, D):
        # 根据 y 对 X 进行重新排序
        indices = torch.argsort(D.squeeze(), dim=0, descending=False)
        X_sorted = X[indices]

        # 注意要使用 torch.matmul 或者 @ 进行矩阵乘法
        ZX = self.domain_self_expression(X_sorted)

        # 将 ZX 变回原始输入 X 的排序
        X_restored = ZX[torch.argsort(indices)]

        self.x2 = X
        self.dx = X_restored
        return X_restored

    def loss(self):
        Coefficient = self.class_self_expression.Coefficient
        loss_coe = F.mse_loss(Coefficient, self.zero1, reduction='sum')
        loss_self = F.mse_loss(self.x-self.zx, self.zero2, reduction='sum')
        loss_block = F.mse_loss(torch.mul(self.block, Coefficient), self.zero1, reduction='sum')
        loss_sum = torch.sum(torch.abs(Coefficient) - Coefficient)
        loss_se = self.w_c * loss_coe + self.w_s * loss_self + self.w_b * loss_block
        return loss_se, loss_coe, loss_self, loss_block, loss_sum

    def loss_s(self):
        loss_se1, loss_coe1, loss_self1, loss_block1, loss_sum1 = self.loss_class()
        loss_se2, loss_coe2, loss_self2, loss_block2, loss_sum2 = self.loss_domain()
        loss_se = loss_se1 + loss_se2
        loss_coe = loss_coe1 + loss_coe2
        loss_self = loss_self1 + loss_self2
        loss_block = loss_block1 + loss_block2
        loss_sum = loss_sum1 + loss_sum2
        return loss_se, loss_coe, loss_self, loss_block, loss_sum

    def loss_domain(self):
        Coefficient = self.domain_self_expression.Coefficient
        loss_coe = F.mse_loss(Coefficient, self.domain_zero1, reduction='sum')
        loss_self = F.mse_loss(self.x2-self.dx, self.domain_zero2, reduction='sum')
        # loss_self = 0
        loss_block = F.mse_loss(torch.mul(self.domain_block, Coefficient), self.domain_zero1, reduction='sum')
        loss_sum = torch.sum(torch.abs(Coefficient) - Coefficient)
        loss_se = self.w_c * loss_coe + self.w_s * loss_self + self.w_b * loss_block
        return loss_se, loss_coe, loss_self, loss_block, loss_sum

    def loss_class(self):
        Coefficient = self.class_self_expression.Coefficient
        loss_coe = F.mse_loss(Coefficient, self.class_zero1, reduction='sum')
        loss_self = F.mse_loss(self.x1-self.zx, self.class_zero2, reduction='sum')
        # loss_self = 0
        loss_block = F.mse_loss(torch.mul(self.class_block, Coefficient), self.class_zero1, reduction='sum')
        loss_sum = torch.sum(torch.abs(Coefficient) - Coefficient)
        loss_se = self.w_c * loss_coe + self.w_s * loss_self + self.w_b * loss_block
        return loss_se, loss_coe, loss_self, loss_block, loss_sum

    def change_class_list(self, class_list):
        mask_block = torch.ones((sum(class_list), sum(class_list)), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        index = 0
        for i in range(len(class_list)):
            mask_block[index:index + class_list[i], index:index + class_list[i]] = 0
            index += class_list[i]
        class_block = (mask_block + torch.eye(sum(class_list), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        return class_block
        # plt.imshow(class_block.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图形
        # plt.show()

    def change_domain_list(self, domain_list):
        window = len(domain_list) - 1
        # window = 5
        mask_block = torch.ones((sum(domain_list), sum(domain_list)), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        num = domain_list[0]
        y_index = 0
        x_index = num
        for i in range(len(domain_list)):
            if x_index + window * num < sum(domain_list):
                mask_block[y_index:y_index + num, x_index:x_index + window * num] = 0
            else:
                mask_block[y_index:y_index + num, x_index:] = 0
                mask_block[y_index:y_index + num, :(i - (len(domain_list) - window - 1)) * num] = 0
            y_index += num
            x_index += num
        # mask_block[sum(domain_list[:-1]):, :(self.domain_num - 1) * num] = 0
        mask_block[sum(domain_list[:-1]):, :window * num] = 0
        domain_block = mask_block
        return domain_block
        # plt.imshow(domain_block.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图形
        # plt.show()


class MySelfAttentionNetLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_attention_heads, dropout_prob, class_output_dim, domain_dim, batch_size, weight_c):
        super(MySelfAttentionNetLayer, self).__init__()
        self.multihead_attention = SelfAttention(d_model, num_attention_heads, dropout_prob)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.feed_forward2 = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.btn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(0.2)

        self.self_expression = SelfExpressionLayer(class_output_dim, domain_dim, batch_size, d_model, weight_c,)

    def forward(self, x, y, d, len_s):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        # 1
        # Self-Expression
        # sx = x[:len_s]
        # tx = x[len_s:]
        # se_output = self.self_expression.class_forward(sx, y)
        # sx = sx + self.dropout(se_output)
        # sx = self.norm3(sx)
        # x = torch.concat((sx, tx), dim=0)

        # 2
        sx = x[:len_s]
        tx = x[len_s:]
        se_output = self.self_expression.class_forward(sx, y)
        se_output = torch.concat((se_output, tx), dim=0)
        x = x + self.dropout(se_output)
        x = self.norm3(x)
        self.self_expression.zx = x[:len_s]

        se_output = self.self_expression.domain_forward(x, d)
        x = x + self.dropout(se_output)
        x = self.norm4(x)

        # 3
        # sx = x[:len_s]
        # tx = x[len_s:]
        # se_output = self.self_expression.class_forward(sx, y)
        # se_output = torch.concat((se_output, tx), dim=0)
        # se_output = self.self_expression.domain_forward(se_output, d)
        # x = x + self.dropout(se_output)
        # x = self.norm4(x)

        # Feed-Forward
        ff_output = self.feed_forward2(x)
        x = x + self.dropout(ff_output)
        x = self.norm5(x)

        return x

    def valid(self, x):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        # Feed-Forward
        ff_output = self.feed_forward2(x)
        x = x + self.dropout(ff_output)
        x = self.norm5(x)

        return x


# 定义一个完整的 Self-Attention Net
class MySelfAttentionNet(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, class_output_dim, domain_dim, batch_size, weight_c):
        super(MySelfAttentionNet, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.btn = nn.BatchNorm1d(d_model)
        self.ln = nn.LayerNorm(d_model)
        self.encoder = nn.Sequential()
        for i in range(num_layers):
            if i == 1:
                self.encoder.add_module('self-Attention%d' % i,
                                        MySelfAttentionNetLayer(d_model, d_model, num_heads, dropout, class_output_dim,
                                                                domain_dim, batch_size, weight_c))
            else:
                self.encoder.add_module('self-Attention%d' % i,
                                        MySelfAttentionNetLayer(d_model, d_model, num_heads, dropout, class_output_dim,
                                                                domain_dim, batch_size, weight_c))

    def forward(self, sx, tx, y, d):
        x = torch.concat((sx, tx), dim=0)
        len_s = sx.shape[0]
        # x = self.embedding(x)
        x = self.ln(x)
        for layer in self.encoder:
            residual = x  # 保存残差连接
            # x = self.btn(x)
            x = layer(x, y, d, len_s)
            x = x + residual  # 添加残差连接
        return x

    def valid(self, x):
        # x = self.embedding(x)
        x = self.ln(x)
        for layer in self.encoder:
            residual = x  # 保存残差连接
            # x = self.btn(x)
            x = layer.valid(x)
            x = x + residual  # 添加残差连接
        return x

    def loss(self):
        loss = 0
        for layer in self.encoder:
            temp_loss, _, _, _, _ = layer.self_expression.loss_s()
            loss += temp_loss
        return loss


class MMDLoss(nn.Module):
    """
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    """
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class MyCSNet(nn.Module):
    def __init__(self, input_size_list, batch_size, d_model, num_AttLayers, num_heads, dropout=0.1, weight_c=1e-1):
        super(MyCSNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_c = weight_c
        self.multi_modal = True
        self.input_dim = d_model * 1
        self.output_dim = 3
        self.domain_num = 12
        self.input_size = input_size_list[0]

        self.embedding1 = nn.Sequential(
            nn.Linear(input_size_list[0], d_model),
            # nn.Linear(d_model, d_model)
        )
        # self.embedding2 = nn.Linear(input_size_list[1], d_model)

        self.selfAtt_encoder = MySelfAttentionNet(self.input_size, d_model, num_heads, num_AttLayers, dropout, self.output_dim, self.domain_num, batch_size, weight_c)

        self.class_classifier = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.output_dim),

            # nn.Linear(in_features=self.input_dim, out_features=64),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=self.output_dim),

            # nn.Linear(in_features=self.input_dim, out_features=self.output_dim),

        )

        self.domain_classifier = nn.Sequential(
            # nn.Linear(in_features=self.input_dim, out_features=64),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=self.domain_num),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=self.output_dim),

            nn.Linear(in_features=self.input_dim, out_features=self.domain_num),
        )

        self.criterion_mmd = MMDLoss()
        self.loss_mmd = 0

    def forward(self, source_x_list, target_x_list=None, y=None, d=None, alpha=0, valid=False):
        if valid:
            return self.valid(source_x_list)
        f_list = []
        df_list = []
        sx = source_x_list[0].float().to(self.device)
        tx = target_x_list[0].float().to(self.device)
        x = torch.concat((sx, tx), dim=0)
        x = self.embedding1(x)
        sx = x[:sx.shape[0]]
        tx = x[sx.shape[0]:]
        out_df = self.selfAtt_encoder(sx, tx, y, d)
        out_f = out_df[:sx.shape[0]]
        f_list.append(out_f)
        df_list.append(out_df)

        res_feature = torch.cat(f_list, dim=1)
        # self.loss_mmd = self.criterion_mmd(res_feature[:sx.shape[0]], res_feature[sx.shape[0]:])
        cls = torch.softmax(self.class_classifier(res_feature), dim=1)

        # domain_feature = torch.cat(df_list, dim=1)
        # domain_feature = ReverseLayerF.apply(domain_feature, alpha)
        # d_cls = torch.softmax(self.domain_classifier(domain_feature), dim=1)

        domain_feature = ReverseLayerF.apply(x, alpha)
        d_cls = torch.softmax(self.domain_classifier(domain_feature), dim=1)

        return f_list, res_feature, cls, d_cls

    def loss(self, cls, y, d_cls, d):
        # loss_reconstruction_class
        loss_se = self.selfAtt_encoder.loss()

        # loss_class
        class_cross_entropy = nn.CrossEntropyLoss()
        loss_class = class_cross_entropy(cls, y.squeeze())

        # loss_domain
        domain_cross_entropy = nn.CrossEntropyLoss()
        loss_domain = domain_cross_entropy(d_cls, d.squeeze())

        loss = 1 * loss_class + 0.2 * loss_domain + 0.1 * loss_se + 0.0 * self.loss_mmd

        return loss, loss_class, loss_domain, loss_se

    def valid(self, x_list):
        f_list = []
        x = x_list[0].float().to(self.device)
        x = self.embedding1(x)
        out_f = self.selfAtt_encoder.valid(x.float().to(self.device))
        f_list.append(out_f)

        res_feature = torch.cat(f_list, dim=1)
        cls = torch.softmax(self.class_classifier(res_feature), dim=1)

        return f_list, res_feature, cls

