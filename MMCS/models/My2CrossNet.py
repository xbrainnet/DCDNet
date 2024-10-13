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
        self.class_n = output_dim * batch_size * (domain_dim - 1)
        self.class_list = [batch_size * (domain_dim - 1) for _ in range(output_dim)]
        self.class_block = self.change_class_list(self.class_list)
        self.class_self_expression = SelfExpression(self.class_n, self.weight_c, self.class_block)
        self.class_zero1 = torch.zeros((self.class_n, self.class_n),
                                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_zero2 = torch.zeros((self.class_n, self.m),
                                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.domain_n = output_dim * batch_size * domain_dim
        self.domain_list = [batch_size * output_dim for _ in range(domain_dim)]
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
        indices = torch.argsort(Y.squeeze(), dim=0, descending=False)
        X_sorted = X[indices]

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
        loss_self = F.mse_loss(self.x - self.zx, self.zero2, reduction='sum')
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
        loss_self = F.mse_loss(self.x2 - self.dx, self.domain_zero2, reduction='sum')
        # loss_self = 0
        loss_block = F.mse_loss(torch.mul(self.domain_block, Coefficient), self.domain_zero1, reduction='sum')
        loss_sum = torch.sum(torch.abs(Coefficient) - Coefficient)
        loss_se = self.w_c * loss_coe + self.w_s * loss_self + self.w_b * loss_block
        return loss_se, loss_coe, loss_self, loss_block, loss_sum

    def loss_class(self):
        Coefficient = self.class_self_expression.Coefficient
        loss_coe = F.mse_loss(Coefficient, self.class_zero1, reduction='sum')
        loss_self = F.mse_loss(self.x1 - self.zx, self.class_zero2, reduction='sum')
        # loss_self = 0
        loss_block = F.mse_loss(torch.mul(self.class_block, Coefficient), self.class_zero1, reduction='sum')
        loss_sum = torch.sum(torch.abs(Coefficient) - Coefficient)
        loss_se = self.w_c * loss_coe + self.w_s * loss_self + self.w_b * loss_block
        return loss_se, loss_coe, loss_self, loss_block, loss_sum

    def change_class_list(self, class_list):
        mask_block = torch.ones((sum(class_list), sum(class_list)),
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        index = 0
        for i in range(len(class_list)):
            mask_block[index:index + class_list[i], index:index + class_list[i]] = 0
            index += class_list[i]
        class_block = (mask_block + torch.eye(sum(class_list),
                                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        return class_block
        # plt.imshow(class_block.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图形
        # plt.show()

    def change_domain_list(self, domain_list):
        window = len(domain_list) - 1
        # window = 5
        mask_block = torch.ones((sum(domain_list), sum(domain_list)),
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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


class MyCrossAttentionNetLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_attention_heads, dropout_prob, class_output_dim, domain_dim, batch_size, weight_c):
        super(MyCrossAttentionNetLayer, self).__init__()
        self.multihead_attention = CrossAttention(d_model, num_attention_heads, dropout_prob)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.feed_forward2 = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.btn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout_prob)

        self.self_expression = SelfExpressionLayer(class_output_dim, domain_dim, batch_size, d_model, weight_c)

    def forward(self, x1, x2, y, d, len_s):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention(x1, x2)
        x1 = x1 + self.dropout(attn_output)
        x = self.norm1(x1)

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

    def valid(self, x1, x2):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention(x1, x2)
        x1 = x1 + self.dropout(attn_output)
        x = self.norm1(x1)

        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        # Feed-Forward
        ff_output = self.feed_forward2(x)
        x = x + self.dropout(ff_output)
        x = self.norm5(x)

        return x


# 定义一个完整的 Cross-Attention Net
class MyCrossAttentionNet(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, class_output_dim, domain_dim, batch_size, weight_c):
        super(MyCrossAttentionNet, self).__init__()
        self.embedding1 = nn.Linear(input_size[0], d_model)
        self.embedding2 = nn.Linear(input_size[1], d_model)
        self.btn1 = nn.BatchNorm1d(d_model)
        self.btn2 = nn.BatchNorm1d(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.encoder1 = nn.Sequential()
        self.encoder2 = nn.Sequential()
        for i in range(num_layers):
            self.encoder1.add_module('cross-Attention%d' % i,
                                     MyCrossAttentionNetLayer(d_model, d_model, num_heads, dropout, class_output_dim, domain_dim, batch_size, weight_c))
            self.encoder2.add_module('cross-Attention%d' % i,
                                     MyCrossAttentionNetLayer(d_model, d_model, num_heads, dropout, class_output_dim, domain_dim, batch_size, weight_c))

    def forward(self, sx1, sx2, tx1, tx2, y, d):
        x1 = torch.concat((sx1, tx1), dim=0)
        x2 = torch.concat((sx2, tx2), dim=0)
        len_s = sx1.shape[0]
        # x1 = self.embedding1(x1)
        # x2 = self.embedding2(x2)
        x1 = self.ln1(x1)
        x2 = self.ln2(x2)
        for layer1, layer2 in zip(self.encoder1, self.encoder2):
            residual1 = x1  # 保存残差连接
            residual2 = x2  # 保存残差连接
            x1 = layer1(x1, residual2, y, d, len_s)
            x2 = layer2(x2, residual1, y, d, len_s)
            x1 = x1 + residual1  # 添加残差连接
            x2 = x2 + residual2  # 添加残差连接
        return x1, x2

    def valid(self, x1, x2):
        # x1 = self.embedding1(x1)
        # x2 = self.embedding2(x2)
        x1 = self.ln1(x1)
        x2 = self.ln2(x2)
        for layer1, layer2 in zip(self.encoder1, self.encoder2):
            residual1 = x1  # 保存残差连接
            residual2 = x2  # 保存残差连接
            x1 = layer1.valid(x1, residual2)
            x2 = layer2.valid(x2, residual1)
            x1 = x1 + residual1  # 添加残差连接
            x2 = x2 + residual2  # 添加残差连接
        return x1, x2

    def loss(self):
        loss = 0
        for layer1, layer2 in zip(self.encoder1, self.encoder2):
            temp_loss1, _, _, _, _ = layer1.self_expression.loss_s()
            temp_loss2, _, _, _, _ = layer2.self_expression.loss_s()
            loss += temp_loss1
            loss += temp_loss2
        return loss


class AttentionFusion(nn.Module):
    def __init__(self, output_dim):
        super(AttentionFusion, self).__init__()
        self.output_dim = output_dim
        self.attention_weights = nn.Parameter(torch.randn(self.output_dim, requires_grad=True))
        self.x1 = []
        self.x2 = []

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        # Calculate weights for all input samples
        tmp1 = torch.matmul(x1, self.attention_weights)
        tmp2 = torch.matmul(x2, self.attention_weights)

        # Use softmax to calculate attention weights
        alpha_1 = F.softmax(tmp1, dim=0)
        # alpha_2 = F.softmax(tmp2, dim=0)
        alpha_2 = 1 - alpha_1

        # Calculate fused tensor using attention weights
        fused_tensor = alpha_1.unsqueeze(1) * x1 + alpha_2.unsqueeze(1) * x2

        # Convert attention weights to numpy for visualization
        alpha = (alpha_1.detach().cpu().numpy(), alpha_2.detach().cpu().numpy())

        return fused_tensor, alpha

    def loss(self):
        x1 = self.x1
        x2 = self.x2
        anchor = []
        positive = []
        negative = []
        for i in range(x1.shape[0]):
            anchor_tmp = x1[i]
            positive_tmp = x2[i]
            for j in range(x1.shape[0]):
                if i == j:
                    continue
                negative_tmp = x1[j]
                anchor.append(anchor_tmp)
                positive.append(positive_tmp)
                negative.append(negative_tmp)
            for j in range(x2.shape[0]):
                if i == j:
                    continue
                negative_tmp = x2[j]
                anchor.append(anchor_tmp)
                positive.append(positive_tmp)
                negative.append(negative_tmp)

        anchor = torch.stack(anchor)
        positive = torch.stack(positive)
        negative = torch.stack(negative)

        return self.triple_loss(anchor, positive, negative)

    def triple_loss(self, anchor, positive, negative, margin=1.0):
        distance_positive = torch.pairwise_distance(anchor, positive)
        distance_negative = torch.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.relu(margin + distance_positive - distance_negative))
        return loss


class MyCCNet(nn.Module):
    def __init__(self, input_size_list, batch_size, d_model, num_AttLayers, num_heads, dropout=0.1, weight_c=1e-1):
        super(MyCCNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_c = weight_c
        self.multi_modal = True
        self.input_dim = d_model * 2
        self.output_dim = 3
        self.domain_num = 12

        self.embedding1 = nn.Linear(input_size_list[0], d_model)
        self.embedding2 = nn.Linear(input_size_list[1], d_model)

        self.selfAtt_encoder = []
        self.crossAtt_encoder = []
        for i in range(len(input_size_list)):
            self.selfAtt_encoder.append(
                MySelfAttentionNet(input_size_list[i], d_model, num_heads, num_AttLayers, dropout, self.output_dim, self.domain_num, batch_size, weight_c))
        self.crossAtt_encoder.append(MyCrossAttentionNet(input_size_list, d_model, num_heads, num_AttLayers, dropout, self.output_dim, self.domain_num, batch_size, weight_c))
        self.selfAtt_encoder = nn.ModuleList(self.selfAtt_encoder)
        self.crossAtt_encoder = nn.ModuleList(self.crossAtt_encoder)

        self.attention_fusion_eeg = AttentionFusion(d_model)
        self.attention_fusion_eye = AttentionFusion(d_model)
        self.attention_fusion = AttentionFusion(d_model)

        self.class_classifier = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
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

    def forward(self, source_x_list, target_x_list=None, y=None, d=None, alpha=0, valid=False):
        if valid:
            return self.valid(source_x_list)
        f_list = []
        df_list = []

        sx1 = source_x_list[0].float().to(self.device)
        tx1 = target_x_list[0].float().to(self.device)
        sx2 = source_x_list[1].float().to(self.device)
        tx2 = target_x_list[1].float().to(self.device)
        x1 = torch.concat((sx1, tx1), dim=0)
        x2 = torch.concat((sx2, tx2), dim=0)
        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)
        sx1 = x1[:sx1.shape[0]]
        tx1 = x1[sx1.shape[0]:]
        sx2 = x2[:sx2.shape[0]]
        tx2 = x2[sx2.shape[0]:]
        # sx1 = self.embedding1(sx1)
        # tx1 = self.embedding1(tx1)
        # sx2 = self.embedding2(sx1)
        # tx2 = self.embedding2(tx2)

        selfAtt_encoder1 = self.selfAtt_encoder[0]
        selfAtt_encoder2 = self.selfAtt_encoder[1]
        crossAtt_encoder = self.crossAtt_encoder[0]

        out_df = selfAtt_encoder1(sx1, tx1, y, d)
        out_f = out_df[:sx1.shape[0]]
        f_list.append(out_f)
        df_list.append(out_df)
        out_df = selfAtt_encoder2(sx2, tx2, y, d)
        out_f = out_df[:sx2.shape[0]]
        f_list.append(out_f)
        df_list.append(out_df)

        out_df1, out_df2 = crossAtt_encoder(sx1.float().to(self.device), sx2.float().to(self.device),
                                            tx1.float().to(self.device), tx2.float().to(self.device),
                                            y, d)
        out_f1 = out_df1[:sx1.shape[0]]
        out_f2 = out_df2[:sx2.shape[0]]
        f_list.append(out_f1)
        f_list.append(out_f2)
        df_list.append(out_df1)
        df_list.append(out_df2)

        # res_feature = torch.cat(f_list, dim=1)
        res_feature_eeg, _ = self.attention_fusion_eeg(f_list[0], f_list[2])
        res_feature_eye, _ = self.attention_fusion_eye(f_list[1], f_list[3])
        res_feature = torch.cat([res_feature_eeg, res_feature_eye], dim=1)
        cls = torch.softmax(self.class_classifier(res_feature), dim=1)

        # domain_feature = torch.cat(df_list, dim=1)
        # domain_feature = ReverseLayerF.apply(domain_feature, alpha)
        # d_cls = torch.softmax(self.domain_classifier(domain_feature), dim=1)

        x = [x1, x2]
        x = torch.cat(x, dim=1)
        d_cls = torch.softmax(self.domain_classifier(x), dim=1)

        return f_list, res_feature, cls, d_cls

    def loss(self, cls, y, d_cls, d):
        # loss_reconstruction_class
        loss_se = 0
        for encoder in self.selfAtt_encoder:
            loss_se += encoder.loss()
        for encoder in self.crossAtt_encoder:
            loss_se += encoder.loss()

        # loss_class
        cross_entropy = nn.CrossEntropyLoss()
        loss_class = cross_entropy(cls, y.squeeze())

        # loss_domain
        domain_cross_entropy = nn.CrossEntropyLoss()
        loss_domain = domain_cross_entropy(d_cls, d.squeeze())

        # loss_con
        loss_con1 = self.attention_fusion_eeg.loss()
        loss_con2 = self.attention_fusion_eye.loss()
        loss_con = loss_con1 + loss_con2

        loss = 0.8 * loss_class + 0.2 * loss_domain + 0.1 * loss_se + 0.5 * loss_con

        return loss, loss_class, loss_domain, loss_se

    def valid(self, x_list):
        f_list = []
        x1 = x_list[0].float().to(self.device)
        x2 = x_list[1].float().to(self.device)

        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)

        selfAtt_encoder1 = self.selfAtt_encoder[0]
        selfAtt_encoder2 = self.selfAtt_encoder[1]
        crossAtt_encoder = self.crossAtt_encoder[0]

        out_f = selfAtt_encoder1.valid(x1)
        f_list.append(out_f)
        out_f = selfAtt_encoder2.valid(x2)
        f_list.append(out_f)

        out_f1, out_f2 = crossAtt_encoder.valid(x1, x2)
        f_list.append(out_f1)
        f_list.append(out_f2)

        # res_feature = torch.cat(f_list, dim=1)
        res_feature_eeg, _ = self.attention_fusion_eeg(f_list[0], f_list[2])
        res_feature_eye, _ = self.attention_fusion_eye(f_list[1], f_list[3])
        res_feature = torch.cat([res_feature_eeg, res_feature_eye], dim=1)
        cls = torch.softmax(self.class_classifier(res_feature), dim=1)

        return f_list, res_feature, cls

