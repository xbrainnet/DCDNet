import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MlpNet(nn.Module):
    def __init__(self, channels):
        super(MlpNet, self).__init__()
        assert isinstance(channels, list)
        self.encoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.encoder.add_module('linear%d' % i, nn.Linear(channels[i], channels[i + 1]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())
            # self.encoder.add_module('btn%d' % i, nn.BatchNorm1d(num_features=channels[i+1],affine=False))

    def forward(self, x):
        x = self.encoder(x)
        return x


class SelfExpression(nn.Module):
    def __init__(self, n, weight_c, class_block):
        super(SelfExpression, self).__init__()
        block = torch.ones((n, n),
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) - class_block
        block = torch.randn((n, n),
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) * block
        self.Coefficient = nn.Parameter(weight_c * block, requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class SelfAttention(nn.Module):
    """
    Build an Attention Layer
    """

    def __init__(self, d_model, num_attention_heads=1, dropout_prob=0.2):
        """
        hidden_size = 240, num_attention_heads = 1, dropout_prob = 0.2
        即隐层维度为240，注意力头设置为1个
        """

        super(SelfAttention, self).__init__()
        if d_model % num_attention_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (d_model, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(d_model / self.num_attention_heads)  # 每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        # all_head_size = 240 即等于hidden_size

        # query, key, value 的线性变换
        self.query = nn.Linear(d_model, d_model)  # 240,240
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

    def forward(self, hidden_states):
        # 线性变换
        hidden_states = hidden_states.unsqueeze(0)
        mixed_query_layer = self.query(hidden_states)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)  # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        attention_scores = attention_scores
        # 加上mask，将padding所在的表示直接-10000

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = context_layer.squeeze()
        return context_layer  # [bs, seqlen, 240] 得到输出


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads=1, dropout_prob=0.2):
        super(CrossAttention, self).__init__()
        if d_model % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (d_model, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(d_model / self.num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x2)
        mixed_value_layer = self.value(x2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = context_layer.squeeze()
        return context_layer


# 定义前馈神经网络层
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SelfAttentionNetLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_attention_heads=1, dropout_prob=0.4):
        super(SelfAttentionNetLayer, self).__init__()
        self.multihead_attention = SelfAttention(d_model, num_attention_heads, dropout_prob)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


# 定义一个完整的 Self-Attention Net
class SelfAttentionNet(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout=0.1):
        super(SelfAttentionNet, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.btn = nn.BatchNorm1d(d_model)
        self.encoder = nn.Sequential()
        for i in range(num_layers):
            self.encoder.add_module('self-Attention%d' % i, SelfAttentionNetLayer(d_model, d_model, num_heads, dropout))

    def forward(self, x):
        x = self.embedding(x)
        x = self.btn(x)
        for layer in self.encoder:
            residual = x  # 保存残差连接
            # x = self.btn(x)
            x = layer(x)
            x = x + residual  # 添加残差连接
        return x


class CrossAttentionNetLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_attention_heads=1, dropout_prob=0.1):
        super(CrossAttentionNetLayer, self).__init__()
        self.multihead_attention = CrossAttention(d_model, num_attention_heads, dropout_prob)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x1, x2):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention(x1, x2)
        x1 = x1 + self.dropout(attn_output)
        x1 = self.norm1(x1)

        # Feed-Forward
        ff_output = self.feed_forward(x1)
        x1 = x1 + self.dropout(ff_output)
        x1 = self.norm2(x1)

        return x1


# 定义一个完整的 Cross-Attention Net
class CrossAttentionNet(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout=0.1):
        super(CrossAttentionNet, self).__init__()
        self.embedding1 = nn.Linear(input_size[0], d_model)
        self.btn1 = nn.BatchNorm1d(d_model)
        self.embedding2 = nn.Linear(input_size[1], d_model)
        self.btn2 = nn.BatchNorm1d(d_model)
        self.encoder1 = nn.Sequential()
        self.encoder2 = nn.Sequential()
        for i in range(num_layers):
            self.encoder1.add_module('cross-Attention%d' % i,
                                     CrossAttentionNetLayer(d_model, d_model, num_heads, dropout))
            self.encoder2.add_module('cross-Attention%d' % i,
                                     CrossAttentionNetLayer(d_model, d_model, num_heads, dropout))

    def forward(self, x1, x2):
        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)
        x1 = self.btn1(x1)
        x2 = self.btn2(x2)
        for layer1, layer2 in zip(self.encoder1, self.encoder2):
            residual1 = x1  # 保存残差连接
            residual2 = x2  # 保存残差连接
            # x1 = self.btn1(x1)
            # x2 = self.btn2(x2)
            x1 = layer1(x1, residual2)
            x2 = layer2(x2, residual1)
            x1 = x1 + residual1  # 添加残差连接
            x2 = x2 + residual2  # 添加残差连接
        return x1, x2


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Classifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # 添加多个隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 可以选择添加激活函数
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        # 创建模型
        self.layers = nn.ModuleList(layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.softmax(x)
        return x


