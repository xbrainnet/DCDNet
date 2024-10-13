import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def GCCA_loss(H_list):
    r = 1e-4
    eps = 1e-8

    top_k = 15

    AT_list = []

    for H in H_list:
        assert torch.isnan(H).sum().item() == 0

        o_shape = H.size(0)  # N
        m = H.size(1)  # out_dim

        # H1bar = H1 - H1.mean(dim=1).repeat(m, 1).view(-1, m)
        Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
        assert torch.isnan(Hbar).sum().item() == 0

        A, S, B = Hbar.svd(some=True, compute_uv=True)

        A = A[:, :top_k]

        assert torch.isnan(A).sum().item() == 0

        S_thin = S[:top_k]

        S2_inv = 1. / (torch.mul(S_thin, S_thin) + eps)

        assert torch.isnan(S2_inv).sum().item() == 0

        T2 = torch.mul(torch.mul(S_thin, S2_inv), S_thin)

        assert torch.isnan(T2).sum().item() == 0

        T2 = torch.where(T2 > eps, T2, (torch.ones(T2.shape) * eps).to(H.device).double())

        T = torch.diag(torch.sqrt(T2)).float()

        assert torch.isnan(T).sum().item() == 0

        T_unnorm = torch.diag(S_thin + eps)

        assert torch.isnan(T_unnorm).sum().item() == 0

        AT = torch.mm(A, T)
        AT_list.append(AT)

    M_tilde = torch.cat(AT_list, dim=1)

    # print(f'M_tilde shape : {M_tilde.shape}')

    assert torch.isnan(M_tilde).sum().item() == 0

    Q, R = M_tilde.qr()

    assert torch.isnan(R).sum().item() == 0
    assert torch.isnan(Q).sum().item() == 0

    U, lbda, _ = R.svd(some=False, compute_uv=True)

    assert torch.isnan(U).sum().item() == 0
    assert torch.isnan(lbda).sum().item() == 0

    G = Q.mm(U[:, :top_k])
    assert torch.isnan(G).sum().item() == 0

    U = []  # Mapping from views to latent space

    # Get mapping to shared space
    views = H_list
    F = [H.shape[0] for H in H_list]  # features per view
    for idx, (f, view) in enumerate(zip(F, views)):
        _, R = torch.qr(view)
        Cjj_inv = torch.inverse((R.T.mm(R) + eps * torch.eye(view.shape[1], device=view.device)))
        assert torch.isnan(Cjj_inv).sum().item() == 0
        pinv = Cjj_inv.mm(view.T)

        U.append(pinv.mm(G))

    U1, U2 = U[0], U[1]
    _, S, _ = M_tilde.svd(some=True)

    assert torch.isnan(S).sum().item() == 0
    use_all_singular_values = False
    if not use_all_singular_values:
        S = S.topk(top_k)[0]
    corr = torch.sum(S)
    assert torch.isnan(corr).item() == 0
    loss = - corr
    return loss


def loss_cosine_sim(zf_list, y):
    class2_index = torch.argmax(y)
    class1_index = torch.argmax(y[:class2_index])
    loss_list = []
    for zf in zf_list:
        zf0 = zf[:class1_index, :]
        zf1 = zf[class1_index:class2_index, :]
        zf2 = zf[class2_index:, :]
        similarity_matrix0 = cosine_similarity(zf0)
        similarity_matrix1 = cosine_similarity(zf1)
        similarity_matrix2 = cosine_similarity(zf2)
        loss0 = torch.mean(1 - similarity_matrix0)
        loss1 = torch.mean(1 - similarity_matrix1)
        loss2 = torch.mean(1 - similarity_matrix2)
        loss_list.append((loss0 + loss1 + loss2) / 3)
    loss = sum(loss_list) / len(loss_list)
    return loss


def cosine_similarity(tensor):
    n, m = tensor.shape

    # 计算余弦相似度
    # 首先将每行单位化
    normalized_tensor = F.normalize(tensor, p=2, dim=1)

    # 使用矩阵乘法计算余弦相似度
    similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.t())

    # 现在 cosine_similarity[i, j] 包含第 i 行和第 j 行之间的余弦相似度
    # 如果你只关心对角线以下的相似度，可以使用以下代码
    # 保持对角线元素为零
    similarity_matrix = similarity_matrix - torch.diag(torch.diag(similarity_matrix))
    return similarity_matrix


def intra_modal_similarity(f1, f2):
    n, m = f1.shape

    # 首先将每行单位化
    normalized_tensor1 = F.normalize(f1, p=2, dim=1)
    normalized_tensor2 = F.normalize(f2, p=2, dim=1)

    # 使用矩阵乘法计算余弦相似度
    similarity_matrix = torch.mm(normalized_tensor1, normalized_tensor2.t())

    res = (n - torch.trace(similarity_matrix)) / n
    return res


def cross_modal_similarity(f1, f2):
    n, m = f1.shape

    # 首先将每行单位化
    normalized_tensor1 = F.normalize(f1, p=2, dim=1)
    normalized_tensor2 = F.normalize(f2, p=2, dim=1)

    # 使用矩阵乘法计算余弦相似度
    similarity_matrix = torch.mm(normalized_tensor1, normalized_tensor2.t())

    res = (torch.trace(similarity_matrix)) / n
    return res


def kl_divergence(matrix_p, matrix_q):
    # 将概率矩阵转换为概率分布
    distribution_p = F.softmax(matrix_p, dim=1)
    distribution_q = F.softmax(matrix_q, dim=1)

    # 使用KL散度公式计算KL散度
    kl_div = torch.sum(distribution_p * (torch.log(distribution_p) - torch.log(distribution_q)), dim=1)
    # 取平均值，得到整个矩阵的KL散度
    kl_div = torch.mean(kl_div)

    return kl_div


def intra_matrix_kl_divergence_loss(features):
    """
    计算每行特征之间的KL散度，并让除正对角线之外的KL散度最大。

    参数:
    - features: 二维numpy数组，大小为 [n, m]，表示输入的特征矩阵。

    返回:
    - loss: 损失值。
    """

    # 计算每行特征之间的KL散度
    kl_divergence = torch.nn.functional.kl_div(features, features, reduction='none')

    # 计算除正对角线之外的KL散度的均值
    loss = kl_divergence.mean()

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
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
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


def loss_function1(class_Coefficient, class_block, f_list, f_list1, zf_list, out_y, cls, y, w_c, w_s, w_b, w_sum):
    # loss_GCCA
    loss_gcca = GCCA_loss(f_list)

    # loss_reconstruction_class
    n, m = f_list1[0].shape[0], f_list1[0].shape[1]
    zero1 = torch.zeros((n, n)).to(device)
    zero2 = torch.zeros((n, m)).to(device)
    loss_coe = F.mse_loss(class_Coefficient, zero1, reduction='sum')
    loss_self = 0
    margin = 0.1
    for i in range(len(zf_list)):
        q = f_list1[i] - zf_list[i]
        # q[q < -margin] += margin
        # q[q > margin] -= margin
        # q[torch.abs(q) < margin] = 0
        loss_self += F.mse_loss(q, zero2, reduction='sum')
    loss_block = F.mse_loss(torch.mul(class_block, class_Coefficient), zero1, reduction='sum')
    loss_sum = torch.sum(torch.abs(class_Coefficient) - class_Coefficient)
    loss_reconstruction_class = w_c * loss_coe + w_s * loss_self + w_b * loss_block + w_sum * loss_sum
    loss_reconstruction = loss_reconstruction_class

    # loss_class
    cross_entropy = nn.CrossEntropyLoss()
    loss_class = cross_entropy(cls, y)

    # # loss_contrastive
    # if len(f_list) > 1:
    #
    #     margin1 = 0.95
    #     margin2 = -0.05
    #     loss_intra_modal = 0
    #     loss_intra_modal += intra_modal_similarity(f_list[0], f_list[2])
    #     loss_intra_modal += intra_modal_similarity(f_list[1], f_list[3])
    #     loss_intra_modal /= 2
    #     if loss_intra_modal < margin1:
    #         loss_intra_modal = loss_intra_modal
    #     else:
    #         loss_intra_modal = loss_intra_modal*5
    #
    #     loss_cross_modal = 0
    #     loss_cross_modal += cross_modal_similarity(f_list[2], f_list[3])
    #     if loss_cross_modal < margin2:
    #         loss_cross_modal = loss_cross_modal
    #     else:
    #         loss_cross_modal = loss_cross_modal*5
    #     loss_contrastive = loss_intra_modal + loss_cross_modal
    #
    #     loss_similarity = -kl_divergence(f_list[0], f_list[1])-kl_divergence(f_list[2], f_list[3])+kl_divergence(f_list[0], f_list[2])+kl_divergence(f_list[1], f_list[3])
    #
    # else:
    #     loss_similarity = torch.mean(1 - cosine_similarity((f_list[0])))
    #     loss_contrastive = -intra_matrix_kl_divergence_loss(f_list[0])
    loss_similarity = torch.mean(1 - cosine_similarity((f_list[0])))

    loss = loss_class
    # loss = loss_class
    # print("loss_class:", loss_class)
    # print("loss-GCCA:", loss_gcca/n)
    # print("loss-coe:", loss_coe/n)
    # print("loss-self:", loss_self/n)
    # print("loss-block:", loss_block/n)
    # print("-" * 10)

    # # print("loss-similarity:", loss_similarity)
    # print("loss_class:", loss_class/n)

    return loss, loss_class, loss_reconstruction, loss_similarity


def loss_function(class_Coefficient, domain_Coefficient, class_block, domain_block, f_list, tf_list, f_list1, zf_list,
                  f_list2, df_list, out_y, out_d, cls, y, d_cls, d, w_c, w_s, w_b, w_sum):
    # loss_GCCA
    # loss_gcca = GCCA_loss(f_list)

    # loss_reconstruction_class
    n, m = f_list1[0].shape[0], f_list1[0].shape[1]
    zero1 = torch.zeros((n, n)).to(device)
    zero2 = torch.zeros((n, m)).to(device)
    loss_coe = F.mse_loss(class_Coefficient, zero1, reduction='sum')
    loss_self = 0
    for i in range(len(zf_list)):
        loss_self += F.mse_loss(f_list1[i] - zf_list[i], zero2, reduction='sum')
    loss_block = F.mse_loss(torch.mul(class_block, class_Coefficient), zero1, reduction='sum')
    loss_sum = torch.sum(torch.abs(class_Coefficient) - class_Coefficient)
    loss_reconstruction_class = w_c * loss_coe + w_s * loss_self + w_b * loss_block + w_sum * loss_sum

    # loss_reconstruction_domain
    n, m = f_list2[0].shape[0], f_list2[0].shape[1]
    zero1 = torch.zeros((n, n)).to(device)
    zero2 = torch.zeros((n, m)).to(device)
    loss_coe = F.mse_loss(domain_Coefficient, zero1, reduction='sum')
    loss_self = 0
    for i in range(len(df_list)):
        loss_self += F.mse_loss(f_list2[i] - df_list[i], zero2, reduction='sum')
    # loss_block = F.mse_loss(domain_block, domain_Coefficient, reduction='sum')
    loss_block = F.mse_loss(torch.mul(domain_block, domain_Coefficient), zero1, reduction='sum')
    loss_sum = torch.sum(torch.abs(domain_Coefficient) - domain_Coefficient)
    loss_reconstruction_domain = w_c * loss_coe + w_s * loss_self + w_b * loss_block + w_sum * loss_sum

    loss_reconstruction = loss_reconstruction_class

    # loss_class
    class_cross_entropy = nn.CrossEntropyLoss()
    loss_class = class_cross_entropy(cls, y)

    # loss_domain
    domain_cross_entropy = nn.CrossEntropyLoss()
    loss_domain = domain_cross_entropy(d_cls, d)

    # loss_contrastive
    if len(f_list) > 1:

        margin1 = 0.95
        margin2 = -0.05
        loss_intra_modal = 0
        loss_intra_modal += intra_modal_similarity(f_list[0], f_list[2])
        loss_intra_modal += intra_modal_similarity(f_list[1], f_list[3])
        loss_intra_modal /= 2
        if loss_intra_modal < margin1:
            loss_intra_modal = loss_intra_modal
        else:
            loss_intra_modal = loss_intra_modal * 5

        loss_cross_modal = 0
        loss_cross_modal += cross_modal_similarity(f_list[2], f_list[3])
        if loss_cross_modal < margin2:
            loss_cross_modal = loss_cross_modal
        else:
            loss_cross_modal = loss_cross_modal * 5
        loss_contrastive = loss_intra_modal + loss_cross_modal

        loss_similarity = -kl_divergence(f_list[0], f_list[1]) - kl_divergence(f_list[2], f_list[3]) + kl_divergence(
            f_list[0], f_list[2]) + kl_divergence(f_list[1], f_list[3])

    else:
        loss_similarity = torch.mean(1 - cosine_similarity((f_list[0])))
        loss_contrastive = -intra_matrix_kl_divergence_loss(f_list[0])

    loss = loss_class + loss_domain + loss_reconstruction
    # loss = loss_class
    # print("loss_class:", loss_class)
    # print("loss-GCCA:", loss_gcca/n)
    # print("loss-coe:", loss_coe/n)
    # print("loss-self:", loss_self/n)
    # print("loss-block:", loss_block/n)
    # print("-" * 10)

    # # print("loss-similarity:", loss_similarity)
    # print("loss_class:", loss_class/n)

    return loss, loss_class, loss_domain, loss_reconstruction
    # return loss, loss_class, loss_reconstruction, loss_contrastive
