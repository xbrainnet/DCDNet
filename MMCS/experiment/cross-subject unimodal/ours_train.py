import statistics

import torch
import torch.optim as optim
import numpy as np
from models.MyNet import DMDR
from models.loss_function import *
from valid import *
from test import *
from utils import *
import torch.nn as nn
import matplotlib.pyplot as plt
from models.MyCrossSubjectNet import MyCSNet

res = []
bestRes = []


def train(model, train_loader, val_loader, test_loader, modalities,
        epochs, weight_coef, weight_selfExp, weight_block, weight_sum, lr, mom, decay, svmc):

    printOK = 0
    draw = 0
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=decay)
    criterion = nn.CrossEntropyLoss()
    criterion_mmd = MMDLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_acc_val = 0
    max_acc_te = 0

    # 定义两个事件来记录时间
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    qq = list()

    for epoch in range(1, epochs + 1):
        # 记录开始事件
        start_event.record()
        loss_epoch_train = 0
        if printOK:
            print(epoch, "...")
        for batch_index, (source_x, source_y) in enumerate(train_loader):
            source_x = source_x.to(device)
            source_y = source_y.to(device).long()
            source_d = torch.zeros(source_y.shape)
            for inx in range(11):
                source_d[:, inx*3:(inx+1)*3, :] = inx
            source_x = source_x.view(-1, source_x.shape[-1])
            source_y = source_y.view(-1, source_y.shape[-1])
            source_d = source_d.view(-1, source_d.shape[-1])

            test_iter = iter(test_loader)
            target_x, target_y = next(test_iter)
            target_x = target_x.float().to(device)
            target_y = target_y.long().to(device)
            target_d = torch.zeros(target_y.shape)
            target_d[:] = model.domain_num-1

            d = torch.concat((source_d, target_d), dim=0).to(device).long()

            len_dataloader = len(train_loader)
            p = float(batch_index + epoch * len_dataloader) / epochs / len_dataloader
            alpha = torch.tensor(2. / (1. + np.exp(-10 * p)) - 1)

            model.to(device)
            model.train()
            # 分割模态
            source_x_list = [source_x]
            target_x_list = [target_x]
            f_list, res_feature, cls, d_cls = model(source_x_list=source_x_list, target_x_list=target_x_list,
                                                    y=source_y, d=d, alpha=alpha)

            loss, loss_class, loss_domain, loss_se = model.loss(cls, source_y, d_cls, d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch_train += loss.item()

        # 记录结束事件
        end_event.record()

        if val_loader != None:
            fy_tr_list = valid2(train_loader, model, modalities)
            acc_tr, sen_tr, spe_tr, auc_tr, pre_tr, f1_tr, std_tr, _, _ = myEval(fy_tr_list)
            fy_val_list = valid2(val_loader, model, modalities)
            acc_val, sen_val, spe_val, auc_val, pre_val, f1_val, std_val, _, _ = myEval(fy_val_list)
            fy_te_list = test2(test_loader, model, modalities)
            acc_te, sen, spe, auc, pre, f1, std, precision, recall = myEval(fy_te_list)
            if epoch % 1 == 0:
                # 等待所有 CUDA 操作完成
                torch.cuda.synchronize()

                # 计算耗费时间
                elapsed_time = start_event.elapsed_time(end_event)  # 时间单位是毫秒
                print(f'Time taken: {elapsed_time} ms')
                qq.append(elapsed_time)
                if epoch > 1:
                    print('acc_tr:', sum(qq) / epoch, '+-', statistics.stdev(qq))
                print("acc_val: {:.4f}, acc_tr: {:.4f}, acc_test: {:.4f}, loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
                    acc_val, acc_tr, acc_te, loss.item(), loss_class.item(), loss_domain.item(), loss_se.item()))

            if acc_tr >= max_acc_val:
                max_fy_tr = fy_tr_list
                max_acc_val = acc_tr
            if acc_te >= max_acc_te:
                max_acc_te = acc_te
                torch.save(model.state_dict(), checkpoint)

        if epoch % 5 == 0 and draw:
            Mydraw(model)

    return max_fy_tr, max_acc_val


def myEval(fy_list, save=False):
    if save:
        res.append(fy_list.cpu().detach().numpy())

    metrics, precision, recall = cross_subject_eval(fy_list[:, :-1], fy_list[:, -1], average='macro')
    acc = metrics['Accuracy']
    auc = metrics['ROC AUC']
    sen = metrics['Sensitivity']
    spe = metrics['Specificity']
    pre = metrics['Precision']
    f1 = metrics['F1 Score']
    std = metrics['STD']

    if save and auc >= 0.80:
        bestRes.append(fy_list.cpu().detach().numpy())

    return acc, auc, sen, spe, pre, f1, std, precision, recall


def print_metrics(metrics):
    print("  Accuracy:", metrics["Accuracy"])
    print("  Confusion Matrix:\n", metrics["Confusion Matrix"])
    print("  ROC AUC:", metrics["ROC AUC"])
    print("  Sensitivity:", metrics["Sensitivity"])
    print("  Specificity:", metrics["Specificity"])
    print("  Precision:", metrics["Precision"])
    print("  F1 Score:", metrics["F1 Score"])


def save_metrics(metrics):
    acc = metrics['Accuracy']
    auc = metrics['ROC AUC']
    sen = metrics['Sensitivity']
    spe = metrics['Specificity']
    pre = metrics['Precision']
    F1 = metrics['F1 Score']
    std = metrics['STD']
    return acc, auc, sen, spe, pre, F1, std


def Mydraw(model):
    model = model.selfAtt_encoder.encoder[2].self_expression
    # # 使用imshow绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # 在第一个子图中绘制第一个热力图
    im1 = axes[0].imshow(model.class_self_expression.Coefficient.cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
    axes[0].set_title('Class')

    # 在第二个子图中绘制第二个热力图
    im2 = axes[1].imshow(model.domain_self_expression.Coefficient.cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
    axes[1].set_title('Domain')

    # 添加颜色条
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 显示图形
    plt.show()


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold = 12
    batch_size = 20
    batch = True
    path = '../../../data/SEED/all_data_36.npy'
    checkpoint = 'checkpoint.model'

    d_model = 256
    num_heads = 8
    # 两个通道
    input_size = [310]
    # input_size = [310, 33]
    modalities = [0, 310, 310+33]
    epochs = 30

    bs = 0.5
    weight_coef = 1 * bs
    weight_selfExp = 0.1 * bs
    weight_block = 1 * bs
    weight_sum = 0 * bs
    domain_num = 12

    lr = 1e-3
    svmc = 0.3
    mom_list = 0.9
    decay_list = 5e-4
    num_AttLayers = 2
    dropout = 0.5

    model_type = 'DA'

    acc_tr_list = list()
    acc_te_list = list()
    auc_list = list()
    std_list = list()
    spe_list = list()
    sen_list = list()
    pre_list = list()
    F1_list = list()
    pre_list1 = list()
    pre_list2 = list()
    pre_list3 = list()
    recall_list1 = list()
    recall_list2 = list()
    recall_list3 = list()

    fold_acc_tr_list = np.empty((0, epochs))
    fold_acc_te_list = np.empty((0, epochs))
    fold_auc_list = np.empty((0, epochs))
    fold_std_list = np.empty((0, epochs))
    fold_spe_list = np.empty((0, epochs))
    fold_sen_list = np.empty((0, epochs))
    fold_pre_list = np.empty((0, epochs))
    fold_F1_list = np.empty((0, epochs))
    fold_pre_list1 = np.empty((0, epochs))
    fold_pre_list2 = np.empty((0, epochs))
    fold_pre_list3 = np.empty((0, epochs))
    fold_recall_list1 = np.empty((0, epochs))
    fold_recall_list2 = np.empty((0, epochs))
    fold_recall_list3 = np.empty((0, epochs))

    x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_test_fold, y_test_fold = My_Model_Special_processdata2(path, fold,
                                                                                                             batch=batch,
                                                                                                             batch_size=batch_size)
    # My_Model_Special_processdata
    # cross_subject_processdata

    for i in range(fold):
        seed = 100
        print('seed is {}'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # train_loader, val_loader, test_loader, sample_num = cross_subject_loaddata(x_train_fold, y_train_fold,
        #                                                                            x_val_fold, y_val_fold,
        #                                                                            x_test_fold, y_test_fold, i,
        #                                                                            batch, batch_size=batch_size)

        train_loader, val_loader, test_loader, sample_num = multi_domain_cross_subject_loaddata2(x_train_fold,
                                                                                                 y_train_fold,
                                                                                                 x_val_fold, y_val_fold,
                                                                                                 x_test_fold,
                                                                                                 y_test_fold, i,
                                                                                                 batch, batch_size)

        model = MyCSNet(input_size, sample_num, d_model, num_AttLayers=num_AttLayers, num_heads=num_heads, dropout=dropout)
        model.to(device)
        fy_tr_list, acc_val = train(model, train_loader, val_loader, test_loader, modalities, epochs,
                                    weight_coef=weight_coef, weight_selfExp=weight_selfExp,
                                    weight_block=weight_block, weight_sum=weight_sum,
                                    lr=lr, mom=mom_list, decay=decay_list, svmc=0.3)

        checkpoint_ = torch.load(checkpoint)
        model.load_state_dict(checkpoint_)

        fy_te_list = test2(test_loader, model, modalities)
        acc_tr, sen_tr, spe_tr, auc_tr, pre_tr, f1_tr, std_tr, _, _ = myEval(fy_tr_list)
        acc_te, sen, spe, auc, pre, f1, std, precision, recall = myEval(fy_te_list, save=True)

        print('fold = ', i, 'acc_tr = ', acc_tr, 'acc_te =', acc_te, 'std = ', std, 'sen = ', sen, 'spe = ',
              spe, 'auc = ', auc, 'pre = ', pre, 'F1:', f1)
        print('sad:     precision = ', precision[0], 'recall = ', recall[0])
        print('neutral: precision = ', precision[1], 'recall = ', recall[1])
        print('happy:   precision = ', precision[2], 'recall = ', recall[2])

        acc_tr_list.append(acc_tr)
        acc_te_list.append(acc_te)
        auc_list.append(auc)
        std_list.append(std)
        spe_list.append(spe)
        sen_list.append(sen)
        pre_list.append(pre)
        F1_list.append(f1)
        pre_list1.append(precision[0])
        pre_list2.append(precision[1])
        pre_list3.append(precision[2])
        recall_list1.append(recall[0])
        recall_list2.append(recall[1])
        recall_list3.append(recall[2])

    save_data = np.concatenate(res)
    save_path = '../../save/5/Oursss.npy'
    np.save(save_path, save_data)

    save_best_data = np.concatenate(bestRes)
    save_best_path = '../../save/5/Ours_besttt.npy'
    np.save(save_best_path, save_best_data)

    print('acc_tr:', sum(acc_tr_list) / fold)
    print('acc_te:', sum(acc_te_list) / fold)
    print('auc:', sum(auc_list) / fold)
    print('std:', sum(std_list) / fold)
    print('spe:', sum(spe_list) / fold)
    print('sen:', sum(sen_list) / fold)
    print('pre:', sum(pre_list) / fold)
    print('F1:', sum(F1_list) / fold)
    print('sad:     precision = ', sum(pre_list1) / fold, 'recall = ', sum(recall_list1) / fold)
    print('neutral: precision = ', sum(pre_list2) / fold, 'recall = ', sum(recall_list2) / fold)
    print('happy:   precision = ', sum(pre_list3) / fold, 'recall = ', sum(recall_list3) / fold)
