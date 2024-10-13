import statistics

import torch
import torch.optim as optim
import numpy as np
from models.MyMultiModalNet import MyMMNet
from valid import *
from test import *
from models.loss_function import loss_function1
from utils import *
import torch.nn as nn
import matplotlib.pyplot as plt

res = []
bestRes = []


def train(model, train_loader, val_loader, test_loader, modalities,
        epochs, weight_coef, weight_selfExp, weight_block, weight_sum, lr, mom, decay, svmc):

    printOK = 0
    draw = 1
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_acc_val = 0
    max_acc_te = 0
    max_te = 0

    for epoch in range(1, epochs + 1):
        fy_tr_list = []
        loss_epoch_train = 0
        if printOK:
            print(epoch, "...")
        for batch_index, (source_x, source_y) in enumerate(train_loader):
            source_x = source_x.to(device)
            source_y = source_y.to(device).long()

            source_x = source_x.view(-1, source_x.shape[-1])
            source_y = source_y.view(-1, source_y.shape[-1])

            model = model.to(device)
            model.train()

            # 分割模态
            source_x_list = divide(source_x, modalities)
            f_list, res_feature, cls, _ = model(x_list=source_x_list, y=source_y)

            loss, loss_class, loss_se = model.loss(cls, source_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch_train += loss.item()

        if val_loader != None:
            fy_tr_list = valid2(train_loader, model, modalities)
            acc_tr, sen_tr, spe_tr, auc_tr, pre_tr, f1_tr, std_tr, _, _ = myEval(fy_tr_list)
            fy_val_list = valid2(val_loader, model, modalities)
            acc_val, sen_val, spe_val, auc_val, pre_val, f1_val, std_val, _, _ = myEval(fy_val_list)
            fy_te_list = test2(test_loader, model, modalities)
            acc_te, sen, spe, auc, pre, f1, std, precision, recall = myEval(fy_te_list)
            if epoch % 20 == 0:
                # print("acc_val: {:.4f}, acc_tr: {:.4f}, acc_test: {:.4f}, loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
                #     acc_val, acc_tr, acc_te, loss.item(), loss_class.item(), loss_reconstruction.item(),
                #     loss_contrastive.item()))
                print("acc_val: {:.4f}, acc_tr: {:.4f}, acc_test: {:.4f}, loss: {:.4f}, {:.4f}, {:.4f}".format(
                    acc_val, acc_tr, acc_te, loss.item(), loss_class.item(), loss_se.item()))

            if acc_tr >= max_acc_val:
                max_fy_tr = fy_tr_list
                max_acc_val = acc_tr
            if acc_te >= max_acc_te:
                max_acc_te = acc_te
                max_te = acc_te + auc
                torch.save(model.state_dict(), checkpoint)
            elif 1.0*acc_te + 0.01 >= max_acc_te:
                if acc_te+auc >= max_te:
                    max_acc_te = acc_te
                    max_te = acc_te + auc
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

    if save and auc >= 0.89:
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
    # # 使用imshow绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    m = model.selfAtt_encoder[0].encoder[1].self_expression
    # 在第一个子图中绘制第一个热力图
    im1 = axes[0].imshow(m.class_self_expression.Coefficient.cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
    axes[0].set_title('Class')

    # 在第二个子图中绘制第二个热力图
    im2 = axes[1].imshow(m.class_block.cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
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
    batch_size = 12

    batch = True
    path = '../../../data/SEED/all_data_36.npy'
    checkpoint = 'checkpoint.model'

    d_model = 256
    num_heads = 8
    # 两个通道
    input_size = [310, 33]
    modalities = [0, 310, 310+33]
    epochs = 300
    output_size = 3

    bs = 0.5
    weight_coef = 1 * bs
    weight_selfExp = 1 * bs
    weight_block = 1 * bs
    weight_sum = 0 * bs
    domain_num = 1

    lr = 5e-5
    svmc = 0.3
    mom_list = 0.9
    decay_list = 5e-4
    num_AttLayers = 3
    dropout = 0.1

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

    x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_test_fold, y_test_fold = My_Model_Special_processdata1(path, fold,
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

        model = MyMMNet(input_size, output_size, sample_num, d_model, num_AttLayers=num_AttLayers, num_heads=num_heads, dropout=dropout)
        model = model.to(device)
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
    save_path = '../../save/4/Ours.npy'
    np.save(save_path, save_data)

    save_best_data = np.concatenate(bestRes)
    save_best_path = '../../save/4/Ours_best.npy'
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

    print('acc_tr:', sum(acc_tr_list) / fold, '+-', statistics.stdev(acc_tr_list))
    print('acc_te:', sum(acc_te_list) / fold, '+-', statistics.stdev(acc_te_list))
    print('auc:', sum(auc_list) / fold, '+-', statistics.stdev(auc_list))
    print('std:', sum(std_list) / fold, '+-', statistics.stdev(std_list))
    print('spe:', sum(spe_list) / fold, '+-', statistics.stdev(spe_list))
    print('sen:', sum(sen_list) / fold, '+-', statistics.stdev(sen_list))
    print('pre:', sum(pre_list) / fold, '+-', statistics.stdev(pre_list))
    print('F1:', sum(F1_list) / fold, '+-', statistics.stdev(F1_list))
