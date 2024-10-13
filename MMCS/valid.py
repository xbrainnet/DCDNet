import torch
from utils import divide
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def valid(val_loader, model, modalities):
    model.eval()
    f_val_list = []
    fy_val_list = []
    with torch.no_grad():
        for batch_index, (x_val, y_val) in enumerate(val_loader):
            f_val_list = []
            x_val.to(device)
            y_val.to(device)
            x_val = x_val.view(-1, x_val.shape[-1])
            y_val = y_val.view(-1, y_val.shape[-1])
            if model.multi_modal:
                x_val_list = divide(x_val, modalities)
            else:
                x_val_list = [x_val]

            _, _, _, _, _, _, _, _, _, cls, _ = model(x_val_list, valid=True)
            fy_val_list.append(torch.hstack((cls, y_val.to(device))))
        fy_val_list = torch.cat(fy_val_list, dim=0)

    return fy_val_list


def valid2(val_loader, model, modalities):
    model.eval()
    f_val_list = []
    fy_val_list = []
    with torch.no_grad():
        for batch_index, (x_val, y_val) in enumerate(val_loader):
            f_val_list = []
            x_val.to(device)
            y_val.to(device)
            x_val = x_val.view(-1, x_val.shape[-1])
            y_val = y_val.view(-1, y_val.shape[-1])
            if model.multi_modal:
                x_val_list = divide(x_val, modalities)
            else:
                x_val_list = [x_val]

            f_list, res_feature, cls = model(x_val_list, valid=True)
            fy_val_list.append(torch.hstack((cls, y_val.to(device))))
        fy_val_list = torch.cat(fy_val_list, dim=0)

    return fy_val_list


def valid_SVM(val_loader, model, modalities):
    model.eval()
    f_val_list = []
    fy_val_list = []
    # with torch.no_grad():
    for batch_index, (x_val, y_val) in enumerate(val_loader):
        f_val_list = []
        x_val.to(device)
        y_val.to(device)
        x_val = x_val.view(-1, x_val.shape[-1])
        y_val = y_val.view(-1, y_val.shape[-1])
        if model.multi_modal:
            x_val_list = divide(x_val, modalities)
        else:
            x_val_list = [x_val]

        _, _, _, _, _, _, _, res, _ = model(x_val_list, valid=True)
        fy_val_list.append(torch.hstack((res, y_val.to(device))))
    fy_val_list = torch.cat(fy_val_list, dim=0)

    return fy_val_list
