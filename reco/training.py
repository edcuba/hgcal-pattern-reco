import torch
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from torch_geometric.data import Data

from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score, balanced_accuracy_score, roc_auc_score


def train_edge_pred(model, device, optimizer, loss_func, train_dl):
    train_loss = 0.0
    model.train()

    train_true_seg = []
    train_pred_seg = []

    for data in train_dl:

        batch_size = len(data)
        data = data.to(device)

        optimizer.zero_grad()

        seg_pred = model(data.x, data.edge_index)
        y_true = data.y

        if data.mask is not None:
            seg_pred = seg_pred[data.mask]
            y_true = y_true[data.mask]


        loss = loss_func(seg_pred.view(-1, 1), y_true.view(-1, 1).type(torch.float))

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size

        seg_np = y_true.cpu().numpy()
        pred_np = seg_pred.detach().cpu().numpy()

        train_true_seg.append(seg_np.reshape(-1))
        train_pred_seg.append(pred_np.reshape(-1))

    train_true_cls = np.concatenate(train_true_seg)
    train_pred_cls = np.concatenate(train_pred_seg)

    return train_loss, train_true_cls, train_pred_cls


@torch.no_grad()
def test_edge_pred(model, device, loss_func, test_dl, obj_cond=False):
    test_loss = 0.0
    model.eval()

    test_true_seg = []
    test_pred_seg = []
    for data in test_dl:

        batch_size = len(data)
        data = data.to(device)

        seg_pred = model(data.x, data.edge_index)

        y_true = data.y
        if data.mask is not None:
            seg_pred = seg_pred[data.mask]
            y_true = y_true[data.mask]

        loss = loss_func(seg_pred.view(-1, 1), y_true.view(-1, 1).type(torch.float))
        test_loss += loss.item() * batch_size

        seg_np = y_true.cpu().numpy()
        pred_np = seg_pred.detach().cpu().numpy()
        test_true_seg.append(seg_np.reshape(-1))
        test_pred_seg.append(pred_np.reshape(-1))

    test_true_cls = np.concatenate(test_true_seg)
    test_pred_cls = np.concatenate(test_pred_seg)

    return test_loss, test_true_cls, test_pred_cls



def split_geo_train_test(ds, batch_size=64, test_set_fraction=0.1):

    ds_size = len(ds)
    test_set_size = ds_size // int(1. / test_set_fraction)
    train_set_size = ds_size - test_set_size

    print(f"Train set: {train_set_size}, Test set: {test_set_size}")

    train_set, test_set = random_split(ds, [train_set_size, test_set_size])

    # this is very nice - handles the dimensions automatically
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_dl, test_dl


@torch.no_grad()
def roc_auc(model, device, test_dl, truth_threshold=0.7):

    y_pred = []
    y_true = []

    model.eval()

    for data in test_dl:

        if isinstance(data, Data):
            # graph dataset
            l = data.y.reshape(-1)

            data = data.to(device)
            ei = data.edge_index
            if ei is not None:
                model_pred = model(data.x, ei, data.batch)
            else:
                model_pred = model(data.x, data.batch)[:,0]
        else:
            b, l = data
            model_pred = model(b.to(device))
            l = l.reshape(-1)

        y_pred += model_pred.detach().cpu().reshape(-1).tolist()
        y_true += (l > truth_threshold).type(torch.int).tolist()

    return roc_auc_score(y_true, y_pred)


@torch.no_grad()
def precision_recall_curve(model, device, test_dl, beta=0.5, truth_threshold=0.7, step=1, focus_metric="fbeta"):
    """
    Plot the precision/recall curve depending on the decision threshold

    There are two kinds of threshold here:
    - model (0-1 whether we want to cluster this trackster or not)
    - simtrackster, what we consider a relevant trackster (based on the score - usually 0.8)
    """
    model.eval()
    th_values = [i / 100. for i in range(1, 100, step)]

    result = {
        "precision": [],
        "recall": [],
        "fbeta": [],
        "b_acc": [],
    }
    cm = []

    for th in th_values:
        pred = []
        lab = []
        for data in test_dl:

            if isinstance(data, Data):
                # graph dataset
                l = data.y.reshape(-1)

                data = data.to(device)
                ei = data.edge_index
                if ei is not None:
                    model_pred = model(data.x, ei, data.batch)
                else:
                    model_pred = model(data.x, data.batch)[:,0]
            else:
                b, l = data
                model_pred = model(b.to(device))
                l = l.reshape(-1)

            model_pred = model_pred.detach().cpu().reshape(-1)

            pred += (model_pred > th).type(torch.int).tolist()
            lab += (l > truth_threshold).type(torch.int).tolist()

        result["precision"].append(precision_score(lab, pred, zero_division=0))
        result["recall"].append(recall_score(lab, pred))
        result["fbeta"].append(fbeta_score(lab, pred, beta=beta))
        result["b_acc"].append(balanced_accuracy_score(lab, pred))
        cm.append(confusion_matrix(lab, pred).ravel())

    plt.figure()
    for k, v in result.items():
        plt.plot(th_values, v, label=k)

    plt.xlabel("Threshold")
    plt.legend()
    plt.show()

    bi = np.argmax(result[focus_metric])
    decision_th = th_values[bi]

    tn, fp, fn, tp = cm[bi]
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"TH: {decision_th}", " ".join([f"{k}: {v[bi]:.3f}" for k, v in result.items()]))


def train_mlp(model, device, opt, loader, loss_obj):
    epoch_loss = 0
    for batch, labels in loader:
        # reset optimizer and enable training mode
        opt.zero_grad()
        model.train()

        # move data to the device
        batch = batch.to(device)
        labels = labels.to(device)

        # get the prediction tensor
        z = model(batch).reshape(-1)

        # compute the loss
        loss = loss_obj(z, labels)
        epoch_loss += loss

        # back-propagate and update the weight
        loss.backward()
        opt.step()

    return float(epoch_loss)
