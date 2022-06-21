import sys
import torch
import argparse
from OverlapDataset import OverlapDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import data
from tqdm import tqdm
import torch.nn.functional as F
import models
import torch.nn as nn
from math import sqrt
from torchmetrics import ConfusionMatrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="train dataset file")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate for Adam (default: 0.001)", metavar="")
    parser.add_argument("--cuda", action="store_true", help="use CUDA for model training")
    parser.add_argument("--type", type=str, default="B", help="Model type: A, B or C")
    parser.add_argument("--extended_cigar", action="store_true", help="Use extended cigar strings with X and = instead of M")
    parser.add_argument("--fn", action="store_true", help="Classify false negatives")
    return parser.parse_args()

def train_model(model, train_dl, val_dl, device, epochs, lr, model_type, nclasses, fig_output):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    weights = train_dl.dataset.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_loss = None
    classes = ('FP', 'TP', 'FN') if nclasses == 3 else ('FP', 'TP')
    start = time.time()

    for i in tqdm(range(epochs)):
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0
        for x, y, l in train_dl:
            optimizer.zero_grad()
            y = y.to(device)
            cigar, counts, num, l = x[0].to(device), x[1].to(device), x[2].to(device), l.to(device)
            if (model_type == "B"):           
                y_pred = model(cigar, counts)["logits"]
            elif(model_type == "A"):
                y_pred = model(num)
            elif(model_type == "C"):
                y_pred = model(cigar, counts, num)
            else:
                raise ValueError("Invalid model type")
#             print(y_pred.shape)
#             print(y.shape)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
        val_loss, confusion_matrix = validation_metrics(model, val_dl, device, model_type, nclasses)
        if i % 5 == 1:
            if (best_loss == None or best_loss > val_loss):
                best_loss = val_loss
                conf_perc = confusion_matrix / confusion_matrix.sum(dim=1).reshape(-1,1) * 100
                df_cm = pd.DataFrame(conf_perc.cpu(), index = [j for j in classes],
                     columns = [j for j in classes])
                plt.figure(figsize = (12,7))
                sn.heatmap(df_cm, annot=True, fmt='g')
                plt.xlabel('predictions')
                plt.ylabel('labels')
                plt.savefig(fig_output)
                print("Best acc: " + str(confusion_matrix.diag().sum()/confusion_matrix.sum()))
        if i % 20 == 1:
            print("train loss %.3f, train acc %.3f, val loss %.3f, val accuracy %.3f" % (sum_loss/total, correct/total, val_loss, confusion_matrix.diag().sum()/confusion_matrix.sum()))
#             print(confusion_matrix)
    end = time.time()
    print("Training time: " + str(end - start))

def validation_metrics (model, valid_dl, device, model_type, nclasses):
    model.eval()
    
    sum_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    confmat = ConfusionMatrix(num_classes=nclasses).to(device)
    confusion_matrix = torch.zeros(nclasses, nclasses).to(device)
    
    for x, y, l in valid_dl:
        y = y.long().to(device)
        cigar, counts, num, l = x[0].to(device), x[1].to(device), x[2].to(device), l.to(device)
        if (model_type == "B"):           
            y_hat = model(cigar, counts)["logits"]
        elif(model_type == "A"):
            y_hat = model(num)
        elif(model_type == "C"):
            y_hat = model(cigar, counts, num)
        else:
            raise ValueError("Invalid model type")
        loss = criterion(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        confusion_matrix += confmat(pred, y)
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/confusion_matrix.sum(), confusion_matrix

def main():
    
    args = parse_arguments()
    
    if (args.cuda and not torch.cuda.is_available()):
        raise ValueError("CUDA is not avaialble")
    else:
        device = torch.device("cuda" if args.cuda else "cpu")
   
    X, y = OverlapDataset.from_file(dataset=args.dataset, extended_cigar=args.extended_cigar, fn=args.fn)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    train_ds = OverlapDataset(X_train, y_train)
    valid_ds = OverlapDataset(X_valid, y_valid)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=data.pad_collate_fn)
    val_dl = DataLoader(valid_ds, batch_size=64, collate_fn=data.pad_collate_fn)
    
    vocab_size = len(data.vocab2index_extended) if args.extended_cigar else len(data.vocab2index)
    
    nclasses = 3 if args.fn else 2

    if args.type == "A":
        model = models.ANN(2, nclasses)
    elif args.type == "B":
        model = models.BERT(vocab_size, 3, 30, nclasses)
    elif args.type == "C":
        model = models.COMB(vocab_size, 3, 30, 2, nclasses)
    else:
        raise ValueError("Invalid model type")        

    model.to(device)
    
    fig_output = args.dataset.split("/")[1].split(".")[0] + "_" + args.type
    if (not args.extended_cigar):
        fig_output += "_M"
    if (args.fn):
        fig_output += "_fn"
    fig_output = "figures/" + fig_output + ".png"
    
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
#     val_loss, val_acc = validation_metrics(model, val_dl, device, args.type)
#     print("val loss %.3f, val accuracy %.3f" % (val_loss, val_acc))

    train_model(model, train_dl, val_dl, device, epochs=500, lr=args.learning_rate, model_type=args.type, nclasses=nclasses, fig_output=fig_output)

if __name__ == "__main__":
    main()
