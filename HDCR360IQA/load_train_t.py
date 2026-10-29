from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def train_oiqa(epoch, net, criterion, optimizer, train_loader):
    net.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch+1}")

    for batch in train_loader:
        imgs = batch['d_img_org'].cuda()
        labels = batch['score'].cuda().squeeze()

        optimizer.zero_grad()
        

        final_score = net(imgs)
        loss = criterion(final_score, labels)


        loss.backward()
        

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(final_score.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()

    raw_mse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    rho_s = spearmanr(all_preds, all_labels)[0]
    rho_p = pearsonr(all_preds, all_labels)[0]

    return total_loss / len(train_loader), rho_s, rho_p, raw_mse


def eval_oiqa(config, epoch, net, criterion, test_loader):
    losses = []
    net.eval()
    pred_epoch = []
    labels_epoch = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Eval Epoch {epoch + 1}"):
            imgs = data['d_img_org'].cuda()
            labels = torch.squeeze(data['score'].type(torch.FloatTensor)).cuda()


            final_score = net(imgs)
            loss = criterion(final_score, labels)

            losses.append(loss.item())
            pred_np = final_score.float().data.cpu().numpy().flatten()
            labels_n = labels.float().data.cpu().numpy().flatten()
            pred_epoch.extend(pred_np)
            labels_epoch.extend(labels_n)

    pred_epoch = np.array(pred_epoch)
    labels_epoch = np.array(labels_epoch)

    raw_mse = np.sqrt(np.mean((pred_epoch - labels_epoch) ** 2))
    rho_s, _ = spearmanr(pred_epoch, labels_epoch) if len(pred_epoch) > 0 else (0.0, 0.0)
    rho_p, _ = pearsonr(pred_epoch, labels_epoch) if len(pred_epoch) > 0 else (0.0, 0.0)

    print('Epoch: ===== loss:{:.4f} ===== SRCC:{:.4f} ===== PLCC:{:.4f} ===== RAW_MSE:{:.6f}'.format(
        np.mean(losses), rho_s, rho_p, raw_mse))

    return np.mean(losses), rho_s, rho_p, raw_mse


