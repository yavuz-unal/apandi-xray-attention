"""
Training script for pediatric appendicitis classification.

Trains EfficientNet-B0 with different attention mechanisms (No Attention, SE, CBAM, CA, PCCA)
using 5-fold stratified cross-validation with early stopping.

Usage:
    python train.py --model effnet_se --data_root /path/to/data
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

from models import build_model

SEED = 42
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4
WD = 1e-4
N_SPLITS = 5
PATIENCE = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class XRayDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(row['klasor'], row['dosya'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def load_dataset(data_root, hasta_dir='hasta-crop-unsharp', kontrol_dir='kontrol-crop-unsharp'):
    rows = []
    for klasor, label in [(os.path.join(data_root, hasta_dir), 1),
                          (os.path.join(data_root, kontrol_dir), 0)]:
        for f in sorted(os.listdir(klasor)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                rows.append({'klasor': klasor, 'dosya': f, 'label': label})
    return pd.DataFrame(rows)


def train_one_fold(model, train_loader, val_loader, device, class_weights, fold, save_dir, model_name):
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0.0
    best_probs, best_labels = None, None
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_losses.append(running_loss / len(train_loader.dataset))
        scheduler.step()

        model.eval()
        epoch_probs, epoch_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                val_loss += criterion(output, labels).item() * imgs.size(0)
                probs = torch.softmax(output, 1)[:, 1]
                epoch_probs.extend(probs.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
        val_losses.append(val_loss / len(val_loader.dataset))

        auc = roc_auc_score(epoch_labels, epoch_probs)
        if auc > best_auc:
            best_auc = auc
            best_probs = epoch_probs.copy()
            best_labels = epoch_labels.copy()
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'{model_name}_fold{fold+1}.pth'))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    return best_auc, best_probs, best_labels, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['effnet_base', 'effnet_se', 'effnet_cbam', 'effnet_ca', 'effnet_pcca',
                                 'densenet_base', 'resnet_base', 'swin_base', 'convnext_base'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    df = load_dataset(args.data_root)
    print(f'Hasta: {(df.label==1).sum()} | Kontrol: {(df.label==0).sum()} | Toplam: {len(df)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_transform, val_transform = get_transforms()
    g = torch.Generator()
    g.manual_seed(SEED)

    fold_aucs = []
    all_labels, all_probs = [], []
    all_loss_curves = []

    set_seed(SEED)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f'\nFold {fold+1}/{N_SPLITS}')
        set_seed(SEED + fold)

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_loader = DataLoader(
            XRayDataset(train_df, train_transform),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
            worker_init_fn=seed_worker, generator=g
        )
        val_loader = DataLoader(
            XRayDataset(val_df, val_transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        model = build_model(args.model).to(device)
        class_weights = compute_class_weight('balanced',
                                              classes=np.array([0, 1]),
                                              y=train_df['label'].values)

        best_auc, best_probs, best_labels, tl, vl = train_one_fold(
            model, train_loader, val_loader, device, class_weights,
            fold, args.save_dir, args.model
        )

        fold_aucs.append(best_auc)
        all_labels.extend(best_labels)
        all_probs.extend(best_probs)
        all_loss_curves.append((tl, vl))
        print(f'  Fold {fold+1} best AUC: {best_auc:.3f}')

    agg_auc = roc_auc_score(all_labels, all_probs)
    print(f'\nMean Fold AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}')
    print(f'Aggregated AUC: {agg_auc:.3f}')

    progress = {
        'fold_aucs': [float(x) for x in fold_aucs],
        'all_labels': [int(x) for x in all_labels],
        'all_probs': [float(x) for x in all_probs],
        'all_loss_curves': all_loss_curves
    }
    with open(os.path.join(args.results_dir, f'progress_{args.model}.json'), 'w') as f:
        json.dump(progress, f)


if __name__ == '__main__':
    main()
