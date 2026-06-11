
# -*- coding: utf-8 -*-
"""
Adapted ST-DADGAT for ASD EEG severity screening.

Data format:
    input_dir/
        subject_1/
            01_xxx_game1.mat
            02_xxx_game2.mat
            ...
        subject_2/
            ...
    label.csv:
        folder,label
        10,1
        11,1
        ...

Default task:
    0 = TD
    1 = Mild ASD
    2 = Moderate ASD
    3 = Severe ASD

This script:
    1) reads raw .mat EEG files with variable name "data"
    2) extracts sliding windows
    3) converts each window into DE band features: (32, 5)
    4) trains ST-DADGAT with LOSO subject split
    5) saves fold results to txt
"""

import os
import math
import random
from itertools import cycle
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from scipy.spatial import distance_matrix
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, precision_score, recall_score


# =========================================================
# 1. Config
# =========================================================
INPUT_DIR = r'E:\YZY\eeg\2_eeg_mat'
LABEL_CSV = r'E:\YZY\eeg\label.csv'
RESULT_DIR = r'E:\YZY\eeg\GCN\HERO-main\Compare\results_ST-DADGAT'

SELECTED_IDS = {"01", "02", "11", "12"}

MAT_KEY = "data"
NUM_CHANNELS = 32
NUM_CLASSES = 4

WINDOW_SIZE = 256
STEP = 128
FS = 256

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
MMD_WEIGHT = 1.0
SEED = 222

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# 2. Reproducibility
# =========================================================
def set_seed(seed=222):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 3. 32-channel 3D coordinates
# =========================================================
def return_coordinates_32():
    coords = np.array([
        [  83.9171,   29.4367,   -6.9900],   # Fp1
        [  58.5120,   -0.3122,   66.4620],   # Fz
        [  53.1112,   50.2438,   42.1920],   # F3
        [  42.4743,   70.2629,  -11.4200],   # F7
        [  18.6433,   77.2149,   24.4600],   # FC5
        [  26.0111,   34.0619,   79.9870],   # FC1
        [ -11.6317,   65.3581,   64.3580],   # C3
        [ -16.0187,   84.1611,   -9.3460],   # T7
        [ -46.5507,   79.5922,   30.9490],   # CP5
        [ -47.2919,   35.5131,   91.3150],   # CP1
        [ -81.1150,   -0.3247,   82.6150],   # Pz
        [ -78.7878,   53.0073,   55.9400],   # P3
        [ -73.4527,   72.4343,   -2.4870],   # P7
        [-112.4490,   29.4134,    8.8390],   # O1
        [-114.8920,   -0.1076,   14.6570],   # Oz
        [-112.1560,  -29.8426,    8.8000],   # O2
        [ -78.5602,  -55.6667,   56.5610],   # P4
        [ -73.0683,  -73.0557,   -2.5400],   # P8
        [ -46.1013,  -83.3218,   31.2060],   # CP6
        [ -47.0731,  -38.3838,   90.6950],   # CP2
        [  -9.1670,   -0.4009,  100.2440],   # Cz
        [ -10.9003,  -67.1179,   63.5800],   # C4
        [ -15.0203,  -85.0799,   -9.4900],   # T8
        [  19.9357,  -79.5341,   24.4380],   # FC6
        [  26.4379,  -34.7841,   78.8080],   # FC2
        [  54.3048,  -51.8362,   40.8140],   # F4
        [  44.4217,  -73.0431,  -12.0000],   # F8
        [  84.8959,  -29.8723,   -7.0800],   # Fp2
        [  41.6523,   70.1019,  -49.9520],   # F9
        [ -73.7657,   73.0093,  -40.9980],   # P9
        [ -74.3903,  -73.8947,  -41.2200],   # P10
        [  42.0667,  -72.1141,  -50.4520],   # F10
    ], dtype=np.float32)
    return coords


def calculate_adjacency_matrix_32(m=60.0):
    coords = return_coordinates_32()
    dist = distance_matrix(coords, coords)
    A = np.exp(-(dist ** 2) / (2 * (m ** 2)))
    np.fill_diagonal(A, 1.0)
    return A.astype(np.float32)


# =========================================================
# 4. Mat loading and DE features
# =========================================================
def load_mat_data(file_path, mat_key="data"):
    def _post_process(data):
        data = np.array(data)
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"data is not 2D in {file_path}, got shape {data.shape}")
        # unify to (channels, points)
        if data.shape[0] > data.shape[1]:
            data = data.T
        return data.astype(np.float32)

    try:
        mat_data = sio.loadmat(file_path)
        if mat_key in mat_data:
            data = mat_data[mat_key]
            data = np.array(data)
            data = np.squeeze(data)

            if data.dtype == object:
                if data.size == 1:
                    data = np.array(data.item())
                else:
                    found = False
                    for x in data.flat:
                        arr = np.array(x)
                        if arr.size > 0:
                            data = arr
                            found = True
                            break
                    if not found:
                        raise ValueError(f"object data is empty in {file_path}")

            return _post_process(data)
    except Exception:
        pass

    with h5py.File(file_path, 'r') as f:
        if mat_key not in f:
            raise KeyError(f"'{mat_key}' not found in {file_path}. Keys: {list(f.keys())}")

        ds = f[mat_key]
        data = np.array(ds)
        data = np.squeeze(data)

        if data.ndim == 2 and np.issubdtype(data.dtype, np.number):
            return _post_process(data)

        if data.ndim <= 2:
            refs = data.flatten()
            candidates = []
            for ref in refs:
                try:
                    obj = f[ref]
                    arr = np.array(obj)
                    arr = np.squeeze(arr)
                    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                        candidates.append(arr)
                except Exception:
                    continue
            if len(candidates) == 1:
                return _post_process(candidates[0])
            if len(candidates) > 1:
                candidates = sorted(candidates, key=lambda x: min(x.shape), reverse=True)
                return _post_process(candidates[0])

        raise ValueError(f"data is not 2D in {file_path}, got shape {data.shape}")


def _bandpass(x, low, high, fs, order=4):
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.9999)
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, x)


def extract_de_features(window, fs=200):
    """
    window: (32, T)
    return: (32, 5)
    """
    bands = [
        (1, 3),    # delta
        (3, 8),    # theta
        (8, 12),   # alpha
        (12, 30),  # beta
        (30, 50),  # gamma
    ]
    c, _ = window.shape
    feats = np.zeros((c, len(bands)), dtype=np.float32)
    for bi, (low, high) in enumerate(bands):
        for ch in range(c):
            xf = _bandpass(window[ch], low, high, fs)
            feats[ch, bi] = np.log(np.var(xf) + 1e-10)
    return feats


def read_label_map(csv_path):
    df = pd.read_csv(csv_path, dtype={"folder": str})
    if "folder" not in df.columns or "label" not in df.columns:
        raise ValueError("label.csv must contain columns: folder,label")
    label_map = {}
    for _, row in df.iterrows():
        label_map[str(row["folder"]).strip()] = int(row["label"])
    return label_map


def scan_records(input_dir, label_map, selected_ids=None):
    records = []
    subjects = []
    subfolders = sorted(os.listdir(input_dir), key=lambda x: str(x))
    for folder in subfolders:
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder not in label_map:
            continue

        mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mat")])
        if selected_ids:
            mat_files = [f for f in mat_files if f.split("_")[0].strip() in selected_ids]
        if len(mat_files) == 0:
            continue

        subjects.append(folder)
        for mat_name in mat_files:
            file_path = os.path.join(folder_path, mat_name)
            data = load_mat_data(file_path, mat_key=MAT_KEY)
            if data.shape[0] != NUM_CHANNELS:
                raise ValueError(f"{file_path} has {data.shape[0]} channels, expected {NUM_CHANNELS}.")
            n_points = data.shape[1]
            if n_points < WINDOW_SIZE:
                continue
            for start in range(0, n_points - WINDOW_SIZE + 1, STEP):
                records.append({
                    "subject": folder,
                    "file_path": file_path,
                    "start": start,
                    "label": label_map[folder],
                })
    subjects = sorted(list(set(subjects)), key=lambda x: str(x))
    return records, subjects


class ASDWindowFeatureDataset(Dataset):
    """
    Feature extraction is done on-the-fly.
    Return:
        x: (32, 5)
        y: long class label 0/1/2/3
    """
    def __init__(self, records, fs=200):
        self.records = records
        self.fs = fs
        self._cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        fp = item["file_path"]
        start = item["start"]
        if fp in self._cache:
            data = self._cache[fp]
        else:
            data = load_mat_data(fp, mat_key=MAT_KEY)
            self._cache[fp] = data

        window = data[:, start:start + WINDOW_SIZE]
        x = extract_de_features(window, fs=self.fs)
        y = int(item["label"])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =========================================================
# 5. MMD
# =========================================================
def _pairwise_distance(x):
    n = x.size(0)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(x, x.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-9)
    return dist


def _gaussian_kernel(features, kernel_mul=2.0, kernel_num=5, fix_sigma=1.0):
    kernel_val = 0
    for sigma_i in np.linspace(fix_sigma / kernel_mul, fix_sigma * kernel_mul, kernel_num):
        gamma = 1.0 / (2 * sigma_i ** 2)
        kernel_val += torch.exp(-gamma * _pairwise_distance(features))
    return kernel_val / kernel_num


def mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5):
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]

    source_features = F.layer_norm(source_features, [source_features.size(1)])
    target_features = F.layer_norm(target_features, [target_features.size(1)])

    features = torch.cat([source_features, target_features], dim=0)
    kernel_val = _gaussian_kernel(features, kernel_mul, kernel_num, fix_sigma=1.0)

    XX = kernel_val[:batch_size, :batch_size]
    YY = kernel_val[batch_size:, batch_size:]
    XY = kernel_val[:batch_size, batch_size:]
    return XX.mean() + YY.mean() - 2 * XY.mean()


# =========================================================
# 6. Model
# =========================================================
def normalize_A(A, symmetry=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt(d + 1e-10)
    D = torch.diag_embed(d)
    L = torch.matmul(torch.matmul(D, A), D)
    return L


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32, ratio=2):
        super(ChannelAttention, self).__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x


class MultiScaleConv(nn.Module):
    """
    Generalized version of the original conv3.
    Input:  (B, C=32, F=5)
    Output: (B, C=32, F_out=15)
    """
    def __init__(self, channels=32):
        super().__init__()
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(channels, channels, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(channels)
        self.bn5 = nn.BatchNorm1d(channels)
        self.bn7 = nn.BatchNorm1d(channels)
        self.act = nn.SELU()
        self.att = ChannelAttention(channels)

    def forward(self, x):
        x3 = self.act(self.bn3(self.conv3(x)))
        x5 = self.act(self.bn5(self.conv5(x)))
        x7 = self.act(self.bn7(self.conv7(x)))
        out = torch.cat([x3, x5, x7], dim=2)
        out = self.att(out)
        return out


class ChannelTransformer(nn.Module):
    """
    Transformer encoder over channels.
    Input:  (B, C, F)
    Output: (B, C, F)
    """
    def __init__(self, feature_dim=5, hidden_dim=32, num_heads=1, dropout=0.3):
        super().__init__()
        self.proj_in = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj_out = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        z = self.proj_in(x)
        z = self.encoder(z)
        z = self.proj_out(z)
        return z


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.m = 0.8

    def forward(self, inp, adj):
        h = torch.matmul(inp, self.W)  # (B,N,out)
        N = h.size(1)

        h_i = h.unsqueeze(2).repeat(1, 1, N, 1)
        h_j = h.unsqueeze(1).repeat(1, N, 1, 1)
        a_input = torch.cat([h_i, h_j], dim=-1)

        e = torch.matmul(self.leakyrelu(a_input), self.a).squeeze(-1)
        dj = self.m * adj.unsqueeze(0) + (1 - self.m) * e
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(dj > 0, dj, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)
        return F.relu(h_prime)


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        super().__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(n_heads)
        ])
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        hidden = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(hidden, self.dropout, training=self.training)
        node_logits = F.elu(self.out_att(x, adj))
        node_logits = F.log_softmax(node_logits, dim=2)
        return hidden, node_logits


class STDADGAT_ASD(nn.Module):
    def __init__(self, n_feat=20, n_hid=64, n_class=4, dropoutg=0.3, alpha=0.2, gn_heads=4, channels=32):
        super().__init__()
        self.channels = channels
        self.n_class = n_class
        self.A = torch.tensor(calculate_adjacency_matrix_32(), dtype=torch.float32)

        self.conv = MultiScaleConv(channels=channels)        # output 15 features
        self.transformer = ChannelTransformer(feature_dim=5, hidden_dim=32, num_heads=1, dropout=0.3)  # output 5 features
        self.gat = GAT(n_feat, n_hid, n_class, dropoutg, alpha, gn_heads)

        gat_flat = channels * n_hid * gn_heads
        self.adapter = nn.Sequential(
            nn.Linear(gat_flat, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fc1 = nn.Linear(gat_flat, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (B, 32, 5)
        L = normalize_A(self.A.to(x.device))

        x_conv = self.conv(x)             # (B,32,15)
        x_trans = self.transformer(x)     # (B,32,5)
        x_feat = torch.cat([x_conv, x_trans], dim=2)  # (B,32,20)

        hidden, node_logits = self.gat(x_feat, L)
        flat = torch.flatten(hidden, 1, 2)

        adapted_features = self.adapter(flat)
        z = self.dropout(F.relu(self.fc1(flat)))
        z = self.dropout(F.relu(self.fc2(z)))
        logits = self.fc3(z)
        return adapted_features, logits


# =========================================================
# 7. Train / Test
# =========================================================
def train_one_epoch(model, source_loader, target_loader, optimizer, epoch, device, mmd_weight=1.0):
    model.train()
    total_loss = 0.0
    class_loss_total = 0.0
    mmd_loss_total = 0.0
    correct = 0
    total = 0

    target_iter = cycle(target_loader)

    for batch_idx, (source_data, source_labels) in enumerate(source_loader):
        try:
            target_data, _ = next(target_iter)
        except StopIteration:
            target_iter = cycle(target_loader)
            target_data, _ = next(target_iter)

        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)

        optimizer.zero_grad()

        source_features, source_outputs = model(source_data)
        target_features, _ = model(target_data)

        class_loss = F.cross_entropy(source_outputs, source_labels)
        mmd = mmd_loss(source_features, target_features)

        loss = class_loss + mmd_weight * mmd
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        class_loss_total += class_loss.item()
        mmd_loss_total += mmd.item()

        _, predicted = source_outputs.max(1)
        correct += predicted.eq(source_labels).sum().item()
        total += source_labels.size(0)

    acc = 100.0 * correct / max(total, 1)
    avg_loss = total_loss / (batch_idx + 1)
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    return avg_loss


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)

        _, outputs = model(data)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return acc, f1, precision, recall, all_preds, all_labels


# =========================================================
# 8. Main
# =========================================================
def main():
    print(f"Using device: {DEVICE}")
    set_seed(SEED)

    label_map = read_label_map(LABEL_CSV)
    records, subject_ids = scan_records(INPUT_DIR, label_map, selected_ids=SELECTED_IDS)

    if len(subject_ids) < 2:
        raise RuntimeError("Need at least 2 subjects for LOSO.")

    print(f"Detected subjects: {subject_ids}")
    print(f"Total window samples: {len(records)}")
    print(f"Selected game IDs: {SELECTED_IDS}")

    result_path = os.path.join(RESULT_DIR, "fold_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("fold\tsubject\tbest_acc\tbest_f1\tbest_precision\tbest_recall\n")

    fold_accs, fold_f1s, fold_pres, fold_recs = [], [], [], []

    for fold, test_subject in enumerate(subject_ids, start=1):
        source_records = [r for r in records if r["subject"] != test_subject]
        target_records = [r for r in records if r["subject"] == test_subject]

        if not source_records or not target_records:
            continue

        source_ds = ASDWindowFeatureDataset(source_records, fs=FS)
        target_ds = ASDWindowFeatureDataset(target_records, fs=FS)

        source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        target_train_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        target_test_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        print("\n==== LOSO Fold ====")
        print(f"Fold {fold}/{len(subject_ids)} | Test subject: {test_subject}")
        print(f"Train windows: {len(source_ds)} | Test windows: {len(target_ds)}")

        model = STDADGAT_ASD(
            n_feat=20,
            n_hid=64,
            n_class=NUM_CLASSES,
            dropoutg=0.3,
            alpha=0.2,
            gn_heads=4,
            channels=NUM_CHANNELS
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_acc, best_f1, best_pre, best_rec = 0.0, 0.0, 0.0, 0.0

        for epoch in range(1, EPOCHS + 1):
            train_one_epoch(model, source_loader, target_train_loader, optimizer, epoch, DEVICE, MMD_WEIGHT)
            acc, f1, pre, rec, _, _ = test(model, target_test_loader, DEVICE)
            print(f"Test: Acc={acc:.2f}% | F1={f1:.4f} | Precision={pre:.4f} | Recall={rec:.4f}")

            if acc > best_acc:
                best_acc, best_f1, best_pre, best_rec = acc, f1, pre, rec

        fold_accs.append(best_acc)
        fold_f1s.append(best_f1)
        fold_pres.append(best_pre)
        fold_recs.append(best_rec)

        line = f"{fold}\t{test_subject}\t{best_acc:.4f}\t{best_f1:.4f}\t{best_pre:.4f}\t{best_rec:.4f}\n"
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(line)

        print(f"[Fold {test_subject}] Best Acc={best_acc:.2f}% | Best F1={best_f1:.4f}")

    acc_mean, acc_std = np.mean(fold_accs), np.std(fold_accs, ddof=1)
    f1_mean, f1_std = np.mean(fold_f1s), np.std(fold_f1s, ddof=1)
    pre_mean, pre_std = np.mean(fold_pres), np.std(fold_pres, ddof=1)
    rec_mean, rec_std = np.mean(fold_recs), np.std(fold_recs, ddof=1)

    print("\n===== LOSO Summary =====")
    print(f"Accuracy : {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f"F1-score : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Precision: {pre_mean:.4f} ± {pre_std:.4f}")
    print(f"Recall   : {rec_mean:.4f} ± {rec_std:.4f}")

    with open(result_path, "a", encoding="utf-8") as f:
        f.write("\nSummary\n")
        f.write(f"Accuracy\t{acc_mean:.4f}\t{acc_std:.4f}\n")
        f.write(f"F1\t{f1_mean:.4f}\t{f1_std:.4f}\n")
        f.write(f"Precision\t{pre_mean:.4f}\t{pre_std:.4f}\n")
        f.write(f"Recall\t{rec_mean:.4f}\t{rec_std:.4f}\n")


if __name__ == "__main__":
    main()
