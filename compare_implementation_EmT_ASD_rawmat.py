
# -*- coding: utf-8 -*-
"""
implementation_EmT_ASD_rawmat.py

Adapted EmT baseline for your ASD severity screening data.

Expected data format:
    INPUT_DIR/
        10/
            05_*.mat
            06_*.mat
            ...
        11/
            ...
    LABEL_CSV:
        folder,label
        10,1
        11,1
        ...

Each .mat contains variable "data" with shape:
    (32, n_points) or (n_points, 32)

Labels:
    0 = TD
    1 = Mild ASD
    2 = Moderate ASD
    3 = Severe ASD

This script:
    - reads raw .mat files
    - selects game files by prefix, e.g. {"05","06","15","16"}
    - cuts sliding windows
    - extracts DE features in 5 frequency bands
    - feeds EmT with shape (N, 1, 32, 5)
    - performs leave-one-subject-out validation
    - saves fold_results.txt
"""

import os
import time
import random
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from scipy import signal

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from EmT_cls import EmT


# =========================================================
# Config
# =========================================================
INPUT_DIR = r'E:\YZY\eeg\2_eeg_mat'
LABEL_CSV = r'E:\YZY\eeg\label.csv'
RESULT_DIR = r'E:\YZY\eeg\GCN\HERO-main\Compare\results_EmT_ASD'

SELECTED_IDS = {"01", "02", "11", "12"}

MAT_KEY = "data"
NUM_CHANNELS = 32
NUM_CLASSES = 4

SAMPLING_RATE = 200
WINDOW_SIZE = 256
STEP = 128

EPOCHS = 20
EVAL_EVERY = 1
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 222

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# Utils
# =========================================================
def setup_seed(seed: int = 222):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1, 3),
    "theta": (3, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 50),
}


def load_mat_data(file_path: str, mat_key: str = "data") -> np.ndarray:
    """
    Compatible with ordinary MAT and MATLAB v7.3 HDF5 MAT.
    Return shape: (channels, points).
    """
    def _post_process(data):
        data = np.array(data)
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"data is not 2D in {file_path}, got shape {data.shape}")
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

    with h5py.File(file_path, "r") as f:
        if mat_key not in f:
            raise KeyError(f"'{mat_key}' not found in {file_path}. Keys: {list(f.keys())}")

        data = np.array(f[mat_key])
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


def read_label_map(csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(csv_path, dtype={"folder": str})
    if "folder" not in df.columns or "label" not in df.columns:
        raise ValueError("label.csv must contain columns: folder,label")

    label_map = {}
    for _, row in df.iterrows():
        folder = str(row["folder"]).strip()
        label = int(row["label"])
        label_map[folder] = label
    return label_map


def extract_de_features(window: np.ndarray, sampling_rate: int = 200) -> np.ndarray:
    """
    Differential Entropy-like log-variance feature.
    input:
        window: (32, T)
    output:
        features: (32, 5)
    """
    eps = 1e-10
    n_channels, _ = window.shape
    band_names = list(BANDS.keys())
    feats = np.zeros((n_channels, len(band_names)), dtype=np.float32)

    nyq = 0.5 * sampling_rate
    for b_idx, name in enumerate(band_names):
        low, high = BANDS[name]
        low_cut = max(low / nyq, 1e-6)
        high_cut = min(high / nyq, 0.999)
        if low_cut >= high_cut:
            high_cut = min(low_cut + 1e-3, 0.999)

        b, a = signal.butter(4, [low_cut, high_cut], btype="band")

        for ch in range(n_channels):
            x = window[ch]
            try:
                xf = signal.filtfilt(b, a, x)
            except ValueError:
                xf = x
            var = np.var(xf, ddof=0)
            feats[ch, b_idx] = np.log(var + eps)

    return feats


def scan_records(input_dir: str, label_map: Dict[str, int], selected_ids=None):
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
            fp = os.path.join(folder_path, mat_name)
            data = load_mat_data(fp, MAT_KEY)

            if data.shape[0] != NUM_CHANNELS:
                raise ValueError(f"{fp} has {data.shape[0]} channels, expected {NUM_CHANNELS}.")

            n_points = data.shape[1]
            if n_points < WINDOW_SIZE:
                continue

            for start in range(0, n_points - WINDOW_SIZE + 1, STEP):
                records.append({
                    "subject": folder,
                    "file_path": fp,
                    "start": start,
                    "label": label_map[folder],
                })

    subjects = sorted(list(set(subjects)), key=lambda x: str(x))
    return records, subjects


class ASDWindowDEDataset(Dataset):
    """
    On-the-fly DE feature extraction.
    Return:
        x: (1, 32, 5), because EmT expects (batch, sequence, channel, feature)
        y: integer label 0/1/2/3
    """
    def __init__(self, records, sampling_rate=200):
        self.records = records
        self.sampling_rate = sampling_rate
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        fp = item["file_path"]
        start = item["start"]

        if fp in self.cache:
            data = self.cache[fp]
        else:
            data = load_mat_data(fp, MAT_KEY)
            self.cache[fp] = data

        window = data[:, start:start + WINDOW_SIZE]
        feats = extract_de_features(window, self.sampling_rate)  # (32,5)
        feats = feats[None, :, :]  # (1,32,5)
        label = int(item["label"])

        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def standardize_by_train(train_ds_features, test_ds_features):
    """
    Used when features are precomputed. Currently not used because features are extracted on the fly.
    """
    pass


# =========================================================
# Train / Test
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        logits = model(x)
        pred = logits.argmax(dim=1)

        preds_all.extend(pred.detach().cpu().numpy().tolist())
        labels_all.extend(y.detach().cpu().numpy().tolist())

    if len(labels_all) == 0:
        return 0.0, 0.0, 0.0, 0.0, [], []

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    pre = precision_score(labels_all, preds_all, average="macro", zero_division=0)
    rec = recall_score(labels_all, preds_all, average="macro", zero_division=0)
    return acc, f1, pre, rec, preds_all, labels_all


def train_one_fold(train_records, test_records, test_subject, fold_idx, total_folds):
    train_ds = ASDWindowDEDataset(train_records, sampling_rate=SAMPLING_RATE)
    test_ds = ASDWindowDEDataset(test_records, sampling_rate=SAMPLING_RATE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = EmT(
        layers_graph=[1, 2],
        layers_transformer=2,
        num_adj=2,
        num_chan=NUM_CHANNELS,
        num_feature=5,
        hidden_graph=32,
        K=4,
        num_head=8,
        dim_head=16,
        dropout=0.25,
        num_class=NUM_CLASSES,
        graph2token="Linear",
        encoder_type="Cheby",
        alpha=0.25,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc, best_f1, best_pre, best_rec = 0.0, 0.0, 0.0, 0.0
    best_preds, best_labels = [], []

    print(f"\nFold {fold_idx}/{total_folds} | Test subject: {test_subject}")
    print(f"Train windows: {len(train_ds)} | Test windows: {len(test_ds)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).long()

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

        train_acc = correct / max(total, 1)
        train_loss = total_loss / max(total, 1)

        acc, f1, pre, rec, preds, labels = evaluate(model, test_loader, DEVICE)

        if acc > best_acc:
            best_acc, best_f1, best_pre, best_rec = acc, f1, pre, rec
            best_preds, best_labels = preds, labels

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} | "
            f"val_acc={acc:.4f} f1={f1:.4f} best={best_acc:.4f}"
        )

    return best_acc, best_f1, best_pre, best_rec, best_preds, best_labels


def main():
    print(f"Using device: {DEVICE}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Label csv: {LABEL_CSV}")
    print(f"Selected game IDs: {SELECTED_IDS}")

    setup_seed(SEED)

    label_map = read_label_map(LABEL_CSV)
    records, subjects = scan_records(INPUT_DIR, label_map, selected_ids=SELECTED_IDS)

    if len(subjects) < 2:
        raise RuntimeError("Need at least 2 subjects for LOSO.")

    print(f"Detected subjects: {len(subjects)}")
    print(f"Total window samples: {len(records)}")

    result_path = os.path.join(RESULT_DIR, "fold_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("fold\tsubject\tbest_acc\tbest_f1\tbest_precision\tbest_recall\twindow_pred\n")

    fold_accs, fold_f1s, fold_pres, fold_recs = [], [], [], []

    for fold_idx, test_subject in enumerate(subjects, start=1):
        train_records = [r for r in records if r["subject"] != test_subject]
        test_records = [r for r in records if r["subject"] == test_subject]

        if len(train_records) == 0 or len(test_records) == 0:
            continue

        acc, f1, pre, rec, preds, labels = train_one_fold(
            train_records, test_records, test_subject, fold_idx, len(subjects)
        )

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_pres.append(pre)
        fold_recs.append(rec)

        counts = np.bincount(np.array(preds, dtype=int), minlength=NUM_CLASSES)
        dist_str = ", ".join([f"{i}:{int(counts[i])}" for i in range(NUM_CLASSES)])

        line = f"{fold_idx}\t{test_subject}\t{acc:.4f}\t{f1:.4f}\t{pre:.4f}\t{rec:.4f}\t{dist_str}\n"
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(line)

        current_mean = np.mean(fold_accs)
        current_std = np.std(fold_accs, ddof=1) if len(fold_accs) > 1 else 0.0
        print(
            f"Fold {fold_idx}/{len(subjects)} finished | "
            f"subject={test_subject} | acc={acc:.4f} | "
            f"mean={current_mean:.4f} | std={current_std:.4f} | "
            f"window_pred: {dist_str}"
        )

    if len(fold_accs) == 0:
        raise RuntimeError("No folds completed.")

    acc_mean = np.mean(fold_accs)
    acc_std = np.std(fold_accs, ddof=1) if len(fold_accs) > 1 else 0.0
    f1_mean = np.mean(fold_f1s)
    f1_std = np.std(fold_f1s, ddof=1) if len(fold_f1s) > 1 else 0.0
    pre_mean = np.mean(fold_pres)
    pre_std = np.std(fold_pres, ddof=1) if len(fold_pres) > 1 else 0.0
    rec_mean = np.mean(fold_recs)
    rec_std = np.std(fold_recs, ddof=1) if len(fold_recs) > 1 else 0.0

    print("\n===== LOSO Summary =====")
    print(f"Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
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
