'''EDA + PIPELINE CLASSIFICATION (ResNet)'''

import os, math, warnings, sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ---------------------- CONFIG ---------------------- #
DATA_ROOT = r""
CSV_PATH  = os.path.join(DATA_ROOT, "train.csv")           # expected columns: image_id,label
IMG_DIR   = os.path.join(DATA_ROOT, "train_images")        # images folder
SAVE_BEST = os.path.join(DATA_ROOT, "best_model.pt")       # Where the model will be saved (F1-score metric)

IMG_SIZE  = 448          # IMG_size, adjust according to hardware
BATCH     = 32           # Batch size, adjust according to hardware
EPOCHS    = 15           # Num_epochs
LR        = 1e-3         # First Learning Rate
WD        = 1e-4         # Weight decay
NUM_WORKS = 4
SEED      = 69           # Seed
BACKBONE  = "resnet50"   # {"resnet18","resnet50"}
FREEZE    = True         # Freeze backbone at startup (transfer learning)
USE_SAMPLER = True       # activate weighted sampler if there is imbalance
TMAX_SCHED  = EPOCHS     # for CosineAnnealingLR (Dynamic Lr)

# ---------------------- UTILS ---------------------- #
def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_fn():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unfreeze_backbone_and_lower_lr(model, optimizer, base_lr=LR, wd=5e-5):
    for p in model.parameters():
        p.requires_grad = True
    for g in optimizer.param_groups:
        g["lr"] = base_lr * 0.1
        g["weight_decay"] = wd

# ---------------------- EDA  ---------------------- #
class EDA:
    def __init__(self, path, ftype, sql_query=None):
        self.df = None
        try:
            if ftype in ('csv',):
                self.df = pd.read_csv(path)
                print(f"{'-'*5}CSV loaded ok{'-'*5}")
            elif ftype in ('xls','xlsx','xlsm','excel'):
                self.df = pd.read_excel(path)
                print(f"{'-'*5}Excel loaded ok{'-'*5}")
            elif ftype in ('sqlite','sql') and sql_query:
                connect = sqlite3.connect(path)
                self.df = pd.read_sql_query(sql_query, connect)
                print(f"{'-'*5}SQL loaded ok{'-'*5}")
            elif ftype in ('dataframe','pd'):
                self.df = path
            else:
                raise ValueError("Invalid type or missing SQL query")
        except Exception as e:
            print(f"ERROR loading dataset: {e}")

    def show_first_last_row(self):
        print(f"First rows:\n{self.df.head()}\n\nLast rows:\n{self.df.tail()}\n")

    def show_dataset_shape(self):
        print(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} cols")

    def show_columns_types(self):
        print(self.df.info())

    def missing_values(self):
        nmiss = self.df.isnull().sum().sum()
        if nmiss == 0:
            print(f"\n{'-'*5}No nulls{'-'*5}")
        else:
            print("\nNulls % by column:")
            print((self.df.isnull().sum()/len(self.df))*100)
            plt.figure(figsize=(8,4))
            sns.heatmap(self.df.isnull(), cbar=True, cmap='cividis')
            plt.title("Nulls heatmap"); plt.tight_layout(); plt.show()

    def detect_duplicates(self):
        dups = self.df.duplicated()
        if not any(dups):
            print(f"\n{'-'*5}No duplicates{'-'*5}")
        else:
            print(f"\n{'-'*5}Duplicates found{'-'*5}\n", self.df.duplicated().value_counts())

    def calculate_basic_values(self):
        try:
            print(f"\nNumeric summary:\n{self.df.describe()}")
            print(f"\nCategorical summary:\n{self.df.describe(include=['object','category'])}")
            print(f"\nMost frequent categorical values:\n{self.df.select_dtypes(include=['object','category']).mode().iloc[0]}")
        except ValueError as e:
            print(f"Err: {e}")

    def unique_values_sumary(self):
        for c in self.df.columns:
            print(f"\nCol {c}: {self.df[c].nunique()} unique\nTop:\n{self.df[c].value_counts().head()}")

    def plot_categorical_distribution(self, columns=None, max_unique=50, mode='grid', max_cols=10, n_cols_grid=3, top_n=None, include_numeric_small_card=True):
        df = self.df
        if df is None or df.empty:
            print("Empty DF")
            return

        if columns is None:
            cat_like = df.select_dtypes(include=['object','category','string']).columns.tolist()
            if include_numeric_small_card:
                num_small = [c for c in df.columns if (pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=False) <= max_unique)]
            else:
                num_small=[]
            candidates = list(dict.fromkeys(cat_like + num_small))
        else:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                print(f"Omitted not found: {missing}")
            candidates = [c for c in columns if c in df.columns]

        reduced_columns = [c for c in candidates if df[c].nunique(dropna=False) <= max_unique][:max_cols]
        if not reduced_columns:
            print("No categorical/low-card columns")
            return

        def _value_counts_series(s: pd.Series):
            s = s.astype("object").where(~s.isna(), "__NaN__")
            vc = s.value_counts(dropna=False)
            return vc.head(top_n) if top_n else vc

        if mode == 'grid':
            n = len(reduced_columns)
            n_cols = min(n_cols_grid, n)
            n_rows = math.ceil(n/n_cols)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.8*n_rows))
            axs = np.array(axs).reshape(-1)
            for idx, col in enumerate(reduced_columns):
                vc = _value_counts_series(df[col])
                plot_df = vc.reset_index(); plot_df.columns = [col,'count']
                ax = axs[idx]
                sns.barplot(data=plot_df, x=col, y='count', ax=ax)
                ax.set_title(f"{col} (n={len(df)})"); ax.set_ylabel("Freq"); ax.set_xlabel(col)
            for j in range(idx+1, len(axs)): fig.delaxes(axs[j])
            plt.tight_layout(); plt.show()
        else:
            for col in reduced_columns:
                vc = _value_counts_series(df[col])
                plot_df = vc.reset_index(); plot_df.columns = [col,'count']
                plt.figure(figsize=(7.5,4.5))
                sns.barplot(data=plot_df, x=col, y='count')
                plt.title(f"{col} (n={len(df)})")
                plt.tight_layout(); plt.show()

# ---------------------- CSV Dataset (Classification) ---------------------- #
class CassavaCSV(Dataset):
    def __init__(self, df, img_dir, transform=None, img_col="image_id", y_col="label"):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.img_col = img_col
        self.y_col = y_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, row[self.img_col])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Cannot open {img_path}: {e}")
        #img = Image.open(img_path).convert("RGB")
        y = int(row[self.y_col])
        if self.transform: img = self.transform(img)
        return img, y

# ---------------------- Transforms ---------------------- #
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def make_transforms(img_size=IMG_SIZE):
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf

# ---------------------- Model + Optim ---------------------- #
def build_model(num_classes, backbone=BACKBONE, freeze_backbone=FREEZE, device=None):
    if backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_f = model.fc.in_features
    elif backbone == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_f = model.fc.in_features
    else:
        raise ValueError("backbone must be {'resnet18','resnet50'}")
    model.fc = nn.Linear(in_f, num_classes)

    if freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("fc"):
                p.requires_grad = False

    if device is None:
        device = device_fn()
    model = model.to(device)
    return model, device

def make_optimizer_and_sched(model, lr=LR, wd=WD):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=lr, weight_decay=wd)
    # Restarts: T_0=2 epochs, then double period (2, 4, 8, ...)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2)
    return opt, sched


# ---------------------- Metrics and Loops ---------------------- #
def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

from tqdm import tqdm
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="train", ncols=80):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        loss_sum += loss.item() * bs
        acc_sum  += (logits.argmax(1) == y).float().sum().item()
        n += bs
    return loss_sum/n, acc_sum/n

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            loss_sum += loss.item() * bs
            acc_sum  += (logits.argmax(1) == y).float().sum().item()
            n += bs
            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(logits.argmax(1).detach().cpu().numpy())
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)

        f1m = f1_score(y_true, y_pred, average="macro")
        cm  = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        return (loss_sum/n, acc_sum/n, f1m, cm, y_true, y_pred)

# ---------------------- Plot history ---------------------- #
def PlotIt(df_hist):
    plt.figure(figsize=(11,4))
    sns.lineplot(data=df_hist, x="Epoch", y="Train_loss", marker='o', label="Train_loss")
    sns.lineplot(data=df_hist, x="Epoch", y="Val_loss", marker='o', label="Val_loss")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(11,4))
    sns.lineplot(data=df_hist, x="Epoch", y="Train_acc", marker='o', label="Train_acc")
    sns.lineplot(data=df_hist, x="Epoch", y="Val_acc", marker='o', label="Val_acc")
    sns.lineplot(data=df_hist, x="Epoch", y="Val_f1m", marker='o', label="Val_f1m")
    plt.tight_layout()
    plt.show()

# ---------------------- Unbalanced sampler ---------------------- #
def make_weighted_sampler(df_labels, power=1.0):
    counts = df_labels.value_counts().to_dict()  # label -> count
    weights_per_class = {c: (1.0/max(1, counts[c]))**power for c in counts}
    weights = df_labels.map(weights_per_class).astype("float32").values
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler

# ---------------- Show an example ------------- #
CASSAVA_NAMES = [
    "0: CBB (bacterial blight)",
    "1: CBSD (brown streak)",
    "2: CGM (green mottle)",
    "3: CMD (mosaic disease)",
    "4: Healthy"
]
def denormalize(img, mean=MEAN, std=STD):
    mean = torch.tensor(mean).view(-1,1,1)
    std = torch.tensor(std).view(-1,1,1)
    return (img * std + mean).clamp(0,1)

def show_prediction(model, dataset, idx=0, class_names=None, device=None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img, label = dataset[idx]
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        pred = logits.argmax(1).item()

    # Denormalization
    img_vis = denormalize(img).permute(1, 2, 0).cpu().numpy()

    plt.imshow(img_vis)
    plt.axis("off")
    true_lbl = class_names[label] if class_names else label
    pred_lbl = class_names[pred] if class_names else pred
    plt.title(f"True: {true_lbl} | Pred: {pred_lbl}")
    plt.show()


# ---------------------- MAIN ---------------------- #
def main():
    set_seed()
    device = device_fn()
    print("Device:", device)

    # ---------- EDA ----------
    data = EDA(CSV_PATH, 'csv')
    data.show_first_last_row()
    data.show_dataset_shape()
    data.show_columns_types()
    data.missing_values()
    data.detect_duplicates()
    data.calculate_basic_values()
    data.unique_values_sumary()
    # Distribution of labels
    if "label" in data.df.columns:
        plt.figure(figsize=(6,3))
        sns.countplot(data=data.df, x="label")
        plt.title("Label distribution"); plt.tight_layout(); plt.show()

    # ---------- Split ----------
    df = data.df.copy()
    #assert {"image_id","label"}.issubset(df.columns), "train.csv must have columns: image_id,label"
    num_classes = df["label"].nunique()
    print("Classes:", num_classes, "->", Counter(df["label"]))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(df["image_id"], df["label"]))
    df_train = df.iloc[tr_idx].reset_index(drop=True)
    df_val   = df.iloc[va_idx].reset_index(drop=True)

    # ---------- Transforms & Datasets ----------
    train_tf, val_tf = make_transforms(IMG_SIZE)
    train_ds = CassavaCSV(df_train, IMG_DIR, transform=train_tf)
    val_ds = CassavaCSV(df_val,   IMG_DIR, transform=val_tf)

    # ---------- DataLoaders ---------- #
    if USE_SAMPLER:
        sampler = make_weighted_sampler(df_train["label"], power=1.0) ########################POWER CORRECTION
        train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                                  num_workers=NUM_WORKS, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                  num_workers=NUM_WORKS, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=NUM_WORKS, pin_memory=True)

    # ---------- Model + Optim ---------- #
    model, device = build_model(num_classes=num_classes, backbone=BACKBONE, freeze_backbone=FREEZE, device=device)

    '''(Effective Number, Cui et al. 2019):'''
    counts = df_train["label"].value_counts().sort_index().values
    beta = 0.999
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff_num
    w = torch.tensor(w / w.sum() * len(w), dtype=torch.float32).to(device)
    #criterion = nn.CrossEntropyLoss(weight=w)
    criterion = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.005)


    optimizer, scheduler = make_optimizer_and_sched(model, lr=LR, wd=WD)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ---------- Loop ---------- #
    hist = {"Epoch":[], "Train_loss":[], "Val_loss":[], "Train_acc":[], "Val_acc":[], "Val_f1m":[]}
    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        # ---- progressive unfreeze from epoch 2 ----
        if epoch == 2 and FREEZE:
            unfreeze_backbone_and_lower_lr(model, optimizer, base_lr=LR, wd=5e-5)
            print("[INFO] Unfreeze backbone + LR -> LR/10, WD=5e-5")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        va_loss, va_acc, va_f1, cm, y_true, y_pred = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        # checkpoint F1 macro
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), SAVE_BEST)
            print(f"Saved best @epoch {epoch} (F1m={best_f1:.4f}) -> {SAVE_BEST}")

        hist["Epoch"].append(epoch)
        hist["Train_loss"].append(tr_loss)
        hist["Val_loss"].append(va_loss)
        hist["Train_acc"].append(tr_acc)
        hist["Val_acc"].append(va_acc)
        hist["Val_f1m"].append(va_f1)

        print(f"[{epoch:02d}] train: loss={tr_loss:.4f} acc={tr_acc*100:.2f}%  | "
              f"val: loss={va_loss:.4f} acc={va_acc*100:.2f}% f1m={va_f1:.4f}")

    df_hist = pd.DataFrame(hist)
    print("\nHistory:\n", df_hist.tail())
    PlotIt(df_hist)

    # ---------- Final report ----------
    # Confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False)
    plt.title("Confusion Matrix (val)"); plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.show()

    # Report by class
    print("\nClassification report (val):\n",
          classification_report(y_true, y_pred, digits=4))

    # Example: see prediction of the first image of val_ds
    import random
    n = random.randint(1, 10)
    show_prediction(model, val_ds, idx=n,class_names=CASSAVA_NAMES, device=device)


if __name__ == "__main__":

    main()
