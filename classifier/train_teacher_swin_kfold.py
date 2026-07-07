# train_teacher_swin_kfold.py

import os
import argparse
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

import timm
from safetensors.torch import load_file

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================
# 1. 类别定义
# =========================
ID2NAME = {
    0: "fruitlet",
    1: "hard",
    2: "mature",
    3: "first_dilatation",
    4: "growing",
    5: "second_dilatation"
}

CLASS_TO_STAGE = {v: k for k, v in ID2NAME.items()}


# =========================
# 2. 固定随机种子
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 3. 数据集定义
# =========================
class FolderStageDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (image_path, stage_id)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]

        img = Image.open(p).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, int(y)


def collect_samples(root_dir: str):
    """
    root_dir/
        fruitlet/
        hard/
        mature/
        first_dilatation/
        growing/
        second_dilatation/
    """
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    all_samples = []

    for folder, stage_id in CLASS_TO_STAGE.items():
        class_dir = root / folder

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        class_samples = []

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            class_samples.extend(list(class_dir.glob(ext)))

        if len(class_samples) == 0:
            raise RuntimeError(f"No images found in {class_dir}")

        for p in sorted(class_samples):
            all_samples.append((p, stage_id))

        print(f"[DATA] class={folder:20s} id={stage_id} num={len(class_samples)}")

    return all_samples


# =========================
# 4. 数据增强
# =========================
def build_transforms(imgsz=224):
    train_tf = T.Compose([
        T.RandomResizedCrop(imgsz, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(10),
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_tf = T.Compose([
        T.Resize(int(imgsz * 1.14)),
        T.CenterCrop(imgsz),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tf, val_tf


# =========================
# 5. 加载 Swin 预训练权重
# =========================
def load_local_pretrained_weights(model, pretrained_path):
    if pretrained_path is None or pretrained_path == "":
        print("[PRETRAIN] pretrained_path is empty, train from random init.")
        return model

    if not os.path.exists(pretrained_path):
        print(f"[PRETRAIN] file not found: {pretrained_path}")
        print("[PRETRAIN] train from random init.")
        return model

    print(f"[PRETRAIN] loading local weights: {pretrained_path}")

    state_dict = load_file(pretrained_path)
    model_dict = model.state_dict()

    matched_state_dict = {}

    for k, v in state_dict.items():
        key = k

        # 兼容部分权重中可能存在的 model. 前缀
        if key.startswith("model."):
            key = key[len("model."):]

        if key in model_dict and v.shape == model_dict[key].shape:
            matched_state_dict[key] = v

    model_dict.update(matched_state_dict)
    model.load_state_dict(model_dict)

    print(f"[PRETRAIN] matched params: {len(matched_state_dict)} / {len(model_dict)}")

    return model


# =========================
# 6. 构建 Swin Teacher
# =========================
def build_teacher(
    model_name="swin_tiny_patch4_window7_224",
    num_classes=6,
    pretrained_path=None,
    freeze_backbone=False
):
    """
    推荐 model_name:
    - swin_tiny_patch4_window7_224
    - swin_small_patch4_window7_224
    - swin_base_patch4_window7_224
    """

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    model = load_local_pretrained_weights(model, pretrained_path)

    if freeze_backbone:
        print("[MODEL] freeze backbone, only train last stage and head.")

        for p in model.parameters():
            p.requires_grad = False

        # timm Swin 通常包含 layers 或 stages，做兼容处理
        if hasattr(model, "layers"):
            for p in model.layers[-1].parameters():
                p.requires_grad = True

        if hasattr(model, "stages"):
            for p in model.stages[-1].parameters():
                p.requires_grad = True

        # 分类头
        for name, p in model.named_parameters():
            if name.startswith("head") or name.startswith("classifier"):
                p.requires_grad = True
    else:
        print("[MODEL] finetune all parameters.")

        for p in model.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"[MODEL] {model_name}")
    print(f"[MODEL] trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

    return model


# =========================
# 7. 评估函数
# =========================
@torch.no_grad()
def evaluate_teacher(model, loader, device, ce):
    model.eval()

    total_loss = 0.0
    total = 0

    y_true = []
    y_pred = []

    pbar = tqdm(loader, desc="Eval", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = ce(logits, y)

        pred = logits.argmax(dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total += bs

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total, 1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(ID2NAME)))
    )

    return avg_loss, acc, macro_f1, cm


# =========================
# 8. 训练单折
# =========================
def train_one_fold(
    fold,
    train_samples,
    val_samples,
    args,
    device
):
    print(f"\n========== Fold {fold + 1}/{args.num_folds} ==========")
    print(f"[Fold {fold + 1}] train num = {len(train_samples)}")
    print(f"[Fold {fold + 1}] val   num = {len(val_samples)}")

    train_tf, val_tf = build_transforms(args.imgsz)

    train_ds = FolderStageDataset(train_samples, transform=train_tf)
    val_ds = FolderStageDataset(val_samples, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model = build_teacher(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained_path=args.pretrained_path,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    use_amp = bool(args.amp) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = -1.0
    best_f1 = -1.0
    best_epoch = -1
    best_cm = None
    bad = 0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_path = save_dir / f"best_model_fold{fold + 1}.pth"
    last_path = save_dir / f"last_model_fold{fold + 1}.pth"
    cm_path = save_dir / f"confusion_matrix_fold{fold + 1}.npy"

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total = 0
        y_true_train = []
        y_pred_train = []

        pbar = tqdm(
            train_loader,
            desc=f"Fold {fold + 1} Epoch {epoch:03d}/{args.epochs}",
            leave=True,
            dynamic_ncols=True
        )

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip
            )

            scaler.step(optimizer)
            scaler.update()

            pred = logits.argmax(dim=1)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

            y_true_train.extend(y.detach().cpu().numpy().tolist())
            y_pred_train.extend(pred.detach().cpu().numpy().tolist())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        scheduler.step()

        train_loss = total_loss / max(total, 1)
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_f1 = f1_score(
            y_true_train,
            y_pred_train,
            average="macro",
            zero_division=0
        )

        val_loss, val_acc, val_f1, cm = evaluate_teacher(
            model=model,
            loader=val_loader,
            device=device,
            ce=ce
        )

        print(
            f"Fold {fold + 1} | "
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macroF1={val_f1:.4f}"
        )

        history.append({
            "fold": fold + 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # 推荐按 macro-F1 保存，因为你的六分类可能类别不均衡
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            best_cm = cm.copy()
            bad = 0

            torch.save({
                "model": model.state_dict(),
                "model_name": args.model_name,
                "num_classes": args.num_classes,
                "imgsz": args.imgsz,
                "ID2NAME": ID2NAME,
                "CLASS_TO_STAGE": CLASS_TO_STAGE,
                "best_acc": float(best_acc),
                "best_macro_f1": float(best_f1),
                "best_epoch": int(best_epoch),
                "fold": int(fold + 1),
                "pretrained_path": args.pretrained_path,
            }, best_path)

            print(
                f"  ✅ saved best: {best_path} "
                f"(epoch={best_epoch}, acc={best_acc:.4f}, macroF1={best_f1:.4f})"
            )
        else:
            bad += 1

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "model_name": args.model_name,
            "num_classes": args.num_classes,
            "imgsz": args.imgsz,
            "ID2NAME": ID2NAME,
            "CLASS_TO_STAGE": CLASS_TO_STAGE,
            "epoch": int(epoch),
            "fold": int(fold + 1),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
        }, last_path)

        if bad >= args.patience:
            print(f"[Fold {fold + 1}] Early stopping at epoch {epoch}.")
            break

    if best_cm is not None:
        np.save(cm_path, best_cm)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        save_dir / f"history_fold{fold + 1}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "fold": fold + 1,
        "best_acc": best_acc,
        "best_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "best_path": str(best_path)
    }


# =========================
# 9. 主函数：10 折 Teacher
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/root/autodl-tmp/cross_dataset",
        help="dataset root with class folders"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/ultralytics-yolo11-main-origin/classifier/teacher_model/weight_teacher_swin",
        help="folder to save best_model_fold*.pth"
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/root/autodl-tmp/pretrained/SwinTransformer/model.safetensors",
        help="local Swin pretrained model.safetensors"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="swin_tiny_patch4_window7_224",
        help="timm Swin model name"
    )

    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.08)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="freeze backbone and train only last stage + head"
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="use mixed precision training"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n========== Swin Teacher 10-Fold Training ==========")
    print(f"[ARGS] root_dir        = {args.root_dir}")
    print(f"[ARGS] save_dir        = {args.save_dir}")
    print(f"[ARGS] model_name      = {args.model_name}")
    print(f"[ARGS] pretrained_path = {args.pretrained_path}")
    print(f"[ARGS] device          = {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_samples = collect_samples(args.root_dir)

    all_paths = [p for p, y in all_samples]
    all_labels = [y for p, y in all_samples]

    print(f"[DATA] total samples = {len(all_samples)}")

    label_counts = pd.Series(all_labels).value_counts().sort_index()
    print("[DATA] label counts:")
    for cls_id, count in label_counts.items():
        print(f"  {cls_id}: {ID2NAME[int(cls_id)]} -> {count}")

    skf = StratifiedKFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=args.seed
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(all_paths, all_labels)
    ):
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        result = train_one_fold(
            fold=fold,
            train_samples=train_samples,
            val_samples=val_samples,
            args=args,
            device=device
        )

        fold_results.append(result)

        pd.DataFrame(fold_results).to_csv(
            save_dir / "teacher_swin_10fold_results.csv",
            index=False,
            encoding="utf-8-sig"
        )

    df = pd.DataFrame(fold_results)

    print("\n========== 10-Fold Teacher Results ==========")
    print(df)

    print("\n===== Mean ± Std =====")
    print(f"Acc      : {df['best_acc'].mean():.4f} ± {df['best_acc'].std():.4f}")
    print(f"Macro-F1 : {df['best_macro_f1'].mean():.4f} ± {df['best_macro_f1'].std():.4f}")

    summary_path = save_dir / "teacher_swin_10fold_results.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved results to: {summary_path}")
    print(f"Teacher weights saved to: {save_dir}")
    print("Expected files: best_model_fold1.pth ... best_model_fold10.pth")


if __name__ == "__main__":
    main()