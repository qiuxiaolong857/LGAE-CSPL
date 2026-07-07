# classifier/train_student_swin_hier_kd.py
"""
Swin-Transformer Student training with Hierarchical classification and KD.

This script is adapted from your original classifier/train_student.py:
- Teacher ensemble: timm Swin checkpoints trained by train_teacher_swin_kfold.py
- Student backbone: timm Swin Transformer
- Keeps original KD, q-table mixing, warmup/ramp, and hierarchical coarse/fine loss
- Supports 3 seeds: 0, 42, 2024
"""

import os
import re
import glob
import argparse
import random
from pathlib import Path

import cv2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from scenario_aware_augmentation_paper_v2 import get_classification_train_transform

try:
    from safetensors.torch import load_file
except Exception:
    load_file = None

from dataset import ROIFolderDataset


ID2NAME = {
    0: "fruitlet",
    1: "hard",
    2: "mature",
    3: "first_dilatation",
    4: "growing",
    5: "second_dilatation",
}

ordinal_names = [
    "fruitlet",
    "first_dilatation",
    "growing",
    "hard",
    "second_dilatation",
    "mature",
]

EPS = 1e-8

# 与你原始 Hier 代码保持一致：
# coarse0: fruitlet / first_dilatation
# coarse1: growing / hard
# coarse2: second_dilatation / mature
PAIR_IDS = {
    0: (0, 3),
    1: (4, 1),
    2: (5, 2),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_key(path):
    """Make best_model_fold10 sort after best_model_fold9."""
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def find_teacher_ckpts(teacher_dir, pattern="best_model_fold*.pth"):
    paths = sorted(
        glob.glob(os.path.join(teacher_dir, pattern)),
        key=natural_key
    )
    if len(paths) == 0:
        raise FileNotFoundError(f"No teacher ckpts found in {teacher_dir} with pattern={pattern}")

    print("[Teacher] ckpts:")
    for p in paths:
        print("  ", p)

    return paths


def load_safetensors_pretrained(model, weights_path, prefix=""):
    """
    Load local timm safetensors weights.
    It only loads keys with identical name and shape.
    Classification head mismatch will be skipped automatically.
    """
    if weights_path is None or weights_path == "":
        print("[Pretrain] empty weights path, use random init.")
        return model

    if not os.path.exists(weights_path):
        print(f"[Pretrain] file not found: {weights_path}")
        print("[Pretrain] use random init.")
        return model

    if load_file is None:
        raise ImportError("safetensors is not installed. Run: pip install safetensors")

    state_dict = load_file(weights_path)
    model_dict = model.state_dict()

    matched = {}

    for k, v in state_dict.items():
        key = k
        if key.startswith("model."):
            key = key[len("model."):]

        if prefix:
            key = prefix + key

        if key in model_dict and v.shape == model_dict[key].shape:
            matched[key] = v

    model_dict.update(matched)
    model.load_state_dict(model_dict)

    print(f"[Pretrain] loaded: {weights_path}")
    print(f"[Pretrain] matched params: {len(matched)} / {len(model_dict)}")

    return model


def qstage_to_qcoarse_qfine(q_stage_6: torch.Tensor):
    """
    q_stage_6: (B, 6), each row sums to 1.
    returns:
      q_coarse: (B, 3)
      q_fine  : (B, 3, 2)
    """
    B = q_stage_6.size(0)

    q_coarse = torch.zeros(
        B, 3,
        device=q_stage_6.device,
        dtype=q_stage_6.dtype
    )

    q_fine = torch.zeros(
        B, 3, 2,
        device=q_stage_6.device,
        dtype=q_stage_6.dtype
    )

    for g in range(3):
        a, b = PAIR_IDS[g]
        qa = q_stage_6[:, a]
        qb = q_stage_6[:, b]
        qc = qa + qb

        q_coarse[:, g] = qc

        denom = qc + EPS
        q_fine[:, g, 0] = qa / denom
        q_fine[:, g, 1] = qb / denom

    q_coarse = q_coarse / (q_coarse.sum(dim=1, keepdim=True) + EPS)

    return q_coarse, q_fine


def kl_distill(student_logits: torch.Tensor, teacher_probs: torch.Tensor, T: float):
    """
    student_logits: (B, C)
    teacher_probs : (B, C), probability distribution
    """
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(log_p_s, teacher_probs, reduction="batchmean") * (T * T)

class AlbumentationsClassificationTransform:
    """
    Apply Albumentations augmentation to PIL image and convert it to normalized tensor.
    This transform is used only for training ROI classification images.
    """

    def __init__(self, image_size=224):
        self.aug = get_classification_train_transform(image_size=image_size)

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(3, 1, 1)

    def __call__(self, img: Image.Image):
        # PIL RGB -> numpy RGB
        img = np.array(img.convert("RGB"))

        # Albumentations augmentation
        out = self.aug(image=img)
        img = out["image"]

        # numpy RGB uint8 -> torch tensor, [0, 1]
        img = np.ascontiguousarray(img)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # ImageNet normalization
        x = (x - self.mean) / self.std

        return x


class EnsembleTeacher(nn.Module):
    """
    10-fold timm Swin teacher ensemble.
    forward returns stage-6 probability distribution: (B, 6).

    This class is compatible with teacher ckpts saved by train_teacher_swin_kfold.py:
      ckpt["model"]
      ckpt["model_name"]
    """

    def __init__(
        self,
        ckpt_paths,
        device="cuda",
        num_classes=6,
        teacher_model_name="swin_tiny_patch4_window7_224",
        use_fp16=True,
    ):
        super().__init__()

        self.device = device
        self.num_classes = num_classes
        self.teacher_model_name = teacher_model_name
        self.use_fp16 = use_fp16 and device.startswith("cuda")

        self.models = nn.ModuleList()

        for p in ckpt_paths:
            ckpt = torch.load(p, map_location="cpu")
            model_name = ckpt.get("model_name", teacher_model_name)

            m = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes
            )

            m.load_state_dict(ckpt["model"], strict=True)
            m.eval().to(device)

            for param in m.parameters():
                param.requires_grad = False

            self.models.append(m)

            print(f"[Teacher] loaded {p}")
            print(f"[Teacher] model_name={model_name}")

        if len(self.models) == 0:
            raise RuntimeError("No teacher models loaded!")

    @torch.no_grad()
    def forward(self, x, T=4.0):
        logits_sum = None

        if self.use_fp16:
            with torch.cuda.amp.autocast(True):
                for m in self.models:
                    logits = m(x)
                    logits_sum = logits if logits_sum is None else logits_sum + logits
        else:
            for m in self.models:
                logits = m(x)
                logits_sum = logits if logits_sum is None else logits_sum + logits

        logits_avg = logits_sum / len(self.models)
        probs = torch.softmax(logits_avg / T, dim=1)
        return probs


class SwinFlatStudent(nn.Module):
    """
    Swin flat student:
    output: logits (B, 6)
    """

    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        num_classes=6,
        pretrained_path=None,
    ):
        super().__init__()

        self.model_name = model_name
        self.net = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )

        self.net = load_safetensors_pretrained(
            self.net,
            pretrained_path,
            prefix=""
        )

    def forward(self, x):
        return self.net(x)


class SwinHierStudent(nn.Module):
    """
    Swin hierarchical student:
    output:
      coarse_logits   : (B, 3)
      fine_logits_all : (B, 3, 2)
    """

    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        num_classes=6,
        pretrained_path=None,
    ):
        super().__init__()

        self.model_name = model_name

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        self.backbone = load_safetensors_pretrained(
            self.backbone,
            pretrained_path,
            prefix=""
        )

        in_dim = self.backbone.num_features

        self.coarse_head = nn.Linear(in_dim, 3)
        self.fine_heads = nn.ModuleList([
            nn.Linear(in_dim, 2),
            nn.Linear(in_dim, 2),
            nn.Linear(in_dim, 2),
        ])

    def forward(self, x):
        feat = self.backbone(x)

        coarse_logits = self.coarse_head(feat)

        fine_logits_all = torch.stack(
            [head(feat) for head in self.fine_heads],
            dim=1
        )

        return coarse_logits, fine_logits_all


@torch.no_grad()
def evaluate_hier(model, loader, device, num_classes, fine_to_coarse, fine_to_within):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    pair_to_fine = {}
    for fine_id in range(num_classes):
        c = int(fine_to_coarse[fine_id].item())
        w = int(fine_to_within[fine_id].item())
        pair_to_fine[(c, w)] = fine_id

    pbar = tqdm(loader, desc="Eval", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        coarse_logits, fine_logits_all = model(x)

        coarse_y = fine_to_coarse[y]
        within_y = fine_to_within[y]

        loss_coarse = ce(coarse_logits, coarse_y)

        idx = coarse_y.view(-1, 1, 1).expand(-1, 1, 2)
        fine_logits = fine_logits_all.gather(1, idx).squeeze(1)

        loss_fine = ce(fine_logits, within_y)

        loss = 1.6 * loss_coarse + 1.0 * loss_fine

        bs = x.size(0)
        total_loss += loss.item() * bs

        coarse_pred = coarse_logits.argmax(dim=1)
        idxp = coarse_pred.view(-1, 1, 1).expand(-1, 1, 2)
        fine_logits_p = fine_logits_all.gather(1, idxp).squeeze(1)
        within_pred = fine_logits_p.argmax(dim=1)

        pred = torch.empty_like(y)
        for i in range(bs):
            pred[i] = pair_to_fine[
                (int(coarse_pred[i].item()), int(within_pred[i].item()))
            ]

        total_correct += (pred == y).sum().item()
        total += bs

        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    macro_f1 = macro_f1_from_cm(cm, num_classes)

    return avg_loss, acc, macro_f1, cm


@torch.no_grad()
def evaluate_flat(model, loader, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    pbar = tqdm(loader, desc="Eval", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

        loss = ce(logits, y)
        pred = logits.argmax(dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (pred == y).sum().item()
        total += bs

        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    macro_f1 = macro_f1_from_cm(cm, num_classes)

    return avg_loss, acc, macro_f1, cm


def macro_f1_from_cm(cm, num_classes):
    eps = 1e-12
    f1s = []

    for k in range(num_classes):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        f1s.append(f1)

    return sum(f1s) / len(f1s)


class LetterboxResize:
    """Equal-ratio resize + padding to fixed size."""

    def __init__(self, new_size=224, pad_value=(114, 114, 114)):
        self.new_size = new_size
        self.pad_value = pad_value

    def __call__(self, img: Image.Image):
        img = np.array(img)
        h, w = img.shape[:2]

        scale = self.new_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        top = (self.new_size - nh) // 2
        bottom = self.new_size - nh - top
        left = (self.new_size - nw) // 2
        right = self.new_size - nw - left

        img_padded = cv2.copyMakeBorder(
            img_resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.pad_value
        )

        return Image.fromarray(img_padded)


def build_weighted_sampler(ds: ROIFolderDataset, num_classes: int):
    labels = [y for _, y in ds.samples]

    counts = torch.bincount(
        torch.tensor(labels),
        minlength=num_classes
    ).float()

    counts = torch.clamp(counts, min=1.0)

    class_weights = 1.0 / counts
    sample_weights = [class_weights[y].item() for y in labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler, counts


def get_alpha_eff(epoch: int, alpha: float, warmup: int, ramp: int):
    if alpha <= 0:
        return 0.0
    if epoch <= warmup:
        return 0.0
    if ramp <= 0:
        return float(alpha)

    t = (epoch - warmup) / float(ramp)
    t = max(0.0, min(1.0, t))

    return float(alpha) * t


def get_beta_eff(epoch: int, beta_final: float, delay: int, ramp: int):
    if beta_final <= 0:
        return 0.0
    if epoch <= delay:
        return 0.0
    if ramp <= 0:
        return float(beta_final)

    t = (epoch - delay) / float(ramp)
    t = max(0.0, min(1.0, t))

    return float(beta_final) * t


def build_hier_mapping(train_ds, num_classes, device):
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    groups = [
        ["fruitlet", "first_dilatation"],
        ["growing", "hard"],
        ["second_dilatation", "mature"],
    ]

    name_to_group = {}
    for g, names in enumerate(groups):
        for w, n in enumerate(names):
            name_to_group[n] = (g, w)

    fine_to_coarse = torch.zeros(num_classes, dtype=torch.long)
    fine_to_within = torch.zeros(num_classes, dtype=torch.long)

    for fine_id in range(num_classes):
        cls_name = idx_to_class[fine_id]
        if cls_name not in name_to_group:
            raise RuntimeError(f"[Hier] class '{cls_name}' not in groups={groups}")

        c, w = name_to_group[cls_name]
        fine_to_coarse[fine_id] = c
        fine_to_within[fine_id] = w

    return fine_to_coarse.to(device), fine_to_within.to(device)


def build_dataloaders(args):
    # train_tf = transforms.Compose([
    #     LetterboxResize(args.imgsz),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.2),
    #     transforms.ColorJitter(
    #         brightness=0.25,
    #         contrast=0.25,
    #         saturation=0.25,
    #         hue=0.03
    #     ),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=(0.485, 0.456, 0.406),
    #         std=(0.229, 0.224, 0.225)
    #     ),
    # ])
    train_tf = AlbumentationsClassificationTransform(
        image_size=args.imgsz
    )

    val_tf = transforms.Compose([
        LetterboxResize(args.imgsz),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_root = os.path.join(args.data_root, "train", args.iou_level)
    val_root = os.path.join(args.data_root, "val", args.iou_level)
    test_iou_level = args.test_iou_level if args.test_iou_level is not None else args.iou_level
    test_root = os.path.join(args.data_root, "test", test_iou_level)

    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"train_root not found: {train_root}")
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f"val_root not found: {val_root}")
    if not os.path.isdir(test_root):
        print(f"[WARN] test_root not found: {test_root}")
        print("[WARN] test_loader will be None.")
        test_root = None

    NAME2ID = {v: k for k, v in ID2NAME.items()}

    train_ds = ROIFolderDataset(
        train_root,
        transform=train_tf,
        class_to_idx=NAME2ID
    )

    val_ds = ROIFolderDataset(
        val_root,
        transform=val_tf,
        class_to_idx=NAME2ID
    )

    test_ds = None
    if test_root is not None:
        test_ds = ROIFolderDataset(
            test_root,
            transform=val_tf,
            class_to_idx=NAME2ID
        )

    print(f"[DATA] data_root = {args.data_root}")
    print(f"[DATA] iou_level = {args.iou_level}")
    print(f"[DATA] test_iou_level = {args.test_iou_level if args.test_iou_level is not None else args.iou_level}")
    print(f"[DATA] train_root = {train_root}")
    print(f"[DATA] val_root   = {val_root}")
    print(f"[DATA] test_root  = {test_root}")
    print(f"[DATA] class_to_idx = {train_ds.class_to_idx}")

    sampler = None
    shuffle = True

    if args.use_sampler:
        sampler, counts = build_weighted_sampler(train_ds, args.num_classes)
        shuffle = False
        print(f"[Sampler] class counts = {counts.tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )

    return train_ds, train_loader, val_loader, test_loader


def build_student(args, device):
    if args.use_hier:
        model = SwinHierStudent(
            model_name=args.student_model_name,
            num_classes=args.num_classes,
            pretrained_path=args.student_pretrained_path
        ).to(device)
    else:
        model = SwinFlatStudent(
            model_name=args.student_model_name,
            num_classes=args.num_classes,
            pretrained_path=args.student_pretrained_path
        ).to(device)

    return model


def train_one_seed(
    seed,
    args,
    device,
    train_ds,
    train_loader,
    val_loader,
    test_loader,
    fine_to_coarse,
    fine_to_within,
    teacher,
    q_table,
    save_dir,
):
    print(f"\n========== Seed {seed} ==========")

    set_seed(seed)

    model = build_student(args, device)

    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    use_amp = bool(args.amp) and device == "cuda"

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_path = save_dir / f"best_seed{seed}.pt"
    last_path = save_dir / f"last_seed{seed}.pt"

    start_epoch = 1

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        if "scaler" in ckpt and scaler is not None and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])

        print(f"[RESUME] loaded {last_path}, start_epoch={start_epoch}")
    else:
        print(f"[RESUME] seed={seed} starts from scratch")

    best_acc = 0.0
    best_f1 = 0.0

    history = []

    amp_device_type = "cuda" if device == "cuda" else "cpu"

    for epoch in range(start_epoch, args.epochs + 1):
        alpha_eff = get_alpha_eff(
            epoch,
            args.distill_alpha,
            args.distill_warmup,
            args.distill_ramp
        )

        beta_eff = get_beta_eff(
            epoch,
            args.proto_beta,
            args.qtable_delay,
            args.qtable_ramp
        )

        model.train()

        total_loss = 0.0
        total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Seed {seed} Epoch {epoch:03d}/{args.epochs}",
            leave=True,
            dynamic_ncols=True
        )

        for x, y in train_bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=amp_device_type,
                enabled=use_amp
            ):
                if args.use_hier:
                    coarse_logits, fine_logits_all = model(x)

                    coarse_y = fine_to_coarse[y]
                    within_y = fine_to_within[y]

                    loss_coarse = ce(coarse_logits, coarse_y)

                    idx = coarse_y.view(-1, 1, 1).expand(-1, 1, 2)
                    fine_logits = fine_logits_all.gather(1, idx).squeeze(1)

                    loss_fine = ce(fine_logits, within_y)

                    loss = args.coarse_loss_weight * loss_coarse + loss_fine

                    if args.use_kd and alpha_eff > 0:
                        if teacher is None:
                            raise RuntimeError("use_kd=1 requires teacher model")

                        with torch.no_grad():
                            p_teacher = teacher(x, T=args.distill_T)
                            p_teacher = p_teacher / (
                                p_teacher.sum(dim=1, keepdim=True) + EPS
                            )

                            if q_table is not None:
                                p_proto = q_table[y]
                                q_stage_6 = (1.0 - beta_eff) * p_teacher + beta_eff * p_proto
                            else:
                                q_stage_6 = p_teacher

                            q_stage_6 = q_stage_6 / (
                                q_stage_6.sum(dim=1, keepdim=True) + EPS
                            )

                            q_coarse, q_fine = qstage_to_qcoarse_qfine(q_stage_6)

                        kd_coarse = kl_distill(coarse_logits, q_coarse, T=args.distill_T)

                        kd_fine = 0.0
                        for g in range(3):
                            w = q_coarse[:, g].detach()
                            log_p = F.log_softmax(
                                fine_logits_all[:, g, :] / args.distill_T,
                                dim=-1
                            )

                            kl_per = F.kl_div(
                                log_p,
                                q_fine[:, g, :],
                                reduction="none"
                            ).sum(dim=1) * (args.distill_T * args.distill_T)

                            kd_fine = kd_fine + (w * kl_per).mean()

                        loss = loss + alpha_eff * (kd_coarse + kd_fine)

                else:
                    logits = model(x)

                    loss = ce(logits, y)
                    loss_coarse = torch.tensor(0.0, device=device)
                    loss_fine = loss

                    if args.use_kd and alpha_eff > 0:
                        if teacher is None:
                            raise RuntimeError("use_kd=1 requires teacher model")

                        with torch.no_grad():
                            p_teacher = teacher(x, T=args.distill_T)
                            p_teacher = p_teacher / (
                                p_teacher.sum(dim=1, keepdim=True) + EPS
                            )

                            if q_table is not None:
                                p_proto = q_table[y]
                                q_stage_6 = (1.0 - beta_eff) * p_teacher + beta_eff * p_proto
                            else:
                                q_stage_6 = p_teacher

                            q_stage_6 = q_stage_6 / (
                                q_stage_6.sum(dim=1, keepdim=True) + EPS
                            )

                        loss = loss + alpha_eff * kl_distill(
                            logits,
                            q_stage_6,
                            T=args.distill_T
                        )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip
            )

            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

            train_loss = total_loss / max(total, 1)

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                c=f"{loss_coarse.item():.4f}",
                f=f"{loss_fine.item():.4f}",
                avg=f"{train_loss:.4f}",
                alpha=f"{alpha_eff:.3f}",
                beta=f"{beta_eff:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        scheduler.step()

        if args.use_hier:
            val_loss, val_acc, val_f1, val_cm = evaluate_hier(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=args.num_classes,
                fine_to_coarse=fine_to_coarse,
                fine_to_within=fine_to_within
            )
        else:
            val_loss, val_acc, val_f1, val_cm = evaluate_flat(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=args.num_classes
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macroF1={val_f1:.4f}"
        )

        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "alpha_eff": alpha_eff,
            "beta_eff": beta_eff,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc

            torch.save({
                "model": model.state_dict(),
                "student_model_name": args.student_model_name,
                "teacher_model_name": args.teacher_model_name,
                "class_to_idx": train_ds.class_to_idx,
                "imgsz": args.imgsz,
                "num_classes": args.num_classes,
                "best_macro_f1": best_f1,
                "best_acc": best_acc,
                "ordinal_names": ordinal_names,
                "seed": seed,
                "use_hier": args.use_hier,
                "use_kd": args.use_kd,
            }, best_path)

            np.save(save_dir / f"val_cm_best_seed{seed}.npy", val_cm.cpu().numpy())

            print(f"  saved best to {best_path} (macroF1={best_f1:.4f})")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "student_model_name": args.student_model_name,
            "teacher_model_name": args.teacher_model_name,
            "class_to_idx": train_ds.class_to_idx,
            "imgsz": args.imgsz,
            "num_classes": args.num_classes,
            "epoch": epoch,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "ordinal_names": ordinal_names,
            "seed": seed,
            "use_hier": args.use_hier,
            "use_kd": args.use_kd,
        }, last_path)

        print(f"  saved last to {last_path}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        save_dir / f"history_seed{seed}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_acc = None
    test_f1 = None

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    if test_loader is not None:
        if args.use_hier:
            test_loss, test_acc, test_f1, test_cm = evaluate_hier(
                model=model,
                loader=test_loader,
                device=device,
                num_classes=args.num_classes,
                fine_to_coarse=fine_to_coarse,
                fine_to_within=fine_to_within
            )
        else:
            test_loss, test_acc, test_f1, test_cm = evaluate_flat(
                model=model,
                loader=test_loader,
                device=device,
                num_classes=args.num_classes
            )

        np.save(save_dir / f"test_cm_best_seed{seed}.npy", test_cm.cpu().numpy())

        print(
            f"[TEST] seed={seed} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"test_macroF1={test_f1:.4f}"
        )

    return {
        "seed": seed,
        "best_val_acc": best_acc,
        "best_val_macro_f1": best_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default=r"/root/autodl-tmp/ultralytics-yolo11-main-origin/roi_dataset/yolov11-legm-roi_dataset",
        help="dataset root containing train/val/test"
    )

    parser.add_argument(
        "--iou_level",
        type=str,
        default="iou0p6",
        choices=["iou0p2", "iou0p4", "iou0p6", "iou0p8", "iou1p0", "iou2p0"]
    )

    parser.add_argument("--test_iou_level", type=str, default=None, choices=["iou0p2", "iou0p4", "iou0p6", "iou0p8", "iou1p0", "iou2p0"], help="test split iou level; if None, use --iou_level")

    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use_sampler",
        type=int,
        default=1,
        help="1=use WeightedRandomSampler, 0=shuffle"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"/root/autodl-tmp/ultralytics-yolo11-main-origin/runs/roi_classifier/Swin_Hier_KD_iou2p0"
    )

    parser.add_argument(
        "--student_model_name",
        type=str,
        default="swin_tiny_patch4_window7_224"
    )

    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="swin_tiny_patch4_window7_224"
    )

    parser.add_argument(
        "--student_pretrained_path",
        type=str,
        default=r"/root/autodl-tmp/pretrained/Swim_Transformer/model.safetensors",
        help="local timm Swin pretrained safetensors"
    )

    parser.add_argument(
        "--teacher_dir",
        type=str,
        default=r"/root/autodl-tmp/ultralytics-yolo11-main-origin/classifier/teacher_model/weight_teacher_swin",
        help="folder containing best_model_fold*.pth"
    )

    parser.add_argument(
        "--teacher_pattern",
        type=str,
        default="best_model_fold*.pth"
    )

    parser.add_argument(
        "--use_online_teacher",
        type=int,
        default=1,
        help="1=use EnsembleTeacher, 0=do not use teacher"
    )

    parser.add_argument("--use_hier", type=int, default=1)
    parser.add_argument("--use_kd", type=int, default=1)

    parser.add_argument("--distill_alpha", type=float, default=0.3)
    parser.add_argument("--distill_T", type=float, default=4.0)
    parser.add_argument("--distill_warmup", type=int, default=5)
    parser.add_argument("--distill_ramp", type=int, default=5)

    parser.add_argument(
        "--qtable_csv",
        type=str,
        default=r"/root/autodl-tmp/ultralytics-yolo11-main-origin/classifier/swin_q_table_fold_train.csv",
        help="q-table csv, columns: p0..p(C-1), rows: C"
    )

    parser.add_argument("--proto_beta", type=float, default=0.2)
    parser.add_argument("--qtable_delay", type=int, default=15)
    parser.add_argument("--qtable_ramp", type=int, default=0)

    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--coarse_loss_weight", type=float, default=1.6)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False
    )

    args = parser.parse_args()

    args.use_sampler = bool(args.use_sampler)
    args.use_online_teacher = bool(args.use_online_teacher)
    args.use_hier = int(args.use_hier)
    args.use_kd = int(args.use_kd)

    if args.use_kd == 0:
        args.use_online_teacher = False
        args.proto_beta = 0.0
        args.distill_alpha = 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== Swin Student Training ==========")
    print(f"[ARGS] data_root = {args.data_root}")
    print(f"[ARGS] save_dir = {args.save_dir}")
    print(f"[ARGS] student_model_name = {args.student_model_name}")
    print(f"[ARGS] teacher_model_name = {args.teacher_model_name}")
    print(f"[ARGS] use_hier = {args.use_hier}")
    print(f"[ARGS] use_kd = {args.use_kd}")
    print(f"[ARGS] device = {device}")

    train_ds, train_loader, val_loader, test_loader = build_dataloaders(args)

    fine_to_coarse, fine_to_within = build_hier_mapping(
        train_ds=train_ds,
        num_classes=args.num_classes,
        device=device
    )

    q_table = None

    if args.proto_beta > 0:
        q_csv = Path(args.qtable_csv)

        if not q_csv.exists():
            raise FileNotFoundError(f"qtable_csv not found: {q_csv}")

        df = pd.read_csv(str(q_csv))

        q_table = torch.tensor(
            df[[f"p{i}" for i in range(args.num_classes)]].values,
            dtype=torch.float32,
            device=device
        )

        q_table = q_table / (q_table.sum(dim=1, keepdim=True) + EPS)

        print("[Q] loaded:", q_csv, "shape=", tuple(q_table.shape))
    else:
        print("[Q] proto_beta == 0, do not use q_table")

    teacher = None

    if args.use_online_teacher:
        ckpts = find_teacher_ckpts(args.teacher_dir, args.teacher_pattern)

        print(f"[EnsembleTeacher] found {len(ckpts)} ckpts")

        teacher = EnsembleTeacher(
            ckpt_paths=ckpts,
            device=device,
            num_classes=args.num_classes,
            teacher_model_name=args.teacher_model_name,
            use_fp16=True
        )

    SEEDS = [0, 42, 2024]

    all_results = []

    for seed in SEEDS:
        result = train_one_seed(
            seed=seed,
            args=args,
            device=device,
            train_ds=train_ds,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            fine_to_coarse=fine_to_coarse,
            fine_to_within=fine_to_within,
            teacher=teacher,
            q_table=q_table,
            save_dir=save_dir,
        )

        all_results.append(result)

        pd.DataFrame(all_results).to_csv(
            save_dir / f"swin_student_hier{args.use_hier}_kd{args.use_kd}_results_3seeds.csv",
            index=False,
            encoding="utf-8-sig"
        )

    df = pd.DataFrame(all_results)

    print("\n===== Final Results (mean ± std) =====")
    print(f"Experiment: Swin-Hier{args.use_hier}-KD{args.use_kd}")

    print(
        f"Val Acc      : {df['best_val_acc'].mean():.4f} ± {df['best_val_acc'].std():.4f}"
    )
    print(
        f"Val Macro-F1 : {df['best_val_macro_f1'].mean():.4f} ± {df['best_val_macro_f1'].std():.4f}"
    )

    if df["test_acc"].notna().any():
        print(
            f"Test Acc      : {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}"
        )
        print(
            f"Test Macro-F1 : {df['test_macro_f1'].mean():.4f} ± {df['test_macro_f1'].std():.4f}"
        )

    result_path = save_dir / f"swin_student_hier{args.use_hier}_kd{args.use_kd}_results_3seeds.csv"
    df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved result csv: {result_path}")


if __name__ == "__main__":
    main()
