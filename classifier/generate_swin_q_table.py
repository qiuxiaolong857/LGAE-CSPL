import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T
from PIL import Image
import pandas as pd
import timm


NUM_CLASSES = 6


class SimpleDataset(Dataset):
    def __init__(self, samples, tf):
        self.samples = samples
        self.tf = tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y


def load_teacher(ckpt_path, device, default_model_name="swin_tiny_patch4_window7_224"):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_name = ckpt.get("model_name", default_model_name)

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(ckpt["model"], strict=True)

    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print(f"Loaded teacher: {ckpt_path}")
    print(f"Model name: {model_name}")

    return model


def load_teachers(ckpt_list, device, default_model_name="swin_tiny_patch4_window7_224"):
    teachers = []

    for p in ckpt_list:
        teachers.append(
            load_teacher(
                ckpt_path=p,
                device=device,
                default_model_name=default_model_name
            )
        )

    return teachers


def read_splits(splits_json, fold_id, split="train"):
    obj = json.loads(Path(splits_json).read_text(encoding="utf-8"))

    root = Path(obj["root_dir"])
    folds = obj["folds"]

    if split == "train":
        items = folds[fold_id]["train"]
    elif split == "val":
        items = folds[fold_id]["val"]
    elif split == "all_train":
        items = []
        for f in folds:
            items.extend(f["train"])
    else:
        raise ValueError("split must be train / val / all_train")

    samples = [
        (root / it["path"], int(it["label"]))
        for it in items
    ]

    return samples


def build_q_table_from_splits(
    splits_json,
    fold_id,
    split,
    ckpt_list,
    teacher_model_name="swin_tiny_patch4_window7_224",
    Ttemp=4.0,
    out_csv=None,
    batch_size=16,
    num_workers=2,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teachers = load_teachers(
        ckpt_list=ckpt_list,
        device=device,
        default_model_name=teacher_model_name
    )

    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    samples = read_splits(
        splits_json=splits_json,
        fold_id=fold_id,
        split=split
    )

    print(f"Number of samples for q-table: {len(samples)}")

    ds = SimpleDataset(samples, tf)

    ld = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    q_sum = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    cnt = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for x, y in ld:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y)

            logits_sum = None

            for teacher in teachers:
                logits = teacher(x)
                logits_sum = logits if logits_sum is None else logits_sum + logits

            logits_avg = logits_sum / len(teachers)

            probs = torch.softmax(logits_avg / Ttemp, dim=1).cpu()

            for i in range(probs.size(0)):
                k = int(y[i])
                q_sum[k] += probs[i]
                cnt[k] += 1

    q = q_sum / (cnt[:, None] + 1e-12)
    q = q / (q.sum(dim=1, keepdim=True) + 1e-12)

    if out_csv is None:
        out_csv = f"swin_q_table_fold{fold_id}_{split}.csv"

    df = pd.DataFrame(
        q.numpy(),
        columns=[f"p{j}" for j in range(NUM_CLASSES)]
    )

    df.insert(0, "stage", list(range(NUM_CLASSES)))
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:", out_csv)
    print(df)

    return q


if __name__ == "__main__":
    ckpts = [
        f"D:\\python_space\\ultralytics-yolo11-main-origin\\classifier\\teacher_model\\weight_teacher_swin\\best_model_fold{i}.pth"
        for i in range(1, 11)
    ]

    build_q_table_from_splits(
        splits_json="D:\\python_space\\ultralytics-yolo11-main-origin\\classifier\\splits.json",
        fold_id=0,
        split="train",
        ckpt_list=ckpts,
        teacher_model_name="swin_tiny_patch4_window7_224",
        Ttemp=4.0,
        out_csv="D:\\python_space\\ultralytics-yolo11-main-origin\\classifier\\swin_q_table_fold_train.csv",
        batch_size=16,
        num_workers=2
    )