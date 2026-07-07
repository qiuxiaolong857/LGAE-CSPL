import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    prob_cols = [f"p{i}" for i in range(args.num_classes)]
    for c in prob_cols:
        if c not in df.columns:
            raise ValueError(f"Column {c} not found in {in_csv}")

    if len(df) != args.num_classes:
        raise ValueError(
            f"Expected {args.num_classes} rows, but got {len(df)} rows."
        )

    probs = df[prob_cols].values.astype(np.float32)

    # ------------------------------------------------------------
    # 1. Shuffled prototype control
    # ------------------------------------------------------------
    # Class ID mapping in your training script:
    # 0: fruitlet
    # 1: hard
    # 2: mature
    # 3: first_dilatation
    # 4: growing
    # 5: second_dilatation
    #
    # This permutation shifts prototypes along the developmental order:
    # fruitlet <- first_dilatation
    # first_dilatation <- growing
    # growing <- hard
    # hard <- second_dilatation
    # second_dilatation <- mature
    # mature <- fruitlet
    #
    # In class-id row order, this is:
    # row 0 <- old row 3
    # row 1 <- old row 5
    # row 2 <- old row 0
    # row 3 <- old row 4
    # row 4 <- old row 1
    # row 5 <- old row 2
    shuffle_perm = [3, 5, 0, 4, 1, 2]

    df_shuffled = df.copy()
    df_shuffled[prob_cols] = probs[shuffle_perm]
    df_shuffled[prob_cols] = (
        df_shuffled[prob_cols].values
        / df_shuffled[prob_cols].values.sum(axis=1, keepdims=True)
    )

    shuffled_path = out_dir / "swin_q_table_shuffled.csv"
    df_shuffled.to_csv(shuffled_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 2. Random soft-label control
    # ------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    random_probs = rng.dirichlet(
        alpha=np.ones(args.num_classes),
        size=args.num_classes
    ).astype(np.float32)

    df_random = df.copy()
    df_random[prob_cols] = random_probs
    df_random[prob_cols] = (
        df_random[prob_cols].values
        / df_random[prob_cols].values.sum(axis=1, keepdims=True)
    )

    random_path = out_dir / "swin_q_table_random.csv"
    df_random.to_csv(random_path, index=False, encoding="utf-8-sig")

    print("Saved shuffled q-table to:", shuffled_path)
    print("Saved random q-table to:", random_path)

    print("\nShuffled q-table:")
    print(df_shuffled[prob_cols])

    print("\nRandom q-table:")
    print(df_random[prob_cols])


if __name__ == "__main__":
    main()