#!/usr/bin/env python
"""
export_referit_csv.py

Iterate through Referit3DDataset and export a CSV containing:
    data_idx, sentence, tgt_object_id, tgt_object_label,
    is_multiple, is_view_dependent, is_hard
"""

import csv
import os
from pathlib import Path

import torch   # only to suppress “unused import” linters
from dataset.referit3d import Referit3DDataset   # adapt if your path is different

# --------------------------------------------------------------------------- #
# 1. Instantiate the dataset                                                   #
#    – change the args below if you want a different split or settings.        #
# --------------------------------------------------------------------------- #
dataset = Referit3DDataset(
    split="val",          # "train" | "val" | "test"
    anno_type="sr3d",       # "nr3d"  | "sr3d"
    max_obj_len=60,
    num_points=1024,
    pc_type="gt",           # "gt"    | "pred"   (forced to "gt" for train)
    sem_type="607",
    filter_lang=False,
    sr3d_plus_aug=True,
)

# --------------------------------------------------------------------------- #
# 2. Prepare the output path                                                   #
# --------------------------------------------------------------------------- #
out_csv = Path(f"referit3d_{dataset.split}_{dataset.anno_type}_meta.csv").expanduser()
out_csv.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 3. Walk through the dataset and write each row                               #
# --------------------------------------------------------------------------- #
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # header
    writer.writerow(
        [
            "data_idx",
            "sentence",
            "tgt_object_id",
            "tgt_object_label",
            "is_multiple",
            "is_view_dependent",
            "is_hard",
        ]
    )

    for sample in dataset:  # direct iteration avoids DataLoader collate quirks
        writer.writerow(
            [
                sample["data_idx"],
                sample["sentence"],
                sample["tgt_object_id"].item(),
                sample["tgt_object_label"].item(),
                int(sample["is_multiple"]),        # bool → 0/1 for easy CSV usage
                int(sample["is_view_dependent"]),
                int(sample["is_hard"]),
            ]
        )

print(f"✓ CSV saved to {out_csv.resolve()}")
