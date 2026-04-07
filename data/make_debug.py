#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path

"""
use as follows:
python make_debug.py \
  --jsonl /mnt/e/Purple/data/raw/train/train_anno_realease_repath.jsonl \
  --imgs_root /mnt/e/Purple/data/raw/train/imgs \
  --out_dir /mnt/e/Purple/data/debug \
  --debug_split_samples 5 \
  --caption_mode prompt \
  --shard 006 \
  --max_scan 500
"""


def pick_caption(row: dict, mode: str) -> str:
    if mode == "prompt":
        cap = row.get("prompt", "")
    elif mode == "task2":
        cap = row.get("Task2", {}).get("Caption", "")
    elif mode == "prefer_task2":
        cap = row.get("Task2", {}).get("Caption", "") or row.get("prompt", "")
    else:
        raise ValueError(f"Unknown caption mode: {mode}")
    return str(cap).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--imgs_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--debug_split_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--caption_mode",
        type=str,
        default="prompt",
        choices=["prompt", "task2", "prefer_task2"],
    )
    parser.add_argument("--shard", type=str, default=None, help="e.g. 006")
    parser.add_argument(
        "--max_scan",
        type=int,
        default=5000,
        help="Stop scanning after this many matching rows",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    jsonl_path = Path(args.jsonl).resolve()
    imgs_root = Path(args.imgs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_imgs = out_dir / "images"
    out_imgs.mkdir(parents=True, exist_ok=True)

    reservoir = []
    scanned = 0
    matched_shard = 0
    valid = 0
    missing_img = 0
    missing_caption = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            scanned += 1

            rel_img_raw = row.get("img_path", "")
            if not isinstance(rel_img_raw, str) or not rel_img_raw.strip():
                continue

            if args.shard is not None:
                if not (
                    rel_img_raw.startswith(f"./{args.shard}/")
                    or rel_img_raw.startswith(f"{args.shard}/")
                ):
                    continue

            matched_shard += 1

            rel_img = rel_img_raw.lstrip("./")
            img_path = imgs_root / rel_img
            if not img_path.exists():
                missing_img += 1
                continue

            caption = pick_caption(row, args.caption_mode)
            if not caption:
                missing_caption += 1
                continue

            valid += 1

            item = (row, img_path, caption)

            if len(reservoir) < args.debug_split_samples:
                reservoir.append(item)
            else:
                j = rng.randint(0, valid - 1)
                if j < args.debug_split_samples:
                    reservoir[j] = item

            if matched_shard >= args.max_scan:
                break

            if valid % 100 == 0:
                print(
                    f"scanned={scanned} matched_shard={matched_shard} valid={valid}",
                    flush=True,
                )

    if not reservoir:
        raise RuntimeError("No valid samples found.")

    shard_suffix = f"_shard{args.shard}" if args.shard else ""
    out_jsonl = out_dir / f"debug_{len(reservoir)}{shard_suffix}.jsonl"

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, (row, src_img, caption) in enumerate(reservoir):
            dst_name = f"{i:06d}{src_img.suffix.lower()}"
            dst_img = out_imgs / dst_name
            shutil.copy2(src_img, dst_img)

            record = {
                "id": i,
                "image": f"images/{dst_name}",
                "caption": caption,
                "source_img_path": row.get("img_path"),
                "source_prompt": row.get("prompt"),
                "source_task2_caption": row.get("Task2", {}).get("Caption", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Total scanned: {scanned}")
    print(f"Rows matching shard filter: {matched_shard}")
    print(f"Valid rows: {valid}")
    print(f"Missing images: {missing_img}")
    print(f"Missing captions: {missing_caption}")
    print(f"Wrote images to: {out_imgs}")
    print(f"Wrote metadata to: {out_jsonl}")


if __name__ == "__main__":
    main()