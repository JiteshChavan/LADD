import json
import os

input_jsonl = "/mnt/e/purple/datasets/prepare/jdb/raw/train/train_anno_realease_repath.jsonl"
images_dir = "/mnt/e/purple/datasets/prepare/jdb/raw/train/imgs"
output_jsonl = "/mnt/e/purple/datasets/prepare/jdb/raw/train/debug_2_shard006.jsonl"

N = 2
count = 0

with open(input_jsonl, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
    for line in f_in:
        d = json.loads(line)
        relpath = d["img_path"].strip("./")

        if not relpath.startswith("006/"):
            continue

        abs_path = os.path.join(images_dir, relpath)
        if not os.path.isfile(abs_path):
            continue

        f_out.write(line)
        count += 1

        if count >= N:
            break

print(f"Wrote {count} samples to {output_jsonl}")