import os
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


class ShardedLatentsDataset(Dataset):
    """
    Loads one or more .pt shards saved like:
        {
            "samples": [
                {
                    "latents": Tensor[16, 64, 64],
                    "cap_feats": Tensor[seq_len, hidden_dim],
                    "text": str,
                    "relpath": str,
                },
                ...
            ],
            "num_samples": N,
        }

    Supports:
    - single shard path
    - list of shard paths
    - directory containing *.pt shards
    """

    def __init__(
        self,
        shard_paths: Union[str, Sequence[str]],
        load_into_memory: bool = True,
    ):
        self.load_into_memory = load_into_memory
        self.samples: List[Dict[str, Any]] = []
        self.index: List[Dict[str, Any]] = []

        resolved_paths = self._resolve_shard_paths(shard_paths)
        if len(resolved_paths) == 0:
            raise ValueError(f"No shard files found from input: {shard_paths}")

        self.shard_paths = resolved_paths

        if self.load_into_memory:
            self._load_all_shards_into_memory()
        else:
            self._build_global_index_only()

    def _resolve_shard_paths(self, shard_paths: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(shard_paths, str):
            if os.path.isdir(shard_paths):
                paths = [
                    os.path.join(shard_paths, f)
                    for f in sorted(os.listdir(shard_paths))
                    if f.endswith(".pt")
                ]
                return paths
            elif os.path.isfile(shard_paths):
                return [shard_paths]
            else:
                raise FileNotFoundError(f"Path not found: {shard_paths}")

        paths = []
        for p in shard_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Shard not found: {p}")
            paths.append(p)
        return list(paths)

    def _load_all_shards_into_memory(self):
        total = 0
        for shard_id, shard_path in enumerate(self.shard_paths):
            obj = torch.load(shard_path, map_location="cpu")
            shard_samples = obj["samples"]

            for local_idx, sample in enumerate(shard_samples):
                self.samples.append(sample)
                self.index.append(
                    {
                        "global_idx": total,
                        "shard_id": shard_id,
                        "shard_path": shard_path,
                        "local_idx": local_idx,
                    }
                )
                total += 1

    def _build_global_index_only(self):
        total = 0
        for shard_id, shard_path in enumerate(self.shard_paths):
            obj = torch.load(shard_path, map_location="cpu")
            num_samples = int(obj["num_samples"])

            for local_idx in range(num_samples):
                self.index.append(
                    {
                        "global_idx": total,
                        "shard_id": shard_id,
                        "shard_path": shard_path,
                        "local_idx": local_idx,
                    }
                )
                total += 1

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.index[idx]

        if self.load_into_memory:
            sample = self.samples[idx]
        else:
            obj = torch.load(meta["shard_path"], map_location="cpu")
            sample = obj["samples"][meta["local_idx"]]

        latents = sample["latents"]
        cap_feats = sample["cap_feats"]
        text = sample.get("text", "")
        relpath = sample.get("relpath", "")

        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)
        if not torch.is_tensor(cap_feats):
            cap_feats = torch.tensor(cap_feats)

        return {
            "idx": idx,
            "latents": latents,        # [16, 64, 64]
            "cap_feats": cap_feats,    # [seq_len, hidden_dim], variable length
            "text": text,
            "relpath": relpath,
            "data_type": "image",
        }


def collate_precomputed(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch passed to collate_precomputed")

    latents = torch.stack([x["latents"] for x in batch], dim=0)  # [B, 16, 64, 64]

    cap_feats = [x["cap_feats"] for x in batch]   # keep as list, variable seq len
    texts = [x["text"] for x in batch]
    relpaths = [x["relpath"] for x in batch]
    idxs = [x["idx"] for x in batch]
    data_types = [x["data_type"] for x in batch]

    return {
        "idx": idxs,
        "latents": latents,
        "cap_feats": cap_feats,
        "text": texts,
        "relpath": relpaths,
        "data_type": data_types,
    }


if __name__ == "__main__":
    shard_path = "debug/precomputed_pt/shard_000.pt"

    ds = ShardedLatentsDataset(shard_path, load_into_memory=True)
    print("len(ds):", len(ds))

    sample = ds[0]
    print("sample keys:", sample.keys())
    print("sample latents:", sample["latents"].shape)
    print("sample cap_feats:", sample["cap_feats"].shape)
    print("sample text:", sample["text"])

    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_precomputed)
    batch = next(iter(dl))

    print("batch keys:", batch.keys())
    print("batch latents:", batch["latents"].shape)
    print("batch cap_feats type:", type(batch["cap_feats"]))
    print("batch cap_feats[0]:", batch["cap_feats"][0].shape)
    print("batch texts[0]:", batch["text"][0])