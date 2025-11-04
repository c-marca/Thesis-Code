import os     # To create directories 
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import glob                                                 # filename pattern-matching to select shards
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List
import bisect
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _worker_init(_):
    import torch, os
    torch.set_num_threads(1)

if torch.cuda.is_available():
    device = torch.device('cuda')
    pin_memory = True
    
else:
    device = torch.device('cpu')
    pin_memory = False

class FastWVCacheDataset(Dataset):
    def __init__(self, cache_dir, cond=None, mode="AND", imagenet_norm=True, max_shards_in_mem=1):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "shard-*.pt")))
        assert self.files, f"no shards in {cache_dir}"

        self.imagenet_norm = imagenet_norm
        self._mean = torch.as_tensor(IMNET_MEAN, dtype=torch.float32).view(-1,1,1)
        self._std  = torch.as_tensor(IMNET_STD,  dtype=torch.float32).view(-1,1,1)

        # small shard cache (per worker process)
        self.max_shards_in_mem = int(max(1, max_shards_in_mem))
        self._cache = OrderedDict()  # fi -> pack

        # load once to probe metadata
        probe = torch.load(self.files[0], map_location="cpu")
        self.has_fg = ("fg" in probe) and ("fg_keys" in probe)
        self.fg_keys = list(map(str, probe.get("fg_keys", []))) if self.has_fg else []
        self.k2i = {k:i for i,k in enumerate(self.fg_keys)}

        # precompute pos/neg indices once
        pos_idx = [self.k2i[k] for k, v in (cond or {}).items() if v == 1] if self.has_fg else []
        neg_idx = [self.k2i[k] for k, v in (cond or {}).items() if v == 0] if self.has_fg else []

        # build global index with a single read per shard
        self.idx = []
        for fi, path in enumerate(self.files):
            pack = torch.load(path, map_location="cpu")
            N = pack["y"].shape[0]
            if not cond or not self.has_fg:
                self.idx += [(fi, j) for j in range(N)]
            else:
                fg = pack["fg"].to(torch.uint8)  # [N,K]
                ok = torch.ones(N, dtype=torch.bool)
                if pos_idx:
                    sel = (fg[:, pos_idx] == 1)
                    ok &= sel.all(1) if mode == "AND" else sel.any(1)
                if neg_idx:
                    ok &= (fg[:, neg_idx] == 0).all(1)
                js = torch.nonzero(ok, as_tuple=True)[0].tolist()
                self.idx += [(fi, int(j)) for j in js]
            # keep memory low during init
            del pack

    def __len__(self):
        return len(self.idx)

    def _get_shard(self, fi):
        # LRU fetch
        hit = self._cache.get(fi, None)
        if hit is not None:
            self._cache.move_to_end(fi)
            return hit
        pack = torch.load(self.files[fi], map_location="cpu")
        self._cache[fi] = pack
        if len(self._cache) > self.max_shards_in_mem:
            self._cache.popitem(last=False)  # evict LRU
        return pack

    def __getitem__(self, i):
        fi, j = self.idx[i]
        pack = self._get_shard(fi)

        x = pack["x"][j]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        # fast path if you stored CHW float32 in [0,1]
        if x.dtype != torch.float32:
            x = x.float()
        # normalize from 0..255 only if needed
        if x.max() > 1.0:
            x = x / 255.0

        # ensure CHW
        if x.ndim == 3 and x.shape[-1] in (1,3):
            x = x.permute(2,0,1).contiguous()
        if x.ndim != 3 or x.shape[0] not in (1,3):
            raise RuntimeError(f"unexpected image shape: {tuple(x.shape)}; expected CHW with C in (1,3)")

        if self.imagenet_norm:
            C = x.shape[0]
            mean = self._mean[:C]
            std  = self._std[:C]
            x = (x - mean) / std

        y = torch.as_tensor(pack["y"][j], dtype=torch.long)

        if self.has_fg:
            fg = pack["fg"][j].to(torch.uint8)
            fn_list = pack.get("fn", None)
            fn = fn_list[j] if isinstance(fn_list, list) else ""
        else:
            fg = torch.zeros(0, dtype=torch.uint8)
            fn = ""

        return x, y, fg, fn

def collate_xy(batch):
    xs, ys, fgs, fns = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), torch.stack(fgs, 0), list(fns)


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

# Usage
WakeVision_train = FastWVCacheDataset("./datasets/wv_train")
WakeVision_val   = FastWVCacheDataset("./datasets/wv_val")
WakeVision_test  = FastWVCacheDataset("./datasets/wv_test")

# LOADERS

WV_train_ld = DataLoader(
    WakeVision_train,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    prefetch_factor=1,
    pin_memory=True,                 # set False if RAM tight
    persistent_workers=True,         # enable after stable
    worker_init_fn=_worker_init,
    collate_fn=collate_xy,
    drop_last=True
)
WV_train_ld.name = "WakeVision_train_quality"


WV_test_ld = DataLoader(WakeVision_test, batch_size=32, shuffle=False,
                    num_workers=4, pin_memory=pin_memory,
                    persistent_workers=True, prefetch_factor=1,worker_init_fn=_worker_init,
                         collate_fn=collate_xy,drop_last=False)

WV_test_ld.name = "WakeVision_test"

WV_val_ld = DataLoader(WakeVision_val, batch_size=32, shuffle=False,
                    num_workers=8, pin_memory=pin_memory,
                    persistent_workers=True, prefetch_factor=2,worker_init_fn=_worker_init,
                         collate_fn=collate_xy,drop_last=False)

WV_val_ld.name = "WakeVision_val"


# These should be here
# These will be used in the future
def scan_labels(cache_dir):
    files = sorted(glob.glob(os.path.join(cache_dir, "shard-*.pt")))
    assert files, f"no shards in {cache_dir}"
    labels, ds_idx = [], []   # ds_idx holds flat dataset indices [0..N-1]
    base = 0
    for f in files:
        y = torch.load(f, map_location="cpu")["y"].view(-1).to(torch.long)
        n = y.numel()
        labels.append(y)
        ds_idx.extend(range(base, base+n))
        base += n
    labels = torch.cat(labels, dim=0)
    return ds_idx, labels  # len(ds_idx) == len(labels) == N

def make_balanced_selection(labels, per_class_max=None, rng=None):
    if rng is None:
        rng = torch.Generator().manual_seed(0)  # set for deterministic runs
    K = int(labels.max().item()) + 1
    buckets = [(labels == c).nonzero(as_tuple=True)[0] for c in range(K)]
    counts = torch.tensor([b.numel() for b in buckets])
    m = counts.min().item()
    if per_class_max is not None:
        m = min(m, int(per_class_max))
    picks = []
    for b in buckets:
        perm = b[torch.randperm(b.numel(), generator=rng)]
        picks.append(perm[:m])
    sel = torch.cat(picks)
    sel = sel[torch.randperm(sel.numel(), generator=rng)]
    return sel.tolist(), counts.tolist(), m

def count_from_shards(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "shard-*.pt")))
    if not files:
        raise FileNotFoundError(f"No shards in {folder}")

    fg_keys = None
    total = 0
    y0 = 0
    y1 = 0
    fg_sum = None

    for f in tqdm(files, desc="Counting shards", unit="file"):
        blob = torch.load(f, map_location="cpu")
        y = blob["y"].to(torch.long)              # [N]
        fg = blob["fg"].to(torch.long)            # [N, K] 0/1
        if fg_keys is None:
            fg_keys = blob.get("fg_keys", [f"k{i}" for i in range(fg.shape[1])])
            fg_sum = torch.zeros(len(fg_keys), dtype=torch.long)

        n = y.numel()
        y1 += int(y.sum().item())
        y0 += int(n - y.sum().item())
        fg_sum += fg.sum(dim=0)
        total += n

    print(f"total: {total}")
    print(f"person label counts: {{0: {y0}, 1: {y1}}}")
    print("FG positive counts:")
    for i, k in enumerate(fg_keys):
        print(f"  {k}: {int(fg_sum[i].item())}")


#count_from_shards("./datasets/wv_test_cache_128")
