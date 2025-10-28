import os     # To create directories 
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from datasets import load_dataset                           # To get Wake Vision Dataset from Hugging Face Datasets
import glob                                                 # filename pattern-matching to select shards


if torch.cuda.is_available():
    device = torch.device('cuda')
    pin_memory = True
    
else:
    device = torch.device('cpu')
    pin_memory = False


class WVCacheDataset(Dataset):
    def __init__(self, cache_dir, cond=None, mode="AND"):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "shard-*.pt")))
        assert self.files, f"no shards in {cache_dir}"

        # probe first shard
        probe = torch.load(self.files[0], map_location="cpu")
        self.has_fg = ("fg" in probe) and ("fg_keys" in probe)
        self.fg_keys = list(map(str, probe["fg_keys"])) if self.has_fg else []
        self.k2i = {k:i for i,k in enumerate(self.fg_keys)}

        # build index; apply filter only if FG exists
        self.idx = []
        for fi,f in enumerate(self.files):
            n = torch.load(f, map_location="cpu")["y"].shape[0]
            if not cond or not self.has_fg:
                self.idx += [(fi, j) for j in range(n)]
            else:
                pack = torch.load(f, map_location="cpu")
                fg = pack["fg"].to(torch.uint8)      # [N,K]
                pos = [self.k2i[k] for k,v in cond.items() if v==1]
                neg = [self.k2i[k] for k,v in cond.items() if v==0]
                ok = torch.ones(fg.shape[0], dtype=torch.bool)
                if pos:
                    sel = fg[:, pos] == 1
                    ok &= (sel.all(1) if mode=="AND" else sel.any(1))
                if neg:
                    ok &= (fg[:, neg] == 0).all(1)
                self.idx += [(fi, int(j)) for j in torch.nonzero(ok, as_tuple=True)[0].tolist()]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        fi, j = self.idx[i]
        pack = torch.load(self.files[fi], map_location="cpu")
        x = pack["x"][j].float() / 255.0
        y = torch.as_tensor(pack["y"][j], dtype=torch.long)
        if self.has_fg:
            fg = pack["fg"][j].to(torch.uint8)
            fn = pack.get("fn", "")
        else:
            fg = torch.zeros(0, dtype=torch.uint8)
            fn = ""
        return x, y, fg, fn

# Custom Collate
def collate_xy(batch):
    xs, ys, fgs, fns = zip(*batch)              # each from __getitem__
    return (torch.stack(xs),
            torch.stack(ys),
            torch.stack(fgs),                   # requires equal fg length
            list(fns))

# To balance the dataset
'''
flat_idx, labels = scan_labels(cache_dir)

sel, counts, m = make_balanced_selection(labels, per_class_max=None)
sampler = SubsetRandomSampler(sel)
loader = DataLoader(dataset, batch_size=64, sampler=sampler,
                    num_workers=4, pin_memory=True, drop_last=True)

'''
# DATASETS
WakeVision_train = WVCacheDataset("./datasets/wv_train_cache_128") 
WakeVision_test = WVCacheDataset("./datasets/wv_test_cache_128")
WakeVision_val = WVCacheDataset("./datasets/wv_val_cache_128") 

N = len(WakeVision_val)
minival_size = 2_000
g = torch.Generator().manual_seed(42)
perm = torch.randperm(N, generator=g)
minival_idx = perm[:minival_size].tolist()


WakeVision_mini_val = Subset(WakeVision_val, minival_idx)


# LOADERS
WV_train = DataLoader(WakeVision_train, batch_size=128, shuffle=True,
                   num_workers=12, pin_memory=pin_memory,
                    persistent_workers=True, prefetch_factor=4,
                         collate_fn=collate_xy)

WV_test = DataLoader(WakeVision_test, batch_size=128, shuffle=False,
                    num_workers=12, pin_memory=pin_memory,
                    persistent_workers=True, prefetch_factor=4,
                         collate_fn=collate_xy)

WV_val = DataLoader(WakeVision_val, batch_size=128, shuffle=False,
                    num_workers=12, pin_memory=pin_memory,
                    persistent_workers=True, prefetch_factor=4,
                         collate_fn=collate_xy)
                    
WV_mini_val = DataLoader(WakeVision_mini_val, batch_size=128, shuffle=False,
                    num_workers=12, pin_memory=pin_memory,
                    persistent_workers=True, prefetch_factor=4,
                         collate_fn=collate_xy)
                    

# These should be here

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
