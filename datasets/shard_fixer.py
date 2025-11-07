import os, glob, tempfile, shutil, torch

def _mask_like(x, keep):
    # Works for tensors or lists
    if isinstance(x, torch.Tensor):
        return x[keep]
    if isinstance(x, list):
        idx = keep.nonzero(as_tuple=True)[0].tolist()
        return [x[i] for i in idx]
    return x  # leave non-indexables as-is

def clean_dir(cache_dir, label_ok=(0,1), dry_run=False):
    paths = sorted(glob.glob(os.path.join(cache_dir, "shard-*.pt")))
    assert paths, f"no shards in {cache_dir}"
    kept_total = drop_total = 0
    for p in paths:
        pk = torch.load(p, map_location="cpu")
        y = torch.as_tensor(pk["y"], dtype=torch.long)
        keep = torch.zeros_like(y, dtype=torch.bool)
        for v in label_ok:
            keep |= (y == v)
        n0 = y.numel()
        n_keep = int(keep.sum())
        n_drop = n0 - n_keep
        kept_total += n_keep
        drop_total += n_drop
        if n_drop == 0:
            print(f"[ok] {os.path.basename(p)} kept {n_keep}/{n0}")
            continue

        print(f"[fix] {os.path.basename(p)} drop {n_drop}/{n0} rows")

        if dry_run:
            continue

        # apply mask to known array fields
        for k in list(pk.keys()):
            v = pk[k]
            try:
                pk[k] = _mask_like(v, keep)
            except Exception:
                # leave metadata (e.g., 'fg_keys') untouched
                pass

        # enforce dtype and shape for labels after masking
        pk["y"] = torch.as_tensor(pk["y"], dtype=torch.long)

        # atomic write
        ddir = os.path.dirname(p)
        fd, tmp = tempfile.mkstemp(prefix="clean_", suffix=".pt", dir=ddir)
        os.close(fd)
        torch.save(pk, tmp)
        shutil.move(tmp, p)

    print(f"done: kept={kept_total} dropped={drop_total}")

# Run on all splits
for d in ["./wv_val"]:
    clean_dir(d, label_ok=(0,1), dry_run=False)
