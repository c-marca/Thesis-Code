import os     # To create directories
import torch
from datasets import load_dataset                           # To get Wake Vision Dataset from Hugging Face Datasets
from tqdm.auto import tqdm                                  # for progress bar and CLI  messages for dataset generation
from torchvision import transforms                          # To resize images, can use it to make augmentations

# CAN ADD ARGUMENTS:
# --resolution
# --path
# --num_imgs
# --shard_size

# CAN ADD BALANCED DATASETS IN HERE DIRECTLY

FG_KEYS = [
  "predominantly_female","predominantly_male","young","middle_age","older","gender_unknown","age_unknown",
  "near","medium_distance","far",
  "bright","dark","normal_lighting",
  "depiction","person_depiction","non-person_depiction","non-person_non-depiction",
  "body_part"
]
COL2IDX = {k: i for i, k in enumerate(FG_KEYS)}

tfm = transforms.Compose([transforms.Resize((128,128)),
                          transforms.ToTensor(),
                          transforms.ConvertImageDtype(torch.uint8)])

ds = load_dataset("Harvard-Edge/Wake-Vision", split="test", streaming=True)

N = 20_000

SHARD = 4096

pbar = tqdm(total=N, desc="Transforming WV ", unit="img")
buf_x, buf_y, buf_fg, buf_fn = [], [], [], []
os.makedirs("wv_test_cache_128", exist_ok=True)
shard_id = 0

for i, ex in enumerate(ds):
    if i >= N: break
    x = tfm(ex["image"])
    y = torch.tensor(ex["person"], dtype=torch.uint8)
    fg = torch.tensor([ex[k] for k in FG_KEYS], dtype=torch.uint8)
    fn = ex["filename"]

    buf_x.append(x); buf_y.append(y); buf_fg.append(fg); buf_fn.append(fn)

    if len(buf_x) == SHARD:
        torch.save(
            {"x": torch.stack(buf_x),
             "y": torch.stack(buf_y),
             "fg": torch.stack(buf_fg),           # shape [SHARD, len(FG_KEYS)]
             "fn": buf_fn,                        # list[str]
             "fg_keys": FG_KEYS},                 # for safety
            f"wv_test_cache_128/shard-{shard_id:05d}.pt"
        )
        shard_id += 1
        buf_x.clear(); buf_y.clear(); buf_fg.clear(); buf_fn.clear()
    if i % 200 == 0:
        pbar.set_postfix(shard=shard_id)
    pbar.update(1)

pbar.close()

# flush tail
if buf_x:
    torch.save(
        {"x": torch.stack(buf_x),
         "y": torch.stack(buf_y),
         "fg": torch.stack(buf_fg),
         "fn": buf_fn,
         "fg_keys": FG_KEYS},
        f"wv_test_cache_128/shard-{shard_id:05d}.pt"
    )