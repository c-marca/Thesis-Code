''' 
    Author: Carlo Marcantonio
    mail: carlo.marcantonio3@studio.unibo.it
	  
'''
import os     # To create directories 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time                                                 
import argparse                                                                               
from tqdm.auto import tqdm                                  # for progress bar and CLI  messages for dataset generation
import csv                                                    # for training logs
import models 
import loaders


# Create parser

parser = argparse.ArgumentParser()
parser.add_argument("--skip-lr-tuner", action="store_true",
               help="Skip the learning rate optimization step")   # Inactive for now
parser.add_argument("--no-show-plots", action="store_true",
               help="Don't show plots")
parser.add_argument("--no-save-images", action="store_true",
               help="Don't save images")
parser.add_argument(
     "--load-model",
    type=str,                    
    metavar="PATH",
    help="Input the path of the model you want to load"
)
parser.add_argument(
     "--dir-name",
    type=str,                    
    metavar="PATH",
    default = "Results",
    help="Input the name of the project directory"
)

parser.add_argument(

    "--no-ckpts",
    action="store_true",
    help="Only saves initial and final checkpoints, no per-epoch checkpoints "

)

args = parser.parse_args()

load_path = args.load_model      # e.g., "models/m.pt" or None
dir_name = args.dir_name      # e.g., "out/m.pt" or None

# Insert results folder here AND dataset folders

project_name = './training/Results'
project_dir = './' + project_name
plots_dir = project_dir + '/plots/'
models_dir = project_name + '/checkpoints/'

os.makedirs(project_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Choose device
if torch.cuda.is_available():
    device = torch.device('cuda')
    pin_memory = True
    
else:
    device = torch.device('cpu')
    pin_memory = False


# Wake Vision

FG_KEYS = [
  "predominantly_female","predominantly_male","young","middle_age","older","gender_unknown","age_unknown",
  "near","medium_distance","far",
  "bright","dark","normal_lighting",
  "depiction","person_depiction","non-person_depiction","non-person_non-depiction",
  "body_part"
]
COL2IDX = {k: i for i, k in enumerate(FG_KEYS)}

class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self._wrote_header = os.path.getsize(path) > 0
        if not self._wrote_header:
            self.writer.writeheader()
            self.f.flush()

    def log(self, **row):
        self.writer.writerow(row)
        self.f.flush()  # durable enough for most runs

    def close(self):
        self.f.close()


def test(net,test_loader):
    net.to(device)
    # Set gradients to zero
    net.zero_grad()
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    print(f"Testing on {test_loader.name}")
    with torch.inference_mode():
        correct = total = 0
        loss_sum = 0.0

        try:
            total_batches = len(test_loader)
            if total_batches == 0:
                raise ValueError("len(test_loader)==0")
        except Exception:
            try:
                from math import ceil
                total_batches = ceil(len(test_loader.dataset)/test_loader.batch_size)
            except Exception:
                total_batches = None

        pbar = tqdm(total=total_batches, desc="Test", dynamic_ncols=True)

        for imgs, labels, *_ in test_loader:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            out = net(imgs)
            loss = loss_fn(out, labels).item()

            bs = labels.size(0)
            loss_sum += loss * bs
            total += bs
            correct += (out.argmax(1) == labels).sum().item()

            pbar.update(1)
            pbar.set_postfix(loss=loss_sum/total, acc=100.0*correct/total)

        pbar.close()

    avg_loss = loss_sum / max(1, total)
    acc = 100.0 * correct / max(1, total)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {acc:.2f}%")

    return avg_loss,acc
    
def validate(net, val_loader ):
    net.to(device)
    # Set gradients to zero
    net.zero_grad()
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    print(f"Validating on {val_loader.name}")
    with torch.inference_mode():
        correct = total = 0
        loss_sum = 0.0

        try:
            total_batches = len(val_loader)
            if total_batches == 0:
                raise ValueError("len(val_loader)==0")
        except Exception:
            try:
                from math import ceil
                total_batches = ceil(len(val_loader.dataset)/val_loader.batch_size)
            except Exception:
                total_batches = None

        pbar = tqdm(total=total_batches, desc="Val", dynamic_ncols=True)

        for imgs, labels, *_ in val_loader:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            out = net(imgs)
            loss = loss_fn(out, labels).item()

            bs = labels.size(0)
            loss_sum += loss * bs
            total += bs
            correct += (out.argmax(1) == labels).sum().item()

            pbar.update(1)
            pbar.set_postfix(loss=loss_sum/total, acc=100.0*correct/total)

        pbar.close()
    avg_loss = loss_sum / max(1, total)
    acc = 100.0 * correct / max(1, total)
    print(f"Val Loss: {avg_loss:.4f}")
    print(f"Val Accuracy: {acc:.2f}%")

    return avg_loss,acc

# pass optim to train function so one can try different ones
# so far    SGD
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)



def train(net, train_loader, epochs, learning_rate, momentum, weight_decay):



    print("Initial Validation")
    val_loss,val_acc = validate(net, loaders.WV_val_ld)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = torch.as_tensor(IMNET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMNET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    # FP32 speed knobs
    torch.backends.cuda.matmul.fp32_precision = "high"      
    torch.backends.cudnn.conv.fp32_precision = "tf32"       
    torch.backends.cudnn.benchmark = True


    net.to(device)
    net.train()

    print("Saving initial checkpoint")
    torch.save(net.state_dict(), f"{models_dir}{net._get_name()}_untrained.pth")

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    # build optimizer from the actual trainable params
    pg = [p for p in net.parameters() if p.requires_grad]
    
    optimizer = optim.SGD(
        pg, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    grad_accum = 1  
    log_every_steps = 100 * max(1, grad_accum)
    bs = getattr(loaders.sampler, "batch_size", None)

    print(f"Training {net.name} on {train_loader.name} dataset, for Epochs: {epochs}, with  bs={bs}, eff_bs={None if bs is None else bs*grad_accum}, "
          f"imgs={None if bs is None else bs*len(train_loader)}, res = 128x128x3 lr={learning_rate}, mom={momentum}, wd={weight_decay}")
    
    val_loss_list = []
    val_acc_list = []

    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    fields = [
        "split", "epoch", "step", "seen",
        "lr","train_loss", "val_loss", "val_acc"
    ]

    logger = CSVLogger(path = project_dir + "/logs", fieldnames = fields)
    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        t_last = time.perf_counter()
        step = -1
        for step, (imgs, labels, *_) in enumerate(train_loader):

            if imgs.dtype == torch.uint8:
                imgs = imgs.float().mul_(1.0/255.0)

            imgs   = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True).long()

            # FP32 normalization and forward 
            x = (imgs - mean) / std

            out = net(x)
            loss = loss_fn(out, labels) #/ grad_accum                        # logits expected; no softmax
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % log_every_steps == 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                delta = time.perf_counter() - t_last



                if bs is not None:
                    ips = (bs * log_every_steps) / max(delta, 1e-9)
                    print(f"epoch {epoch} step {step+1} image = {bs*step} "
                          f"loss={float(loss.detach()*grad_accum):.4f} imgs/s={ips:.0f}")
                    logger.log(
                                    split="train",
                                    epoch=epoch,
                                    step=step,
                                    seen= bs*step,
                                    lr=learning_rate,
                                    train_loss=float(loss.item()),
                                    val_loss = val_loss,
                                    val_acc=val_acc,
                            )
                else:
                    print(f"epoch {epoch} step {step+1} loss={float(loss.detach()*grad_accum):.4f}")
                t_last = time.perf_counter()

        # flush residual accumulation once per epoch (only if grad_accum > 1)
        if step >= 0 and (step + 1) % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(f"Saving checkpoint for epoch {epoch}")
        torch.save(net.state_dict(), f"{models_dir}{net._get_name()}_epoch_{epoch}.pth")
        val_loss,val_acc = validate(net, loaders.WV_val_ld)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    end_time = time.time()
    print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
    torch.save(net.state_dict(), f"{models_dir}{net._get_name()}_final.pth")
    print("Saving plots")
    if val_acc_list:                       # not empty
        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.tight_layout()
        plt.savefig(plots_dir+"val_accuracy.png")         # prefer savefig in headless runs
        plt.close()
    else:
        print("val_acc_list is empty; nothing to plot.")
    
    if val_loss_list:                       # not empty
        plt.figure()
        plt.plot(range(len(val_loss_list)), val_loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.tight_layout()
        plt.savefig(plots_dir+"val_loss.png")         # prefer savefig in headless runs
        plt.close()
    else:
        print("val_acc_list is empty; nothing to plot.")

    # Add .csv log


def LoadModel(net ,model_path):

    ckpt = torch.load(model_path, map_location="cpu")
    net.load_state_dict(ckpt)

    return net

'''
def hyperparameter_explorator(net,epochs,lr_list):
    for lr in lr_list: ...
        train(net=models.MobileNetFT,train_loader=loaders.WV_train_ld,epochs=1,learning_rate=lr,momentum = 0.9, weight_decay = 1e-4 )
'''


print("Initial Test")
test(net = models.MobileNetFT, test_loader = loaders.WV_test_ld)

train(net=models.MobileNetFT,train_loader=loaders.WV_train_ld,epochs=1,learning_rate=0.01,momentum = 0.9, weight_decay = 1e-4 )

print("Final Validation")
validate(net = models.MobileNetFT, val_loader = loaders.WV_val_ld)
print("Final Test")
test(net = models.MobileNetFT, test_loader = loaders.WV_test_ld)

