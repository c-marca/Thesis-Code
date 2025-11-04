# Thesis-Code
Storage for the code for my EE Thesis.

# Extra requirements ([PULP-TrainLib](https://github.com/pulp/-platform/pulp-trainlib) setup needed)

  ```bash
sudo apt install git-lfs
python -m pip install pytorchcv
python -m pip install onnx
python -m pip install tqdm
python -m pip install datasets

```

# Datasets
-Wake Vision
  - 100k imgs train quality split (4.58 GB)
  - 10k imgs val split  (469 MB)
TOTAL: around 6 GB


Full Wake Vision Splits:

- c.a 18.5k val images: functional
- c.a 55.7k test images: functional
- c.a 1.2M train images: TBD 

Dataset shards are under a datasets directory:

-Shards are already resized to 128x128 resolution  for memory occupation, download and training overhead considerations.

  
  <img src="/assets/wakevision_random.png" width="300" alt="Diagram">

TO PULL SHARDS:
```bash
git lfs pull
  ```
  Shards have been generated 
  - dataset_generator.py to generate Wake Vision Dataset like the user wants
  - User can modify dataset_generator.py to change resolution, image size, split, shard size.
    - can be improved by adding argparse arguments to script

 Wake Vision has finegrained tags for different contexts:
 
  ```python
tfm = transforms.Compose([transforms.Resize((128,128)),
                          transforms.ToTensor(),
                          transforms.ConvertImageDtype(torch.uint8)])

ds = load_dataset("Harvard-Edge/Wake-Vision", split="test", streaming=True)

N = 20_000

SHARD = 4096
```
Dataset composition by class and finegrained can be seen in respective (split)_balance.txt

# Models
 The models.py module makes them available.

Pickled models are uploaded under a models directory:

  Pickled models can be used as training checkpoints.
  
AVAILABLE MODELS:
- MobileNetLP (setup for Linear Probe Transfer Learning)

- MobileNetFD (setup for Linear Probe Transfer Learning)
- MobileNet_no_pre (no pretraining) for MobileNetV1
- MobileNetFT (setup for Partial Finetuning i.e K DWS layers are unfrozen, K=2 , can can be explored).
TO DO:
- MobileNet_unfr (pretrained and unfrozen)
To show effectiveness of Transfer Learning  

- MobileNet is imported from pytorchcv
- MobileNetFD is imported from pytorchcv 
PRETRAINING LOGS ARE PROVIDED

# Finetuner

The finetuner script provides validation, test and training functions.
- It provides CLI logging of hyperparameters, these will be soon turned to .csv logs as well.
- It provides plotting of validation per epoch.

# Loaders

The loaders module provides dataloaders for each dataset split.

These loaders are tuned in such a way as to have high performance on a personal  GPU RTX2080.


# External
Imported [PULP-TrainLib](https://github.com/pulp/-platform/pulp-trainlib) main branch
as a subtree.

Inside there are Custom Deployers for MobileNet and MobileNetFD.

 # Assets
 Images and .onnx and similar
 # Logs
 Memory occupation logs,training logs
* Torchsummary does not take into account the fact that one can discard the activations of the layers before the last updatable layer.
  * see torchsummary log     
* PULP-TrainLib's Deployer does, so we take this as the memory occupation figure.
  * see respective log
-Pretraining logs
 https://pypi.org/project/pytorchcv/
Future Fork of PULP-Trainlib with model deployers
# Training
This section will detail the training process:
- Transfer Learning (FineTuning)
- On-line learning
- Learning rate exploration and scheduling
- Weight Decay
- Accuracy & Loss charts

Table with model accuracy and memory occupation, comparison with the state of the art:
| Models     | Memory     | Test Accuracy |
|------------|------------|---------------|
| MobileNet   | 2 MB      | x             |          |
|MobileNetFD  | 2.1 MB    | x             |

Memory figures are taken from PULP-Trainlib's Deployer.

# Papers
- MobileNet: https://arxiv.org/abs/1704.04861
- MobileNetFD: https://arxiv.org/abs/1802.03750

# TO DO
- Give full dependency list so that users can install them, include commands snippets
- add results of experiments , plots, logs
- add Thesis stub
- use TrainLib tests to profile performance of NNs
- Add hyperparam.csv and train_log.csv
