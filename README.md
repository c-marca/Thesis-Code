# Thesis-Code
Storage for the code for my EE Thesis.

# Extra requirements ([PULP-TrainLib](https://github.com/pulp/-platform/pulp-trainlib) setup needed)

  ```bash
sudo apt install git-lfs
python -m pip install timm
python -m pip install pytorchcv
python -m pip install onnx
```

# Datasets
-Wake Vision
  - 100k imgs train quality split
  - 20k imgs test split
  - 10k imgs val split
    
Dataset shards are under a datasets directory: 
  Shards are already resized to 128 resolution in order for memory occupation, download and training overhead considerations.

  
  <img src="/assets/wakevision_random.png" width="300" alt="Diagram">


  
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
A
Pickled models are uploaded under a models directory:
  Pickled models can be used as training checkpoints.
AVAILABLE MODELS:
- MobileNetLP (setup for Linear Probe Transfer Learning)
- MobileNet_no_pre (no pretraining)
- MobileNetFD (setup for Linear Probe Transfer Learning)

- MobileNet is imported from pytorchcv
- MobileNetFD is imported from pytorchcv 
- SqueezeNet is imported from timm

# External
Imported [PULP-TrainLib](https://github.com/pulp/-platform/pulp-trainlib) main branch
as a subtree.

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
- Where are the models imported from
- Transfer Learning
- On-line learning
- Learning rate exploration and scheduling
- Weight Decay
- Accuracy & Loss charts

Table with model accuracy and memory occupation, comparison with the state of the art:
| Models     | Memory     | Accuracy |
|------------|------------|----------|
| MobileNet   | 2 MB      | x        |          |
|MobileNetFD  | 2.1 MB    | x        |


# Papers
- MobileNet
- MobileNetFD
- SqueezeNet


  
# TO DO
- Give dependency list so that users can install them, include commands snippets
- add results of experiments , plots, logs
- add finetuner.py
- add on-line_trainer.py
- export .onnx files
- add models.py
  
