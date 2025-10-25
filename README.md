# Thesis-Code
Storage for the code for my EE Thesis.

# Extra requirements ([PULP-TrainLib](https://github.com/pulp/-platform/pulp-trainlib) setup needed)

  ```bash
pip install timm
```

# Datasets
-Wake Vision

Dataset shards are under a datasets directory: 
  Shards are already resized to 128 resolution in order for memory occupation, download and training overhead considerations.

  
  <img src="/assets/wakevision_random.png" width="300" alt="Diagram">


  
  Shards have been generated 
  - dataset_generator.py to generate Wake Vision Dataset like the user wants
  - User can modify dataset_generator.py to change resolution, image size, split, shard size.
    - can be improved by adding argparse arguments to script
  ```python
tfm = transforms.Compose([transforms.Resize((128,128)),
                          transforms.ToTensor(),
                          transforms.ConvertImageDtype(torch.uint8)])

ds = load_dataset("Harvard-Edge/Wake-Vision", split="test", streaming=True)

N = 20_000

SHARD = 4096
```
# Models
Pickled models are uploaded under a models directory:
  Pickled models can be used as training checkpoints.

# External
Imported [PULP-TrainLib](https://github.com/pulp/-platform/pulp-trainlib) main branch
as a subtree.

Future Fork of PULP-Trainlib with model deployers
# Training
This section will detail the training process:
- Where are the models imported from
- Transfer Learning
- On-line learning
- Learning rate exploration and scheduling
- Weight Decay
- Accuracy & Loss charts

MobileNetV1 is imported from pytorchcv

  
# TO DO
- Give dependency list so that users can install them, include commands snippets
- add results of experiments , plots, logs
- add pretrainer.py
- add on-line_trainer.py
- export .onnx files
  
