# Thesis-Code
Storage for the code for my EE Thesis
# Datasets
Dataset shards are under a datasets directory: 
  Shards are already resized to 128 resolution in order for memory occupation, download and training overhead considerations.

  
  <img src="/assets/wakevision_random.png" width="300" alt="Diagram">


  
  Shards have been generated 
  - dataset_generator.py to generate Wake Vision Dataset like the user wants
  - User can modify dataset_generator.py to change resolution, image size, split, shard size.
    - can be improved by adding argparse arguments to script
  
# Models
Pickled models are uploaded under a models directory:
  Pickled models can be used as training checkpoints.

# External
Imported PULP-Trainlib main branch

Future Fork of PULP-Trainlib with model deployers
# Training
This section will detail the training process:
- Where are the models imported from
- Transfer Learning
- On-line learning
- Learning rate exploration and scheduling
- Weight Decay
- Accuracy & Loss charts
- 
  
# TO DO
- Give dependency list so that users can install them, include commands snippets
- add results of experiments , plots, logs
- add pretrainer.py
- add on-line_trainer.py
- export .onnx files
  
