## Image-to-SMILES neural network code

### Files/Dirs:
- `model/`:       code for the neural network model  
- `configs/`:     configuration json files for running the model  
- `data_small/`:  small dataset for checking environment and testing code  
- `train.py`:     code for training the neural network  
- `evaluate_txt`: code to evaluate model's performance  
- `makefile`:     run commands to train and evaluate model
- `predict.py`:   interactive shell to extract SMILES from an image using trained models

### Train and evaluate im2smiles network

To train network run:
  `$ make train-<run-name>`

To evaluate netowork run:
  `$ make eval-<run-name>`

### Reference
The Im2Smiles code is modified from the [Im2Latex network](https://github.com/guillaumegenthial/im2latex) by Guillaume Genthial. 
