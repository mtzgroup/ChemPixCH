# Hand Drawn Hydrocarbon Recognition

This repo contains the code and data needed to generate all the data for the paper

  "Recognizing Hand-drawn Hydrocarbon Structures with Neural Networks:
  A Practical Case Study of Deep Learning and Synthetic Data Generation in a Chemistry Context"
  Weir, H., Thompson, K., Woodward, A., Choi, B., Braun, A., Martinez, T.J. 

The Im2Smiles neural network and code to build the datasets for training are included.

The Im2Smiles code is modified from the [Im2Latex network](https://github.com/guillaumegenthial/im2latex) by Guillaume Genthial. 


## Instalation

1. Create a Python 2.7 conda environment  
  `$ conda create -n im2smiles python=2.7`  

2. Activate environment  
  `$ source activate im2smiles`  
  or  
  `$ conda activate im2smiles`  

3. Clone **hand-drawn-hydrocarbon-recognition repo**  
  `$ git clone https://github.com/hayleyweir/hand-drawn-hydrocarbon-recognition.git`  

4. Install requirements  
  `$ make install`  
 
5. Check environment is installed correctly:  
  `$ cd im2lsmiles`  
  `$ make train-small`  


## Building Datasets  

The datasets can be generated in the `data` directory.   

The datasets can be built by running  
  `$ python build.py`  
in the following directories.  

**Figure 2**:  
  Training, Validation and Test set: `data/clean-RDKit/`  

**Figure 7a**:  
  Training, Validation and Test set: `data/synthetic/pipeline_stages/`  

**Figure 7b**:  
  Training, Validation and Test set: `data/synthetic/size_tests/`  
 
**Figure 8a**:  
  Training set: `data/synthetic/size_tests/`  
  Validation set: `data/hand-drawn/hand-drawn-val/`  
  
**Figure 8b**:  
  Training set: `data/hand-drawn/hand-drawn-training/training-sets/`  
  Validation set: `data/hand-drawn/hand-drawn-training/validation-sets/`  

**Figure 8c**:  
  Pre-Training:  
    Training set: `data/synthetic/size_tests/500K/`  
    Validation Sets: `data/hand-drawn/hand-drawn-training/validation-sets/` and `data/synthetic/size_tests/500K/`  
  
  Fine-tuning:  
    Training set:`data/hand-drawn/hand-drawn-training/training-sets/`  
    Validation set: `data/hand-drawn/hand-drawn-training/validation-sets/`  


Hand-drawn test set: `data/hand-drawn/test-set/`  

## Training im2smiles network  

### Getting started  

Move to the im2smiles directory:
  `$ cd im2smiles`

If you haven't already, check the environment is installed correctly by training on the small dataset:  
  `$ make train-small`  

### Training on synthetic and hand-drawn data  

**Figure 2**:  
  `$ make train-clean-rdkit-10K`  
  `$ make train-clean-rdkit-50K`  
  `$ make train-clean-rdkit-100K`  
  `$ make train-clean-rdkit-200K`  
  `$ make train-clean-rdkit-500K`  

**Figure 7a**:  
  `$ make train-SD-stage-rdkitp`  
  `$ make train-rd-stage-rdkitp-aug`  
  `$ make train-rd-stage-rdkitp-aug-bkg`  
  `$ make train-rd-stage-rdkitp-aug-bkg-deg`  

**Figure 7b**:  
  `$ make train-SD-sizes-50K`  
  `$ make train-SD-sizes-100K`  
  `$ make train-SD-sizes-200K`  
  `$ make train-SD-sizes-500K`  

**Figure 8a**:  
  `$ make train-HDval-50K`  
  `$ make train-HDval-100K`  
  `$ make train-HDval-200K`  
  `$ make train-HDval-500K`  

**Figure 8b**:  
  `$ make train-HDtrain-0_100`  
  `$ make train-HDtrain-10_90`  
  `$ make train-HDtrain-50_50`   
  `$ make train-HDtrain-90_10`  
  `$ make train-HDtrain-100_0`  

**Figure 8c**:  
  Restart weights of `HDval-500K` and `SD-500K` training, followed by  
  `$ make train-HDtrain-90_10`  

## Maintainers  
[Hayley Weir](mailto:hweir@stanford.edu)
