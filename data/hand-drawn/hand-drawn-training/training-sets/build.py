'''
This file generates training set of synthetic:hand-drawn data for
ratios:
 - 0:100
 - 10:90
 - 50:50
 - 90:10
 - 100:0.
Using the 500K synthetic dataset and `hand-drawn-train.txt` in this folder.
The hand-drawn images are augmented and degraded to make up to the number 
needed for the training set. The formulas are written to `train.formulas.txt`, 
indices to `train.index.txt` and images to `train_images` within the ratio 
folder. 
'''
import glob
import random
import cv2
import numpy as np
import os 
from rdkit import Chem

from degrade import degrade_img
from augment import augment_mol, augment_bkg

random.seed(1)

def augment_HD(HD_files, n_HD):
    '''Augments and degrades the dataset of hand-drawn images 
    to produce the number needed in the training set'''
    
    HD_aug_imgs = [] # list of augmented/degraded hand-drawn images
    HD_aug_smiles = [] # list of smiles strings
    n_augs_per_img = int(n_HD / len(HD_files))
    remainder = n_HD % len(HD_files)
    print " - {} augmentations per image".format(n_augs_per_img+1)
    
    for f in HD_files:
        img = cv2.imread(f)
        for n in range(n_augs_per_img):
            aug_img = augment_mol(img)
            aug_deg_img = degrade_img(aug_img)
            HD_aug_imgs.append(aug_deg_img)
            HD_aug_smiles.append(f.split("/")[-1].split(".")[0])

    for i in range(remainder):
        img = cv2.imread(HD_files[i])
        aug_img = augment_mol(img)
        aug_deg_img = degrade_img(aug_img)
        HD_aug_imgs.append(aug_deg_img)
        HD_aug_smiles.append(HD_files[i].split("/")[-1].split(".")[0])

    return HD_aug_imgs, HD_aug_smiles


def get_SD_imgs(SD_files):
    ''' Collect synthetic data images from file paths'''
    
    SD_imgs = []
    for f in SD_files:
        print f
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        SD_imgs.append(img)
    
    return SD_imgs


def write_data(z):
    ''' Write smiles and image data'''
    
    if not os.path.isdir("train_images/"):
        os.mkdir("train_images/")

    f_form = open("train.formulas.txt", "w")
    f_idx = open("train.index.txt", "w")
    for idx, (img, smiles) in enumerate(z):
        cv2.imwrite("train_images/{}.png".format(smiles), img)
        f_form.write("{}\n".format(smiles))
        f_idx.write("{} {}.png\n".format(idx,idx))
    f_form.close()
    f_idx.close()


def make_train_set(train_size, ratio):
    ''' Build training  set of hand-drawn/synthetic data
    according to give ratio by reading in the formulas, collecting
    the images, shuffling the data and writing the files.
    '''
    
    print "\nBuilding training set of size {} with ratio {}:{} for synthetic:HD data".format(train_size, int(ratio[0]*100), int(ratio[1]*100))
   
    # collect hand-drawn data smiles and shuffle
    with open("../hand-drawn-train.txt", "r") as f:
        ls = f.readlines()
    HD_files = [l.replace("\n", "") for l in ls]
    HD_files = ["../../../hand-drawn-full/{}.png".format(l) for l in HD_files]
    random.shuffle(HD_files)
    
    # collect synthetic data smiles and shuffle
    with open("../../../../synthetic/size_tests/500K/train.formulas.txt", "r") as f:
        ls = f.readlines()
    SD_smiles = [l.replace("\n", "") for l in ls]
    SD_files = ["../../../../synthetic/size_tests/500K/train_images/{}.png".format(idx) for idx in range(len(SD_smiles))]
    temp = list(zip(SD_smiles, SD_files)) 
    random.shuffle(temp) 
    SD_smiles, SD_files = zip(*temp) 
    SD_smiles = list(SD_smiles)
    SD_files = list(SD_files)


    # determine number of HD and SD datapoints according to ratio
    n_SD = int(train_size*ratio[0])
    n_HD = int(train_size*ratio[1])
    " - number of synthetic imgs in training set: ", n_SD
    " - number of hand-drawn imgs in training set: ", n_HD
    
    # Augment/degrade hand-drawn data to get number needed
    print "Augmenting hand-drawn images"
    HD_aug_imgs, HD_aug_smiles = augment_HD(HD_files, n_HD)
    print " - {} augmented hand-drawn images built".format(len(HD_aug_imgs))
 
    # Read synthetic data images
    print "Collecting synthetic images"
    SD_files = SD_files[0:n_SD]
    SD_smiles = SD_smiles[0:n_SD]
    SD_imgs = get_SD_imgs(SD_files)
    print " - {} synthetic images built".format(len(SD_imgs))

    # Combine synthetic and hand-drawn images to for training set
    print "Combining synthetic and hand-drawn images and shuffling"
    train_imgs = SD_imgs + HD_aug_imgs
    train_smiles = SD_smiles + HD_aug_smiles
    assert (len(train_imgs) == len(train_smiles))
     
    # Shuffle data
    z = zip(train_imgs, train_smiles)
    random.shuffle(z)

    # Write train.formula.txt, train.index.txt and image files 
    write_data(z)
    
    print " - Done."

if __name__ == "__main__":
    
    n = 350000
    # Run test with n=5 first to check its working

    ratios = [(0.0, 1.0), (0.1, 0.9), (0.5, 0.5), (0.9, 0.1), (1.0, 0.0)]
    
    for ratio in ratios:
        path = "{}_{}/".format(int(ratio[0]*100), int(ratio[1]*100))
        
        if not os.path.isdir(path):
            os.mkdir(path)    
        os.chdir(path)

        make_train_set(n, ratio)
        
        os.chdir("..")
