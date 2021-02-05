"""
This file reads a file of smiles strings and generates a 
hand-drawn chemical structure dataset of these molecules.

1. Collect smiles strings from txt file
2. Collect background images
3. For each smiles string:
    3a. Convert smiles string to ong of molecule
    3b. Augment molecule image using molecule 
        augmentation pipeline
    3c. Randomly select background image
    3d. Augment background image using background
        augmentation pipeline
    3e. Combine augmented molecule and augmented background
        using random weighted addition
    3f. Degrade total image
    3g. Save image to folder 
"""

import cv2
import os
import glob
import numpy as np
import random 

from multiprocessing import Pool

import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

from RDKit_modified.mol_drawing import MolDrawing
from RDKit_modified.local_canvas import Canvas

from degrade import degrade_img
from augment import augment_mol, augment_bkg


def get_smiles(filename):
    ''' Read smiles data from *.formulas.txt file'''
    with open(filename) as f:
        lines = f.readlines()
    smiles = [s.split()[0] for s in lines]
    return smiles


def get_background_imgs(path):
    '''Reads in background dataset'''
    bkg_files = glob.glob("{}/*.png".format(path))
    bkgs = [cv2.imread(b) for b in bkg_files]
    return bkgs


def smiles_to_rdkitmod(s, i, img_dir):
    '''Generate RDKit image from smiles string'''
    m = Chem.MolFromSmiles(s)
    AllChem.Compute2DCoords(m)

    # Use modified RDKit code to drawn molecule
    canvas = Canvas(size=(300, 300), name='{}/{}'.format(img_dir,i), imageType='svg')
    drawer = MolDrawing(canvas, drawingOptions=Draw.DrawingOptions)
    drawer.AddMol(m)
    canvas.flush()
    canvas.save()

    # Convert SVG file to PNG 
    svg = svg2rlg("{}/{}.svg".format(img_dir,i))
    renderPM.drawToFile(svg, "{}/{}.png".format(img_dir,i), fmt="PNG")
    os.system("rm {}/{}.svg".format(img_dir,i)) 


def smiles_to_synthetic(s, i, img_dir, stage):
    
    # Convert smiles string to RDKit' image
    smiles_to_rdkitmod(s, i, img_dir)
    if stage == "rdkit*":
        return
    mol = cv2.imread("{}/{}.png".format(img_dir,i))

    # Augment molecule imag
    mol_aug = augment_mol(mol)
    if stage == "rdkit*-aug":
        cv2.imwrite("{}/{}.png".format(img_dir,i), mol_aug)
        return

    # Randomly select background image
    bkg = random.choice(bkgs)

    # Augment background image
    bkg_aug = augment_bkg(bkg)

    # Combine augmented molecule and augmented background
    # using random weighted addition
    p = np.random.uniform(0.1,0.8)
    mol_bkg = cv2.addWeighted(bkg_aug, p, mol_aug, 1-p, gamma=0)
    if stage == "rdkit*-aug-bkg":
        cv2.imwrite("{}/{}.png".format(img_dir,i), mol_bkg)
        return

    # Degrade total image
    mol_bkg_deg = degrade_img(mol_bkg)
    if stage == "rdkit*-aug-bkg-deg":
        cv2.imwrite("{}/{}.png".format(img_dir,i), mol_bkg_deg) 
        return


if __name__ == "__main__":
    
    # RDKit settings
    Draw.DrawingOptions.wedgeBonds = False
    Draw.DrawingOptions.wedgeDashedBonds = False
    Draw.DrawingOptions.bondLineWidth = np.random.uniform(1,7)
    
    # Collect background images
    path_bkg = "../backgrounds/"
    bkgs = get_background_imgs(path_bkg)
    
    stages = ["rdkit*", "rdkit*-aug", "rdkit*-aug-bkg", "rdkit*-aug-bkg-deg"]
    for stage in stages:
        print "Building synthetic data images for {} SMILES dataset".format(stage)
        for d in ["train", "val", "test"]:
            print " > Building {} set".format(d)
            
            # Make image directory
            img_dir = "{}/{}_images".format(stage, d)
            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)

            # Collect SMILES data
            smiles_file = "{}/{}.formulas.txt".format(stage, d)
            smiles = get_smiles(smiles_file)
            
            # Build dataset
            for idx, s in enumerate(smiles):
                smiles_to_synthetic(s, idx, img_dir, stage)
        print "   - Done.\n"

