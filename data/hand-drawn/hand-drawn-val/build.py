import shutil
import random
import glob
import os

def get_smiles(filename):
    ''' Read smiles data from *.formulas.txt file'''
    with open(filename) as f:
        lines = f.readlines()
    smiles = [s.split()[0] for s in lines]
    return smiles

if __name__ == "__main__":

    if not os.path.isdir("val_images"):
        os.mkdir("val_images")

    smiles = get_smiles("val.formulas.txt")

    for idx, s in enumerate(smiles):
        image_path = "../hand-drawn-full/{}.png".format(s)
        shutil.copy(image_path, "val_images/{}.png".format(idx))

