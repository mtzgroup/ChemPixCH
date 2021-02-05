import os
from multiprocessing import Pool

import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

'''
This file generates RDKit image datasets for the datasets: 
  - 10K
  - 50K
  - 100K
  - 200K
  - 500K.
The formulas of each dataset are read from the folder and the RDKit
images are written to train_images, val_images and test_images folders.
'''

def get_smiles(filename):
    ''' Read SMILES data from *.formulas.txt file'''
    with open(filename) as f:
        lines = f.readlines()
    smiles = [s.split()[0] for s in lines]
    return smiles


def smiles_to_rdkit(s, i, img_dir):
    '''Generate RDKit image from SMILES string'''
    m = Chem.MolFromSmiles(s)
    AllChem.Compute2DCoords(m)

    # Create PNG file from SVG to get higher quality image
    Draw.MolToFile(m, "{}/{}.svg".format(img_dir,i), wedgeBonds=False)
    svg = svg2rlg("{}/{}.svg".format(img_dir,i))
    renderPM.drawToFile(svg, "{}/{}.png".format(img_dir,i), fmt="PNG")
    os.system("rm {}/{}.svg".format(img_dir,i))


if __name__ == "__main__":
    
    # RDKit settings
    Draw.DrawingOptions.bondLineWidth = 3
    
    for size in ["10K", "50K", "100K", "200K", "500K"]:
        print "Building RDKit images for {} SMILES dataset".format(size)
        for d in ["train", "val", "test"]:
            print " > Writing {} set to {}/{}_images".format(d,size,d)
            
            # Make image directory
            img_dir = "{}/{}_images".format(size, d)
            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)

            # Collect SMILES data
            smiles_file = "{}/{}.formulas.txt".format(size, d)
            smiles = get_smiles(smiles_file)
            
            # Build dataset
            for idx, s in enumerate(smiles):
                smiles_to_rdkit(s, idx, img_dir)

        print "   - Done.\n"

