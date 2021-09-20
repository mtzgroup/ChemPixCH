from fastapi import FastAPI, File

import numpy as np
from collections import Counter
from rdkit import Chem # if this is super slow then we can get rid of this step

from model.img2seq import Img2SeqModel
from model.utils.general import Config
from model.utils.text import Vocab

import cv2

app = FastAPI(docs_url=None, redoc_url=None)

def load_models(json_dir, weights_dirs):
    '''
    Loads in a model and restores weights

    Params:
     - json_dir: path to folder containing .json files
     - weights_dir: path to folder containing weights for models

    Returns:
     - model: neural network model for converting image to SMILES
    '''
    # load in vocab and model parameters
    config_vocab = Config(json_dir + "vocab.json")
    config_model = Config(json_dir + "model.json")
    vocab = Vocab(config_vocab)

    # build models and restore weights
    models = []
    for weights_dir in weights_dirs:
        model = Img2SeqModel(config_model, weights_dir, vocab)
        model.build_pred()
        model.restore_session(weights_dir + "model.weights/")
        models.append(model)

    return models


json_dir = "results-ensemble/"
ensembles = ["FT-500KHdval/", "FT-500K/", "HDtrain-ratios-90_10/", "HDval-sizes-500K/", "SD-sizes-500K/"]
weights_dirs = [json_dir + ensemble for ensemble in ensembles]
models = load_models(json_dir, weights_dirs)


@app.post("/recognize")
def recognize(file: bytes = File(...)):
    tmp = np.frombuffer(file, dtype=np.uint8)
    img = cv2.imdecode(tmp, flags=cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    img = img[:, :, np.newaxis].astype(np.uint8)

    counter = Counter()

    # predict SMILES
    for i in range(len(models)):
        model = models[i]
        hyp = model.predict(img)[0]

        # get canonical smiles - can be removed if really slow
        try:
            hyp_canon = str(Chem.MolToSmiles(Chem.MolFromSmiles(hyp)))
        except:
            hyp_canon = hyp

        # This is an optimization. Find the top two. If the second most common SMILES
        # cannot beat the most common SMILES even if it gets all the remaining votes,
        # then the most common SMILES wins. (Some randomness with ties.)
        counter.update({hyp_canon: 1})
        most_commons = counter.most_common(2)
        most_common_count = most_commons[0][1]
        second_most_common_count = most_commons[1][1] if len(most_commons) >= 2 else 0
        if most_common_count >= second_most_common_count + (len(models) - i - 1):
            break

    # get highest voted prediction
    prediction, count = counter.most_common(1)[0]
    
    # if all the models disagree, don't output anything.
    if count <= 1:
        return {"smileses": []}

    return {"smileses": [prediction]}
