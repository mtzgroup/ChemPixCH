from scipy.misc import imread
import cv2

from model.img2seq import Img2SeqModel
from model.utils.general import Config, run
from model.utils.text import Vocab
from model.utils.image import greyscale, crop_image, pad_image, \
    downsample_image, TIMEOUT, augment_image



def interactive_shell(model):
    """Creates interactive shell to play with model
    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
Enter a path to a file
input> data/images_test/0.png""")

    while True:
        try:
            # for python 2
            img_path = raw_input("input> ")
        except NameError:
            # for python 3
            img_path = input("input> ")

        if img_path == "exit":
            break

        img = imread(img_path)

        img = cv2.resize(img, (256, 256))
        img = greyscale(img)
        hyps = model.predict(img)

        model.logger.info(hyps[0])


if __name__ == "__main__":
    # restore config and model
    dir_output = "results/small/"
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")
    vocab = Vocab(config_vocab)

    model = Img2SeqModel(config_model, dir_output, vocab)
    model.build_pred()
    model.restore_session(dir_output + "model.weights/")

    interactive_shell(model)
