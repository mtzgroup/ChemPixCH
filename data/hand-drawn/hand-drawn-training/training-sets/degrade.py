import cv2
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import MiniBatchKMeans

def contrast(img):
    if np.random.uniform(0,1)<0.8: # increase contrast
        f = np.random.uniform(1,2)
    else: # decrease contrast
        f = np.random.uniform(0.5,1)
    im_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(im_pil)
    im  = enhancer.enhance(f)
    img = np.asarray(im)
    return np.asarray(im)


def brightness(img):
    f = np.random.uniform(0.4,1.1)
    im_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(im_pil)
    im  = enhancer.enhance(f)
    img = np.asarray(im)
    return np.asarray(im)


def sharpness(img):
    
    if np.random.uniform(0,1)<0.5: # increase sharpness
        f = np.random.uniform(0.1,1)
    else: # decrease sharpness
        f = np.random.uniform(1,10)
    im_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(im_pil)
    im  = enhancer.enhance(f)
    img = np.asarray(im)
    return np.asarray(im)


def s_and_p(img):
    amount = np.random.uniform(0.001, 0.01)
    # add some s&p
    row,col = img.shape
    s_vs_p = 0.5
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in img.shape]
    out[coords] = 1

    #pepper
    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in img.shape]
    out[coords] = 0
    
    return out

def scale(img):
    f = np.random.uniform(0.5,1.5)
    shape_OG = img.shape
    res = cv2.resize(img,None,fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
    res = cv2.resize(res,None,fx=1.0/f, fy=1.0/f, interpolation = cv2.INTER_CUBIC)
    return res


def quantize(img):
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    (h, w) = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    clt = MiniBatchKMeans(n_clusters = 2)
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h, w, 3))
    img = img.reshape((h, w, 3))

    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def invert(img):
    im_inv = cv2.bitwise_not(img)
    return im_inv

def darken(img):
    img =  cv2.subtract(img, np.random.uniform(0,50))
    return img


def degrade_img(img):
    
    # s+p    
    if np.random.uniform(0,1) < 0.1:
        img = s_and_p(img)
    
    # scale
    if np.random.uniform(0,1) < 0.5:
        img = scale(img)

    # brightness
    if np.random.uniform(0,1) < 0.7:
        img = brightness(img)        
    
    # contrast
    if np.random.uniform(0,1) < 0.7:
        img = contrast(img)
        
    # quantize
    if np.random.uniform(0,1) < 0.2:
        img = quantize(img)
        
    # sharpness
    if np.random.uniform(0,1) < 0.5:
        img = sharpness(img)
        
    # darken
    if np.random.uniform(0,1) < 0.5:
        img = darken(img)
    
    # invert
    if np.random.uniform(0,1) < 0.1:
        img = invert(img) 
    
    img = cv2.resize(img, (256,256))
    
    return img
