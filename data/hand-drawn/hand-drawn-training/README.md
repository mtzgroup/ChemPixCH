## Hand-drawn hydrocarbon and synthetic data training and validation set

**Used in Figure 8b and c**

 - 213 images of photographed hand-drawn hydrocarbons for training set
 - 200 images of photographed hand-drawn hydrocarbons for validation set

###Scripts:
Copies images from ../hand-drawn-full/ to build datasets:
 `$ python build.py`

### Training datasets:
Varying ratios of synthetic:hand-drawn data 
 - 0\_100: 0:100 
 - 10\_90: 0:100 
 - 50\_50: 0:100 
 - 90\_10: 0:100 
 - 100\_0: 0:100 

Hand-drawn images in training set are augmented and degraded.
