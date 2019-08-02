## This code including images preprocessing and AutoEncoder

This is Pytorch code for edge connection with GAN,in which a autoencoder structure is utilized for generator

# Folder Structure
* data                              : Images data
* DefectDataset                     : Images of groundtruth and augmented dataset  
    * gt
    * noise
    * background                    : Edges of clutter
* GAN_Image                         : Images during the training procedure
* Model                             
    * DIS
    * GAN
* Template                          : A set of well connected contour images
    * bin_contour
* Test_Image                        : The output of model
    * input
    * output
* src
    * AAE.py
    * AAEWithClassifier.py
    * datasetGenerate.py
    * testAAE.py
    * testAAEWithClassifier.py
    * region.py
* README.md
* script                            : Matlab scripts
* roi                               : Roi cropped by mask


# Usage

## 0. Install Library

`pytorch` <br>
`PIL` <br>
`cv2` <br>

## 1. Use AutoEncoder


```
## Go to src folder
cd src/

## Generate  dataset

python3 datasetGenerate.py

## Train AEGAN
python3 AAE.py

## Test autoencoder
python3 testAAE.py
```

## 2. Use Matlab scripts

Using Matlab opening guidence_template.m

## 3. Recently updata:

19.7.26:  Update AAE.py, region.py, testAAE.py; The problem that the holes inside mask expanded is resolved.

19.8.02:  Update AAEWithClassifier.py, testAAEWithClassifier.py; Add illumination normalization procedure for origin images, then choose SQI technique results as sources. 
