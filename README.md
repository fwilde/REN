REN
===

REN is the acronym for ROI Extraction Net and it offers the automatic extraction of ROIs as binary mask in (microscopic) images using a CNN implemented in Tensorflow.

Usage
-----

```Python
from ren import preprocess, plot, train, test

"""
yields Tensorflow tensors containing the enhanced, greyscale images and 
the binary masks of the ROIs for each image (target model output) 
for training and test

"""
```

Requirements
------------

* Python >= 3.5
* Tensorflow >= 1.9
* https://github.com/hadim/read-roi (in order to read ImageJ's ROI files)
* Scikit-image >= 0.16.2 (for input image preprocessing)
* Tqdm (fancy progress bar)
* Joblib (easy parallelization)
* Matplotlib (visualization)

Installation
------------

First download the zipped repository from here or clone the repository, then install the module with

```bash
pip install -e path_to_package/
```


