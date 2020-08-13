REN
===

REN is the acronym for ROI Extraction Net and it offers the automatic extraction of ROIs as binary mask in (microscopic) images using a CNN implemented in Tensorflow.

Usage
-----

```Python
from ren.preprocessing import get_valid_file_pairs

valid_files = get_valid_file_pairs("data/", imgtype = "tif", roitype = "zip")

"""
yields dict with ordered lists of valid image and roi file pairs
"""
```

Requirements
------------

* python >= 3.5
* tensorflow >= 1.9
* https://github.com/hadim/read-roi (in order to read ImageJ's ROI files)
* scikit-image >= 0.16.2 (support for TIFFs with 16-bit color depth)
* tifffile as plugin for skimage.io.imread
* tqdm (fancy progress bar)
* joblib (easy parallelization)
* matplotlib (visualization)

Installation
------------

First download the zipped repository from here or clone the repository, then install the module with

```bash
pip install -e path_to_package/
```

Notes
-----
It is recommened to use the module tifffile (in combination with scikit-image) to read TIFF files with 16 bit color depth. Pillow (which is integrated in matplotlib) only supports 8 bit color depth which could mean an information loss. For really big TIFFs (approx. 1 GB) used in geoinformatics e.g. satellite imagery, the module osgeo.gdal is recommened.

License
-------
All source code is under the <a href="https://opensource.org/licenses/artistic-license-2.0">Artistic License</a>
