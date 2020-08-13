import os, sys, glob
import tensorflow as tf
import pandas as pd
import numpy as np
from read_roi import read_roi_file, read_roi_zip
import skimage
from tqdm import tqdm
from typing import Dict, Tuple, List, Callable
import joblib
from multiprocessing import cpu_count

"""
Submodule with data preprocessing routines for REN

Author:
    Fabian Wilde, Bioinformatics, University of Greifswald
"""

def get_valid_file_pairs(path : str, imgtype : str = "tif", roitype : str = "zip") -> Dict[Tuple[str],Tuple[str]]:
    """Lists directory content and tries to match image files of given type e.g tiff 
    with (zipped) roi files

    Args:
        path (str): path to directory with image and (zipped) roi files
        imgtype (str): image file type to be used (if multiple present)

    Returns:
        dict: contains ordered lists for matched pairs of image and roi files
    """

    #check function arguments
    valid_roi_types = ['roi', 'zip']
    if not roitype in valid_roi_types:
        raise KeyError("Invalid roi file type given. Expected one of the following:" + str(valid_roi_types))
    valid_img_types = ['tif', 'png', 'jpg']
    if not imgtype in valid_img_types:
        raise KeyError("Invalid img file type given. Expected one of the following:" + str(valid_img_types))
    #check if path contains a string with a valid path
    foo=os.stat(path)

    # scan directory with image and roi files
    img_files = glob.glob(path + "*." + imgtype)
    roi_files = glob.glob(path + "*." + roitype)

    if roitype == "zip":
        #test found zip files, if they contain a roi file
        for roi_file in tqdm(roi_files):
            try:
                foo=read_roi_zip(roi_file)
            except Exception:
                print("Zip file:")
                print(roi_file)
                print("is not a valid zip file or does not contain a roi file.")
                #remove invalid zip file from list
                roi_files.pop(roi_files.index(roi_file))
                continue

    if not ((len(img_files) > 0)|(len(roi_files) > 0)):
        raise ValueError("No image or valid roi files found in given directory.")

    #extract filenames without path or extension
    img_names = [os.path.split(x)[-1].split(".")[0] for x in img_files]
    roi_names = [os.path.split(x)[-1].split(".")[0] for x in roi_files]

    #match file pairs, image file matches roi file, if roi file name = image file[*].zip
    matches = [np.where(np.array([roi_name.find(img_name) for img_name in img_names]) == 0)[0] for roi_name in roi_names]
    match_roi_index = np.array(range(0,len(matches)))
    match_counts = np.array([match.shape for match in matches]).flatten()

    #discard image files with no or multiple matching roi file(s)
    valids = np.array(match_counts == 1)
    matches = np.array(matches)[valids]
    match_roi_index = match_roi_index[valids]
    invalids = ~valids
    invalids_num = np.sum(invalids)

    if invalids_num > 0:
        print("Discarded "+str(invalids_num)+" image files without or multiple matching roi file(s).")

    if np.sum(valids) == 0:
        raise ValueError("No valid pairs of image and roi files could be matched in the given directory.")

    #assemble valid pair lists
    valid_roi_files = tuple(np.array(roi_files)[match_roi_index])
    valid_img_files = tuple(np.array(img_files)[matches.flatten()])

    return dict({'img':valid_img_files, 'roi':valid_roi_files})

def roi_size_stats(roi_files : List[str]):
    """
    Function to analyze roi number and size distributions (per image). Should be used to choose an adequate tile size.

    Args:
    """
    pass

#function to randomly load one of the images and rois, randomly select a tile within that image, randomly apply a rotation.#make sure to choose a tile size so that no ROI is cut to avoid boundary effects

def get_random_tile(img_data : np.ndarray, roi_data : np.ndarray, tile_size : Tuple[int] = (600,600), improve_tile : bool = True, rotate_tile : bool = True, improve_fn : Callable[] = None) -> Dict[tf.TensorArray,tf.TensorArray]:
    """
    Function to randomly select a tile from the image and roi data matrices.

    Args:
        img_data (np.ndarray): path to image files
        roi_data (np.ndarray): path to roi files (*.zip or *.roi)
        tile_size (Tuple[int]): size for tile in px
        rotate_tile (bool): flag whether to perform a random rotation on the image tile
        improve_tile (bool): flag whether selected image tile should be improved (contrast, brightness, etc...)
        improve_fn (Callable): user-defined function handle to act on the selected image tile

    Returns:
        tf.TensorArray: enhanced greyscale image tile and binary roi mask
    """
    
    if not isinstance(img_data,np.ndarray):
        raise TypeError("Invalid data type for image data. Expected numpy.ndarray.")

    if not isinstance(roi_data,np.ndarray):
        raise TypeError("Invalid data type for roi data. Expected numpy.ndarray.")

    if not len(tile_size) == 2:
        raise ValueError("Invalid tile size.")

    #get raw image dimensions
    img_dims = img_data.shape
    #check if tile_size is compatible with image size
    if (tile_size[0] >= img_dims[0])|(tile_size[1] >= img_dims[1]):
        raise ValueError("One or more dimensions of the specified tile size bigger than input image array.")
    #get random edge
    rand_edge_x = np.random.randint(0,high=img_dims[0]-tile_size[0])
    rand_edge_y = np.random.randint(0,high=img_dims[1]-tile_size[1])
    #select tile from raw image
    tile_img = img_dat[rand_edge_x : rand_edge_x + tile_size[0],rand_edge_y : rand_edge_y + tile_size[1]]
    tile_roi = roi_data[rand_edge_x : rand_edge_x + tile_size[0],rand_edge_y : rand_edge_y + tile_size[1]]
    if improve_tile:
        if callable(improve_fn):
            tile_img = improve_fn(tile_img)
        else:
            #use skimage.exposure.adjust_log, adjust_gamma and/or equalize_hist
            pass

    if rotate_tile:
        #rotation angle can only be an integer multiple of 90 in order to avoid clipping when matrix size should be conserved
        tile_img = skimage.transform.rotate(tile_img, np.random.randint(0,high=3)*90, resize = False, preserve_range = True)
        tile_roi = skimage.transform.rotate(tile_roi, np.random.randint(0, high=3)*90, resize = False, preserve_range = True)

def load_file_pair(img_file : str, roi_file : str) -> Dict[np.ndarray,np.ndarray]:
    #test if img_file contains a valid file pat
    foo = os.stat(img_file)
    img_format = os.path.split(img_file)[-1].split(".")[-1]

    #read image file, if image format if tiff, use method from scikit-image
    #since PIL or tensorflow.io can only handle tiff files with 8bit color depth.
    if img_format in ["tif","tiff"]:
        raw_img = skimage.io.imread(img_file, plugin="tifffile")
    else:
        raise NotImplementedError()

    foo  = os.stat(roi_file)
    roi_format = os.path.split(roi_file)[-1].split(".")[-1]
    if roi_format == "zip":
        raw_roi = read_roi_zip(roi_file)
    elif roi_format == "roi":
        raw_roi = read_roi_file(roi_file)
    else:
        raise ValueError("Invalid file type for roi file.")

    #convert roi polygon coordinates to binary mask


    return raw_img, raw_roi

def generate_tile_set(img_files : Tuple[str], roi_files : Tuple[str]):
    pass


