import os, sys, glob
# import tensorflow as tf
import pandas as pd
import numpy as np
from read_roi import read_roi_file, read_roi_zip
import skimage
from tqdm import tqdm

def get_valid_file_pairs(path : str, imgtype : str = "tif", roitype : str = "zip") -> dict:
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
        for roi_file in roi_files:
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
    valid_roi_files = list(np.array(roi_files)[match_roi_index])
    valid_img_files = list(np.array(img_files)[matches.flatten()])

    return dict({'img':valid_img_files, 'roi':valid_roi_files})
