import os, sys, glob
import tensorflow as tf
import pandas as pd
import numpy as np
from read_roi import read_roi_file, read_roi_zip
import skimage
import skimage.io
import skimage.draw
import skimage.transform
from tqdm import tqdm
from typing import Dict, Tuple, List, Callable
from joblib import Parallel, delayed
from multiprocessing import cpu_count

"""
Submodule with data preprocessing routines for REN

Author:
    Fabian Wilde, Bioinformatics, University of Greifswald
"""

def get_valid_files(path : str, imgtype : str = "tif", roitype : str = "zip") -> Dict[Tuple[str],Tuple[str]]:
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
def get_random_tiles(img_data : np.ndarray, roi_data : np.ndarray, num_tiles : int = 32, tile_size : Tuple[int,int] = (600,600), improve_tile : bool = True, rotate_tile : bool = True, mirror_tile : bool = True, check_rois : bool = True, improve_fn : Callable = None, dtype : tf.DType = tf.uint16, verbose : bool = False) -> Dict[List[tf.Tensor],List[tf.Tensor]]:
    """
    Function to randomly select a tile from the image and roi data matrices.

    Args:
        img_data (np.ndarray): path to image files
        roi_data (np.ndarray): path to roi files (*.zip or *.roi)
        tile_size (Tuple[int]): size for tile in px
        rotate_tile (bool): flag whether to perform a random rotation on the image tile
        mirror_tile (bool): flag whether to mirror the image tile along a random axis
        check_rois (bool): flag whether to check that no roi is cut by tile boundaries
        improve_tile (bool): flag whether selected image tile should be improved (contrast, brightness, etc...)
        improve_fn (Callable): user-defined function handle to act on the selected image tile
        dtype (tf.DType): data type for Tensorflow output array
        verbose (bool): verbosity. If False, all prints to stdout are suppressed.

    Returns:
        Dict[List[tf.Tensor],List[tf.Tensor]]: enhanced greyscale image tiles and binary roi masks as Tensorflow Tensors
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

    if improve_tile:
        if callable(improve_fn):
            tile_img = improve_fn(tile_img)
        else:
            #use skimage.exposure.adjust_log, adjust_gamma and/or equalize_hist

            #from the scikit-image docs:
            #transforms the input image pixelwise according to the equation O = gain*log(1 + I) after
            #scaling each pixel to the range [0,1]
            #
            #massively improves edge/texture contrast in the microscope imagery of the podocytes
            #gamma adjust massively increases image noise and more background structures
            img_data = skimage.exposure.adjust_log(img_data, gain=2)
            #try clipping the pixel values on the lowest and highest 0.5% can improve constrast without amplifying the image noise
    
    #TODO: preallocation with static tensor should be faster...
    out_tiles = {'img':[], \
                 'roi':[]}

    i = 0
    if verbose:
        bar = tqdm(total=num_tiles)
    while i < num_tiles:
        #get random edge
        rand_edge_x = np.random.randint(0,high=img_dims[0]-tile_size[0])
        rand_edge_y = np.random.randint(0,high=img_dims[1]-tile_size[1])

        #select tile from raw image
        tile_img = img_data[rand_edge_x : rand_edge_x + tile_size[0],rand_edge_y : rand_edge_y + tile_size[1]]
        tile_roi = roi_data[rand_edge_x : rand_edge_x + tile_size[0],rand_edge_y : rand_edge_y + tile_size[1]]

        if rotate_tile:
        #rotation angle can only be an integer multiple of 90 in order to avoid clipping when matrix size should be conserved
            rand_rot_angle = np.random.randint(0,high=3)*90
            tile_img = skimage.transform.rotate(tile_img, rand_rot_angle, resize = False, preserve_range = True)
            tile_roi = skimage.transform.rotate(tile_roi, rand_rot_angle, resize = False, preserve_range = True)

        if mirror_tile:
            axis = np.random.randint(0,high=2)
            #mirror along x-axis
            if axis == 0:
                tile_img = tile_img[::-1,:]
                tile_roi = tile_roi[::-1,:]
            #mirror along y-axis
            elif axis == 1:
                tile_img = tile_img[:,::-1]
                tile_roi = tile_roi[:,::-1]
            #mirror along diagonal
            elif axis == 2:
                tile_img = tile_img[::-1,::-1]
                tile_roi = tile_roi[::-1,::-1]

        #check if any ROI in tile is cut by tile boundaries
        #basically check if boundary pixels are all zero
        if check_rois:
            boundaries=np.hstack((tile_roi[:,0],tile_roi[:,-1],tile_roi[0,:],tile_roi[-1,:])).flatten()
            if np.any(boundaries):
                #if any pixel on tile boundaries belongs to any roi, discard tile and start over again
                continue

        out_tiles["img"].append(tf.constant(tile_img, dtype=tf.uint16))
        out_tiles["roi"].append(tf.constant(tile_roi, dtype=tf.uint16))
        i += 1
        if verbose:
            bar.update()

    #transform TensorArray to Tensor
    #out_tiles["img"].stack()
    #out_tiles["roi"].stack()
    if verbose:
        bar.close()

    return out_tiles

def load_files(img_file : str, roi_file : str) -> Dict[np.ndarray,np.ndarray]:
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

    #extract polygon paths
    #raw_roi---%NAME.type
    #               .left
    #               .top
    #               .width
    #               .height
    #               .paths---List[List[Tuple]]
    #               .name
    #               .position

    #iterate over roi groups in loaded ROI file
    for roi_group_name in raw_roi:
        #check roi image dimensions
        roi_group = raw_roi[roi_group_name]
        roi_img_height = roi_group["height"]
        roi_img_width = roi_group["width"]
        img_width = raw_img.shape[0]
        img_height = raw_img.shape[1]

        #check dimensions of area where rois have been defined and the actual raw image
        if not (img_height >= roi_img_height):
            raise ValueError("Image height ("+str(img_height)+") smaller than height of ROI definition area ("+str(roi_img_height)+").")            

        if not (img_width >= roi_img_width):
            raise ValueError("Image width ("+str(img_width)+") smaller than width of ROI definition area ("+str(roi_img_width)+").")

    #convert roi polygon coordinates to binary mask:
    #-extract roi polygons
    roi_polygons = [np.array(roi,dtype=np.uint16) for roi in roi_group["paths"]]
    #init empty mask
    roi_mask = np.full(raw_img.shape, False, dtype=np.bool)
    #-draw roi polygons
    for roi_polygon in roi_polygons:
        #logic OR to obtain compund mask of all ROIs
        roi_mask = roi_mask | skimage.draw.polygon2mask(raw_img.shape, roi_polygon)

    roi_mask = roi_mask.T
    
    return raw_img, roi_mask

def generate_tile_set(img_path: str, roi_path : str = "", tiles_per_file : int = 32, tile_size : Tuple[int] = (600,600), img_type : str = "tif", roi_type : str = "zip", num_threads : int = 2) -> Dict[List[tf.Tensor],List[tf.Tensor]]:
    """
    Function to generate an augmented data set, a set of randomly selected, rotated and mirrored tiles,
    from a set of image/roi file pairs

    Args:
        img_path (str): string containing the path to the image files
        roi_path (str): optional string containing the path to the roi files
        tiles_per_file (int): number of tiles to generated from each image / roi file pair
        tile_size (Tuple[int]): tuple with tile size (has to be smaller or equal to image size)
        num_threads (int): number of parallel workers to generate the tile set. If set to -1, maximum available number of CPU threads is used.

    Returns:
        tile_set (Dict[tf.TensorArray,tf.TensorArray]): set of tiles from augmented data set as TensorArrays
    """

    #force using CPU only
    my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")

    #get valid pairs of image and roi file
    files = get_valid_files(img_path, imgtype = img_type, roitype = roi_type)
    img_files = files["img"]
    roi_files = files["roi"]
    array_size =  int(tiles_per_file * len(img_files))

    #probe image datatype by loading a single file
    test_img, test_mask = load_files(img_files[0],roi_files[0])
    img_dtype = test_img.dtype
    
    # allocate arrays
    tile_set = {'img': tf.TensorArray(img_dtype, size=array_size, dynamic_size=False, clear_after_read=False),\
                'roi': tf.TensorArray(tf.bool, size=array_size, dynamic_size=False, clear_after_read=False)}
    
    if num_threads == -1:
        num_threads = multiprocessing.cpu_count()

    def task(img_file, roi_file):
        img_data, roi_mask = load_files(img_file, roi_file)
        tiles = get_random_tiles(img_data, roi_mask, num_tiles = tiles_per_file, tile_size = tile_size)
        return tiles

    # parallel execution
    #tile_set = [task(img_files[i], roi_files[i]) for i in tqdm(range(len(img_files)))]
    tile_set = Parallel(n_jobs = num_threads)(delayed(task)(files["img"][i],files["roi"][i]) for i in tqdm(range(len(files["img"]))))
    return tile_set



