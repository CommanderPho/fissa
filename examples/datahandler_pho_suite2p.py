"""FISSA functions to handle image and roi objects and return the right format.

If a custom version of this file is used (which can be defined at the
declaration of the core FISSA Experiment class), it should have the same
functions as here, with the same inputs and outputs.

Authors:
	Sander W Keemink <swkeemink@scimail.eu>
	Scott C Lowe <scott.code.lowe@gmail.com>

"""

from past.builtins import basestring

from typing import Union, Tuple, Optional
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from tifffile import imread, TiffFile, TiffWriter

from fissa import roitools


def open_tiff(file: str, sktiff: bool) -> Tuple[Union[TiffFile, ScanImageTiffReader], int]:
    """ Returns image and its length from tiff file with either ScanImageTiffReader or tifffile, based on 'sktiff'"""
    if sktiff:
        tif = TiffFile(file)
        Ltif = len(tif.pages)
    else:
        tif = ScanImageTiffReader(file)
        Ltif = 1 if len(tif.shape()) < 3 else tif.shape()[0]  # single page tiffs
    return tif, Ltif


def use_sktiff_reader(tiff_filename, batch_size: Optional[int] = None) -> bool:
    """Returns False if ScanImageTiffReader works on the tiff file, else True (in which case use tifffile)."""
    try:
        with ScanImageTiffReader(tiff_filename) as tif:
            tif.data() if len(tif.shape()) < 3 else tif.data(beg=0, end=np.minimum(batch_size, tif.shape()[0] - 1))
        return False
    except:
        print('NOTE: ScanImageTiffReader not working for this tiff type, using tifffile')
        return True


def image2array(image):
	"""Take the object 'image' and returns an array.

	Parameters
	---------
	image : unknown
		The data. Should be either a tif location, or a list
		of already loaded in data.

	Returns
	-------
	np.array
		A 3D array containing the data as (frames, y coordinate, x coordinate)
		Looks like it's only loading a single frame from each tif
	"""
	if isinstance(image, basestring):
		print('Loading tif at {}...\n'.format(image))
		# open tiff
		use_sktiff = False
		tif, Ltif = open_tiff(image, use_sktiff)
		if Ltif == 1:
			tif_data = tif.data()
		else:
			ix = 1 # Figure out why this offset would be anything but 1. Oh, It's for blocks. It will always be one.
			nTifFrames = Ltif
			tif_data = tif.data(beg=0, end=0 + nTifFrames)

		print('\t done. Contains {} frames.\n'.format(Ltif))
		# tif_data = tifffile.imread(image)
		return tif_data

	if isinstance(image, np.ndarray):
		return image


def getmean(data):
	"""Get the mean image for data.

	Parameters
	----------
	data : array
		Data array as made by image2array. Should be of shape [frames,y,x]

	Returns
	-------
	array
		y by x array for the mean values

	"""
	return data.mean(axis=0)


def rois2masks(rois, data):
	"""Take the object 'rois' and returns it as a list of binary masks.

	Parameters
	----------
	rois : unkown
		Either a string with imagej roi zip location, list of arrays encoding
		polygons, or binary arrays representing masks
	data : array
		Data array as made by image2array. Should be of shape [frames,y,x]

	Returns
	-------
	list
		List of binary arrays (i.e. masks)

	"""
	# get the image shape
	shape = data.shape[1:] # It looks like it's coming in with only a single 512x512 frame!

	# if it's a list of strings
	if isinstance(rois, basestring):
		rois = roitools.readrois(rois)

	print('PHO: rois2masks(...):\n \tdata.shape: {}\n \tdata.shape[1:]: {}\n \tnp.shape(rois[0]): {}\n'.format(data.shape, shape, np.shape(rois[0])))

	if isinstance(rois, list):
		# if it's a something by 2 array (or vice versa), assume polygons
		if np.shape(rois[0])[1] == 2 or np.shape(rois[0])[0] == 2:
			return roitools.getmasks(rois, shape)
		# if it's a list of bigger arrays, assume masks
		elif np.shape(rois[0]) == shape:
			return rois
		elif np.shape(rois[0]) == data.shape:
			print('PHO: WARNING: rois2masks(...): There is only a single frame in data. Still returning rois.')
			return rois
		else:
			print('PHO: size looks wrong!')
			print(np.shape(rois[0]))
			print(shape)

	else:
		raise ValueError('Pho: Wrong rois input format')



def extracttraces(data, masks):
	"""Get the traces for each mask in masks from data.

	Inputs
	--------------------
	data : array
		Data array as made by image2array. Should be of shape [frames,y,x]
	masks : list
		list of binary arrays (masks)

	"""
	# get the number rois and frames
	nrois = len(masks)
	nframes = data.shape[0]

	# predefine output data
	out = np.zeros((nrois, nframes))

	# loop over masks
	for i in range(nrois):  # for masks
		# get mean data from mask
		out[i, :] = data[:, masks[i]].mean(axis=1)

	return out
