import fitsio
import numpy as np
import Config
import healpy as hp
from astropy.table import Table


def validPixCropData(condFiles, stelFile, pixWriteFile, cropWriteFiles):
    '''
    This method is used to get valid pixels and crop all survey property maps to those pixels.
    All files are assumed to be .fits files.
    condFiles is a list of all original survey property files excluding stellar density, which is formatted differently.
    stelFile is the file containing the stellar density.
    pixWriteFile will be the file where valid pixels are written to.
    cropWriteFiles will be a list of files to write all cropped survey properties to.
    '''
    
    # Generation of valid pixels.
    validPix = np.full(12*(4096**2), True, dtype = bool)
    
    for file in condFiles:
        # Read in data.
        condData = fitsio.read(file)
        condSigExt = np.full(12*(4096**2), -1.6375e+30)
        condSigExt[condData['PIXEL']] = condData['SIGNAL']
        
        # Valid pixels will only be pixels with valid values for every survey property.
        validPix[np.where(condSigExt < -100)] = False
    
    stelDensExt = fitsio.read(stelFile)['I'].flatten()
    validPix[np.where(stelDensExt < -100)] = False

    # This gives the actual valid healpixels instead of a boolean list.
    PIX = np.where(validPix)[0]
    
    # This stores the valid pixels in a fits file.
    pix_table = Table()
    pix_table['PIXEL'] = PIX
    pix_table.write(pixWriteFile, overwrite = True)
    
    # Crop each survey property and write the cropped map to the new file.
    for i in range(len(condFiles)):
        # Read in data.
        condData = fitsio.read(condFiles[i])
        condSigExt = np.full(12*(4096**2), -1.6375e+30)
        condSigExt[condData['PIXEL']] = condData['SIGNAL']
        
        # Valid pixel crop.
        condSig = condSigExt[PIX]
        
        # Writes data.
        cond_table = Table()
        cond_table['SIGNAL'] = condSig
        cond_table.write(cropWriteFiles[i], overwrite = True)
        
    # Same as above but for stellar density.
    stelDensExt = fitsio.read(stelFile)['I'].flatten()
    condSig = stelDensExt[PIX]
    stel_table = Table()
    stel_table['SIGNAL'] = condSig
    stel_table.write(cropWriteFiles[-1], overwrite = True)