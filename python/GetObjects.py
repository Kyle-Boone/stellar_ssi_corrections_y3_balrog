'''
This file contains four methods used for getting the objects used throughout the pipeline. 
The first function "getMatStars" gets all the data for Balrog delta stars that are detected.
The second function "getDetStarPositions" gets positions for every injected Balrog delta star.
The third and fourth functions "getMatGalas" and "getDetGalaPositions" perform similar processes but for Balrog galaxies.
'''

import numpy as np
import fitsio
import healpy as hp
from astropy.table import Table
from matplotlib.path import Path


def getMatStars(path, mu, res, matStarFile, detStarFile, validPixFile, writeFile, gCut, classCutoff, cutID = False, valIDFile = None):
    '''
    This method is used to store data on Balrog delta stars that have been detected.
    All files are assumed to be .fits files.
    path is a matplotlib path object describing the isochrone of interest in absolute g magnitude vs gr color space.
    mu is a float for the distance modulus.
    res is an integer corresponing to the healpixel resolution of the validPixFile.
    matStarFile should be the file with information on detected Balrog delta stars (matched table).
    detStarFile should be the file with information on all injected Balrog delta stars (detection table).
    validPixFile contains information on the valid healpixels to use (for this work, each survey property has a valid value).
    writeFile is the file to write information on the detected stars to.
    gCut is a float corresponding to an upper limit cut on g magnitudes.
    classCutoff is a float corresponding to the class given to detected objects. 
    If the EXTENDED_CLASS_SOF field is less than the classCutoff it is called a star and vice versa.
    cutID is an optional indicator for whether cuts will be applied based on the bal_id field. 
    This is primarily used in testing with restricting to some random subset of Balrog objects.
    valIDFile is the file that contains IDs to include if a cut is performed based on ID.
    '''
    
    # Read in data.
    matStarData = fitsio.read(matStarFile, columns = ['true_RA_new', 'true_DEC_new', 
                                                        'meas_EXTENDED_CLASS_SOF',
                                                        'meas_psf_mag', 'meas_cm_mag', 'bal_id'])
    
    detStarData = fitsio.read(detStarFile, columns = ['bal_id', 'flags_foreground', 'flags_badregions', 
                                                'flags_footprint', 'match_flag_1.5_asec'])
    

    RA = matStarData['true_RA_new']
    DEC = matStarData['true_DEC_new']
    
    GMAG_PSF = matStarData['meas_psf_mag'][:,0]
    RMAG_PSF = matStarData['meas_psf_mag'][:,1]
    
    GMAG_CM = matStarData['meas_cm_mag'][:,0]
    RMAG_CM = matStarData['meas_cm_mag'][:,1]
    
    CLASS = matStarData['meas_EXTENDED_CLASS_SOF']
    
    MAT_ID  = matStarData['bal_id']
    
    # This performs a cut on the bal_id field if cutID is true.
    if cutID:
        valID = fitsio.read(valIDFile)['ID']
        idCut = np.isin(MAT_ID, valID)
        
        RA = RA[idCut]
        DEC = DEC[idCut]
        
        GMAG_PSF = GMAG_PSF[idCut]
        RMAG_PSF = RMAG_PSF[idCut]
        
        GMAG_CM = GMAG_CM[idCut]
        RMAG_CM = RMAG_CM[idCut]
        
        CLASS = CLASS[idCut]
        
        MAT_ID = MAT_ID[idCut]
    
    # Magnitudes are the psf magnitude if the object was classified as a star and the cm magnitude if the object was classified as a galaxy.
    GMAG = np.copy(GMAG_PSF)
    RMAG = np.copy(RMAG_PSF)
    
    GMAG[np.where((CLASS >= classCutoff) & (CLASS <= 3))] = GMAG_CM[np.where((CLASS >= classCutoff) & (CLASS <= 3))]
    RMAG[np.where((CLASS >= classCutoff) & (CLASS <= 3))] = RMAG_CM[np.where((CLASS >= classCutoff) & (CLASS <= 3))]
    
    # Sorting based on the bal_id field to assist later with flag cuts.
    sortInds = MAT_ID.argsort()
    MAT_ID = MAT_ID[sortInds[::1]]
    
    RA = RA[sortInds[::1]]
    DEC = DEC[sortInds[::1]]
    
    GMAG = GMAG[sortInds[::1]]
    RMAG = RMAG[sortInds[::1]]
    
    CLASS = CLASS[sortInds[::1]]
    
    # Reading in information on flags for all injected Balrog delta stars.
    FOREGROUND = detStarData['flags_foreground']
    BADREGIONS = detStarData['flags_badregions']
    FOOTPRINT = detStarData['flags_footprint']
    ARCSECONDS = detStarData['match_flag_1.5_asec']
    
    FLAG_ID = detStarData['bal_id']
    
    # Making sure that our detected Balrog delta stars are a subset of all injected Balrog delta stars.
    sanityCut = np.isin(MAT_ID, FLAG_ID)
    MAT_ID = MAT_ID[sanityCut]
    
    RA = RA[sanityCut]
    DEC = DEC[sanityCut]
    
    GMAG = GMAG[sanityCut]
    RMAG = RMAG[sanityCut]
    
    CLASS = CLASS[sanityCut]
    
    # Sorting flags based on bal_id field to assist with flag cuts.
    sortInds = FLAG_ID.argsort()
    FLAG_ID = FLAG_ID[sortInds[::1]]
    
    FOREGROUND = FOREGROUND[sortInds[::1]]
    BADREGIONS = BADREGIONS[sortInds[::1]]
    FOOTPRINT = FOOTPRINT[sortInds[::1]]
    ARCSECONDS = ARCSECONDS[sortInds[::1]]
    
    # This cuts the flags specifically to the detected Balrog delta stars.
    cropInds = np.isin(FLAG_ID, MAT_ID)
            
    FOREGROUND = FOREGROUND[cropInds]
    BADREGIONS = BADREGIONS[cropInds]
    FOOTPRINT = FOOTPRINT[cropInds]
    ARCSECONDS = ARCSECONDS[cropInds]

    # Cutting based on flags, class, and magnitude.
    cutIndices = np.where((FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2) &
                          (CLASS >= 0) &
                          (CLASS <= 3) &
                          (GMAG < gCut))[0]
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    
    GMAG = GMAG[cutIndices]
    RMAG = RMAG[cutIndices]
    
    CLASS = CLASS[cutIndices]
    
    # Isochrone cut using absolute g magnitude "MG" and gr color "GR".
    MG = GMAG - mu
    GR = GMAG - RMAG
    
    filterSelection=Path.contains_points(path,np.vstack((GR,MG)).T)
    
    RA = RA[filterSelection]
    DEC = DEC[filterSelection]
    
    GMAG = GMAG[filterSelection]
    RMAG = RMAG[filterSelection]
    
    CLASS = CLASS[filterSelection]
    
    # Valid pixel cut
    validPix = fitsio.read(validPixFile)['PIXEL']
    
    currentPix = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)
    
    pixCut = np.isin(currentPix, validPix)
    
    RA = RA[pixCut]
    DEC = DEC[pixCut]
    
    GMAG = GMAG[pixCut]
    RMAG = RMAG[pixCut]
    
    CLASS = CLASS[pixCut]
    
    # Writing out data.
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table.write(writeFile, overwrite = True)
    
    
def getDetStarPositions(res, detStarFile, validPixFile, writeFile, cutID = False, valIDFile = None):
    '''
    This method is used to store positional data on every injected Balrog delta star.
    All files are assumed to be .fits files.
    res is an integer corresponing to the healpixel resolution of the validPixFile.
    detStarFile should be the file with information on all injected Balrog delta stars (detection table).
    validPixFile contains information on the valid healpixels to use (for this work, each survey property has a valid value).
    writeFile is the file to write information on the injected stars to.
    cutID is an optional indicator for whether cuts will be applied based on the bal_id field. 
    This is primarily used in testing with restricting to some random subset of Balrog objects.
    valIDFile is the file that contains IDs to include if a cut is performed based on ID.
    '''
    
    # Read in data.
    detStarData = fitsio.read(detStarFile, columns = ['true_ra', 'true_dec',
                                                      'flags_foreground', 'flags_badregions', 
                                                      'flags_footprint', 'match_flag_1.5_asec', 'bal_id'])
    
    validPix = fitsio.read(validPixFile)['PIXEL']
    
    ID = detStarData['bal_id']
    
    RA = detStarData['true_ra']
    DEC = detStarData['true_dec']
    
    FOREGROUND = detStarData['flags_foreground']
    BADREGIONS = detStarData['flags_badregions']
    FOOTPRINT = detStarData['flags_footprint']
    ARCSECONDS = detStarData['match_flag_1.5_asec']
    
    # This performs a cut on the bal_id field if cutID is true.
    if cutID:
        valID = fitsio.read(valIDFile)['ID']
        idCut = np.isin(ID, valID)
        
        RA = RA[idCut]
        DEC = DEC[idCut]
        
        FOREGROUND = FOREGROUND[idCut]
        BADREGIONS = BADREGIONS[idCut]
        FOOTPRINT = FOOTPRINT[idCut]
        ARCSECONDS = ARCSECONDS[idCut]
    
    # Cutting based on flags.
    flagCut = np.where((FOREGROUND == 0) & 
                       (BADREGIONS < 2) & 
                       (FOOTPRINT == 1) & 
                       (ARCSECONDS < 2))[0]
    
    RA = RA[flagCut]
    DEC = DEC[flagCut]
    
    # Valid pixel cut.
    PIX = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)
    
    pixCut = np.isin(PIX, validPix)
    
    PIX = PIX[pixCut]
    
    # Writing out data.
    my_table = Table()
    my_table['PIXEL'] = PIX
    my_table.write(writeFile, overwrite = True)
    
    
def getMatGalas(path, mu, res, deepFiles, matGalaFile, detGalaFile, validPixFile, writeFile, gCut, classCutoff, cutID = False, valIDFile = None):
    '''
    This method is used to store data on Balrog galaxies that have been detected.
    All files are assumed to be .fits files.
    path is a matplotlib path object describing the isochrone of interest in absolute g magnitude vs gr color space.
    mu is a float for the distance modulus.
    res is an integer corresponing to the healpixel resolution of the validPixFile.
    deepFiles is a list of files containing information on deep field objects.
    matGalaFile should be the file with information on detected Balrog galaxies (matched table).
    detGalaFile should be the file with information on all injected Balrog galaxies (detection table).
    validPixFile contains information on the valid healpixels to use (for this work, each survey property has a valid value).
    writeFile is the file to write information on the detected stars to.
    gCut is a float corresponding to an upper limit cut on g magnitudes.
    classCutoff is a float corresponding to the class given to detected objects. 
    If the EXTENDED_CLASS_SOF field is less than the classCutoff it is called a star and vice versa.
    cutID is an optional indicator for whether cuts will be applied based on the bal_id field. 
    This is primarily used in testing with restricting to some random subset of Balrog objects.
    valIDFile is the file that contains IDs to include if a cut is performed based on ID.
    '''
    
    # Reading in data from the deep fields.
    deepCols  = ['KNN_CLASS', 'ID']
    deepID = []
    deepClass= []
    
    for deepFile in deepFiles:
        deepData = fitsio.read(deepFile, columns = deepCols)
        deepID.extend(deepData['ID'])
        deepClass.extend(deepData['KNN_CLASS'])
    
    deepID = np.array(deepID)
    deepClass = np.array(deepClass)
    
    # This will allow me to quickly check object classification without having an overly long array.
    minID = np.min(deepID)
    deepGalID = np.zeros(np.max(deepID) - minID + 1)
    deepGalID[deepID - minID] = deepClass
    
    # Read in data
    wideCols = ['true_id', 'bal_id', 'true_ra', 'true_dec', 'meas_EXTENDED_CLASS_SOF', 'meas_cm_mag', 'meas_psf_mag']
    matBalrData = fitsio.read(matGalaFile, columns = wideCols)
    
    # This ID will be used to match galaxies to their deep field counterparts.
    ID = matBalrData['true_id']
    
    BALR_ID = matBalrData['bal_id']
    
    RA = matBalrData['true_ra']
    DEC = matBalrData['true_dec']
    
    CLASS = matBalrData['meas_EXTENDED_CLASS_SOF']
    
    GMAG_CM = matBalrData['meas_cm_mag'][:,0]
    RMAG_CM = matBalrData['meas_cm_mag'][:,1]
    
    GMAG_PSF = matBalrData['meas_psf_mag'][:,0]
    RMAG_PSF = matBalrData['meas_psf_mag'][:,1]
    
    # This performs a cut on the bal_id field if cutID is true.
    if cutID:
        valID = fitsio.read(valIDFile)['ID']
        idCut = np.isin(BALR_ID, valID)
        
        RA = RA[idCut]
        DEC = DEC[idCut]
        GMAG_PSF = GMAG_PSF[idCut]
        RMAG_PSF = RMAG_PSF[idCut]
        
        GMAG_CM = GMAG_CM[idCut]
        RMAG_CM = RMAG_CM[idCut]
        
        CLASS = CLASS[idCut]
        
        BALR_ID = BALR_ID[idCut]
        
        ID = ID[idCut]
    
    # Magnitudes are the psf magnitude if the object was classified as a star and the cm magnitude if the object was classified as a galaxy.
    GMAG = np.copy(GMAG_CM)
    RMAG = np.copy(RMAG_CM)
    
    GMAG[np.where((CLASS <= classCutoff) & (CLASS >= 0))] = GMAG_PSF[np.where((CLASS <= classCutoff) & (CLASS >= 0))]
    RMAG[np.where((CLASS <= classCutoff) & (CLASS >= 0))] = RMAG_PSF[np.where((CLASS <= classCutoff) & (CLASS >= 0))]
    
    # Sorting based on the bal_id field to assist later with flag cuts.
    sortInds = BALR_ID.argsort()
    BALR_ID = BALR_ID[sortInds[::1]]
    
    ID = ID[sortInds[::1]]
    
    RA = RA[sortInds[::1]]
    DEC = DEC[sortInds[::1]]
    
    GMAG = GMAG[sortInds[::1]]
    RMAG = RMAG[sortInds[::1]]
    
    CLASS = CLASS[sortInds[::1]]
    
    # Reading in information on flags for all injected Balrog delta stars.
    detBalrData = fitsio.read(detGalaFile, columns = ['bal_id', 'flags_foreground', 'flags_badregions', 'flags_footprint', 'match_flag_1.5_asec'])
    
    FLAG_ID = detBalrData['bal_id']
    
    FOREGROUND = detBalrData['flags_foreground']
    BADREGIONS = detBalrData['flags_badregions']
    FOOTPRINT = detBalrData['flags_footprint']
    ARCSECONDS = detBalrData['match_flag_1.5_asec']
    
    # Making sure that our detected Balrog galaxies are a subset of all injected Balrog galaxies.
    sanityCut = np.isin(BALR_ID, FLAG_ID)
    BALR_ID = BALR_ID[sanityCut]
    
    ID = ID[sanityCut]
    
    RA = RA[sanityCut]
    DEC = DEC[sanityCut]
    
    GMAG = GMAG[sanityCut]
    RMAG = RMAG[sanityCut]
    
    CLASS = CLASS[sanityCut]
    
    # Sorting flags based on bal_id field to assist with flag cuts.
    sortInds = FLAG_ID.argsort()
    FLAG_ID = FLAG_ID[sortInds[::1]]
    
    FOREGROUND = FOREGROUND[sortInds[::1]]
    BADREGIONS = BADREGIONS[sortInds[::1]]
    FOOTPRINT = FOOTPRINT[sortInds[::1]]
    ARCSECONDS = ARCSECONDS[sortInds[::1]]
    
    # This cuts the flags specifically to the detected Balrog galaxies.
    cropInds = np.isin(FLAG_ID, BALR_ID)
            
    FOREGROUND = FOREGROUND[cropInds]
    BADREGIONS = BADREGIONS[cropInds]
    FOOTPRINT = FOOTPRINT[cropInds]
    ARCSECONDS = ARCSECONDS[cropInds]

    # Cutting based on flags, class, and magnitude.
    cutIndices = np.where((FOREGROUND == 0) & 
                          (BADREGIONS < 2) & 
                          (FOOTPRINT == 1) & 
                          (ARCSECONDS < 2) &
                          (CLASS >= 0) &
                          (CLASS <= 3)
                          (GMAG < gCut))[0]

    ID = ID[cutIndices]
    RA = RA[cutIndices]
    DEC = DEC[cutIndices]
    GMAG = GMAG[cutIndices]
    RMAG = RMAG[cutIndices]
    CLASS = CLASS[cutIndices]

    # Cutting to objects modeled on deep field galaxies.
    trueClass = deepGalID[ID - minID]
    useInds = np.where((trueClass == 1))[0]

    RA = RA[useInds]
    DEC = DEC[useInds]
    GMAG = GMAG[useInds]
    RMAG = RMAG[useInds]
    CLASS = CLASS[useInds]
    
    # Isochrone cut using absolute g magnitude "MG" and gr color "GR".
    MG = GMAG - mu
    GR = GMAG - RMAG
    
    filterSelection=Path.contains_points(path,np.vstack((GR,MG)).T)
    
    RA = RA[filterSelection]
    DEC = DEC[filterSelection]
    GMAG = GMAG[filterSelection]
    RMAG = RMAG[filterSelection]
    CLASS = CLASS[filterSelection]
    
    # Valid pixel cut.
    validPix = fitsio.read(validPixFile)['PIXEL']
    
    currentPix = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)
    
    pixCut = np.isin(currentPix, validPix)
    
    RA = RA[pixCut]
    DEC = DEC[pixCut]
    GMAG = GMAG[pixCut]
    RMAG = RMAG[pixCut]
    CLASS = CLASS[pixCut]
    
    # Writing out data.
    my_table = Table()
    my_table['RA'] = RA
    my_table['DEC'] = DEC
    my_table['GMAG'] = GMAG
    my_table['RMAG'] = RMAG
    my_table['CLASS'] = CLASS
    my_table.write(writeFile, overwrite = True)
    
    
def getDetGalaPositions(res, deepFiles, detGalaFile, validPixFile, writeFile, cutID = False, valIDFile = None):
    '''
    This method is used to store positional data on all injected Balrog galaxies.
    All files are assumed to be .fits files.
    res is an integer corresponing to the healpixel resolution of the validPixFile.
    deepFiles is a list of files containing information on deep field objects.
    detGalaFile should be the file with information on all injected Balrog galaxies (detection table).
    validPixFile contains information on the valid healpixels to use (for this work, each survey property has a valid value).
    writeFile is the file to write information on the detected stars to.
    cutID is an optional indicator for whether cuts will be applied based on the bal_id field. 
    This is primarily used in testing with restricting to some random subset of Balrog objects.
    valIDFile is the file that contains IDs to include if a cut is performed based on ID.
    '''
    
    # Reading in data from the deep fields.
    deepCols  = ['KNN_CLASS', 'ID']
    deepID = []
    deepClass= []
    
    for deepFile in deepFiles:
        deepData = fitsio.read(deepFile, columns = deepCols)
        deepID.extend(deepData['ID'])
        deepClass.extend(deepData['KNN_CLASS'])
    
    deepID = np.array(deepID)
    deepClass = np.array(deepClass)
    
    # This will allow me to quickly check object classification without having an overly long array.
    minID = np.min(deepID)
    deepGalID = np.zeros(np.max(deepID) - minID + 1)
    deepGalID[deepID - minID] = deepClass
    
    # Read in data
    detGalaData = fitsio.read(detGalaFile, columns = ['true_id', 'true_ra', 'true_dec',
                                                      'flags_foreground', 'flags_badregions', 
                                                      'flags_footprint', 'match_flag_1.5_asec', 'bal_id'])
    
    validPix = fitsio.read(validPixFile)['PIXEL']
    
    BALR_ID = detGalaData['bal_id']
    
    ID = detGalaData['true_id']
    
    RA = detGalaData['true_ra']
    DEC = detGalaData['true_dec']
    
    FOREGROUND = detGalaData['flags_foreground']
    BADREGIONS = detGalaData['flags_badregions']
    FOOTPRINT = detGalaData['flags_footprint']
    ARCSECONDS = detGalaData['match_flag_1.5_asec']
    
    # This performs a cut on the bal_id field if cutID is true.
    if cutID:
        valID = fitsio.read(valIDFile)['ID']
        idCut = np.isin(BALR_ID, valID)
        
        RA = RA[idCut]
        DEC = DEC[idCut]
        
        FOREGROUND = FOREGROUND[idCut]
        BADREGIONS = BADREGIONS[idCut]
        FOOTPRINT = FOOTPRINT[idCut]
        ARCSECONDS = ARCSECONDS[idCut]
        
        ID = ID[idCut]
    
    # Cutting based on flags.
    flagCut = np.where((FOREGROUND == 0) & 
                       (BADREGIONS < 2) & 
                       (FOOTPRINT == 1) & 
                       (ARCSECONDS < 2))[0]
    
    ID = ID[flagCut]
    
    RA = RA[flagCut]
    DEC = DEC[flagCut]
    
    # Cutting to objects modeled on deep field galaxies.
    idCut = np.where((deepGalID[ID - minID] == 1))[0]
    
    RA = RA[idCut]
    DEC = DEC[idCut]
    
    # Valid pixel cut.
    PIX = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)
    
    pixCut = np.isin(PIX, validPix)
    
    PIX = PIX[pixCut]
    
    # Writing out data.
    my_table = Table()
    my_table['PIXEL'] = PIX
    my_table.write(writeFile, overwrite = True)