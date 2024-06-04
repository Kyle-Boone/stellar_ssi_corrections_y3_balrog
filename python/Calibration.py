'''
This file contains two methods related to calibrating average classification rates.
The first method "findMatches" is a spatial matching helper method for the second method.
The second method "calibrations" finds the proper scalar multiple for star and galaxy classification probabilities by comparing to the deep fields.
'''

import numpy as np
import fitsio
import astropy.io.fits as fits
from astropy.table import Table
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord


def findMatches(angleCutoff, RASource, DECSource, RAMatchCatalog, DECMatchCatalog, nthneighbor=1):
    '''
    Performs spatial matching between source objects and a match catalog.
    This should not be directly called by a user.
    '''
    c = SkyCoord(ra=RASource*u.degree, dec=DECSource*u.degree)
    catalog = SkyCoord(ra=RAMatchCatalog*u.degree, dec=DECMatchCatalog*u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog, nthneighbor=nthneighbor)
    matches = d2d < angleCutoff
    return matches, d2d


def calibrations(pixFile, calibrationFile, starProbFiles, galaProbFiles, goldMoreInfoStarFiles, goldMoreInfoGalaFiles, deepFiles, deepCols, res, matchDist):
    '''
    This calibrates star and galaxy classification rates.
    pixFile is where the valid pixels are stored.
    calibrationFile is where the calibration scalars will be stored.
    starProbFiles are the star classification probability files.
    galaProbFiles are the galaxy classification probability files.
    goldMoreInfoStar/GalaFiles are created in the Y3Objects.py program.
    These hold information on gold objects to compare to the deep fields.
    The Prob and MoreInfo files should be of the same length corresponding to magnitude bins.
    All magnitude bins can and should be done at once.
    deepFiles are files of the deep fields, and deepCols are the columns to read in with fitsio to speed up the process.
    res is the full resolution of calculations.
    matchDist is the distance in arcSeconds for a spatial match to the deep fields. 
    A value of 0.5 seems to work well for this.
    '''
    
    # Read in valid pixels
    validPix = fitsio.read(pixFile)['PIXEL']
    pixCheck = np.full(12*(res**2), False, dtype = bool)
    pixCheck[validPix] = True
    
    # Read in deep field information
    deepRA = np.array([])
    deepDEC = np.array([])
    deepClass = np.array([])
    deepFlag = np.array([])
    deepFlagNir = np.array([])

    for deepFile in deepFiles:
        deepData = fitsio.read(deepFile, columns = deepCols)

        deepRA = np.append(deepRA, deepData['RA'])
        deepDEC = np.append(deepDEC, deepData['DEC'])
        deepClass = np.append(deepClass, deepData['KNN_CLASS'])
        deepFlag = np.append(deepFlag, deepData['MASK_FLAGS'])
        deepFlagNir = np.append(deepFlagNir, deepData['MASK_FLAGS_NIR'])
    
    # Apply quality cuts and ambiguous class cuts to the deep fields.
    deepFlagCuts = np.where((deepFlag == 0) &
                            (deepFlagNir == 0) &
                            (deepRA < 120) &
                            (deepClass > 0) &
                            (deepClass <= 3))[0]

    deepRA = deepRA[deepFlagCuts]
    deepDEC = deepDEC[deepFlagCuts]
    deepClass = deepClass[deepFlagCuts]

    # No class objects with bands missing could present an issue so this is a warning.
    if len(np.where(deepClass == 3)[0]) != 0:
        print('WARNING: Objects with no class are present in this deep field selection. ' + str(len(np.where(deepClass == 3)[0])) + ' object(s) out of ' + str(len(deepClass)) + ' have an ambiguous classification.')

    # Check which healpixels contain deep field objects to crop wide field objects immediately.
    deepPix = np.unique(hp.ang2pix(res, deepRA, deepDEC, lonlat = True, nest = True))

    deepPixCheck = np.full(12*(res**2), False, dtype = bool)
    deepPixCheck[deepPix] = True
    
    starAdjustments = []
    galaAdjustments = []

    for i in np.arange(len(goldMoreInfoStarFiles)):
        # Get stars in deep field pixels.
        allStarData = fitsio.read(goldMoreInfoStarFiles[i])
        allStarRA = allStarData['RA']
        allStarDEC = allStarData['DEC']
        allStarPIX = hp.ang2pix(res, allStarRA, allStarDEC, lonlat = True, nest = True)
        allStarRA = allStarRA[np.where(deepPixCheck[allStarPIX])[0]]
        allStarDEC = allStarDEC[np.where(deepPixCheck[allStarPIX])[0]]

        # Get galaxies in deep field pixels.
        allGalaData = fitsio.read(goldMoreInfoGalaFiles[i])
        allGalaRA = allGalaData['RA']
        allGalaDEC = allGalaData['DEC']
        allGalaPIX = hp.ang2pix(res, allGalaRA, allGalaDEC, lonlat = True, nest = True)
        allGalaRA = allGalaRA[np.where(deepPixCheck[allGalaPIX])[0]]
        allGalaDEC = allGalaDEC[np.where(deepPixCheck[allGalaPIX])[0]]
        
        # Find the matches from deep field objects to wide field objects.
        deepStarMatches, _ = findMatches(matchDist*u.arcsec, deepRA, deepDEC, allStarRA, allStarDEC)
        deepGalaMatches, _ = findMatches(matchDist*u.arcsec, deepRA, deepDEC, allGalaRA, allGalaDEC)

        # Get information on the deep field objects matched to wide field stars.
        matchedDeepStarRA = deepRA[deepStarMatches]
        matchedDeepStarDEC = deepDEC[deepStarMatches]
        matchedDeepStarClass = deepClass[deepStarMatches]

        # Get information on the deep field objects matched to wide field galaxies.
        matchedDeepGalaRA = deepRA[deepGalaMatches]
        matchedDeepGalaDEC = deepDEC[deepGalaMatches]
        matchedDeepGalaClass = deepClass[deepGalaMatches]

        # Pixels of deep stars matched to wide stars.
        TSPIX = hp.ang2pix(res, matchedDeepStarRA[np.where(matchedDeepStarClass == 2)[0]], matchedDeepStarDEC[np.where(matchedDeepStarClass == 2)[0]], lonlat = True, nest = True)
        
        # Pixels of deep galaxies matched to wide stars.
        FSPIX = hp.ang2pix(res, matchedDeepStarRA[np.where(matchedDeepStarClass == 1)[0]], matchedDeepStarDEC[np.where(matchedDeepStarClass == 1)[0]], lonlat = True, nest = True)

        # Pixels of deep galaxies matched to wide galaxies.
        TGPIX = hp.ang2pix(res, matchedDeepGalaRA[np.where(matchedDeepGalaClass == 1)[0]], matchedDeepGalaDEC[np.where(matchedDeepGalaClass == 1)[0]], lonlat = True, nest = True)
        
        # Pixels of deep stars matched to wide galaxies.
        FGPIX = hp.ang2pix(res, matchedDeepGalaRA[np.where(matchedDeepGalaClass == 2)[0]], matchedDeepGalaDEC[np.where(matchedDeepGalaClass == 2)[0]], lonlat = True, nest = True)

        # Read in calculated probabilities.
        starCorrProb = np.clip(fitsio.read(starProbFiles[i])['SIGNAL'], 0, 1)
        fullStarProb = np.full(12*(res**2), hp.UNSEEN)
        fullStarProb[validPix] = starCorrProb

        galaCorrProb = np.clip(fitsio.read(galaProbFiles[i])['SIGNAL'], 0, 1)
        fullGalaProb = np.full(12*(res**2), hp.UNSEEN)
        fullGalaProb[validPix] = galaCorrProb

        # Number of correct classifications divided by the expected number of correct classifications.
        starAdjustments.append(len(TSPIX) / (np.sum(fullStarProb[TSPIX[np.where(pixCheck[TSPIX])[0]]]) + np.sum(fullStarProb[FGPIX[np.where(pixCheck[FGPIX])[0]]])))   
        galaAdjustments.append(len(TGPIX) / (np.sum(fullGalaProb[TGPIX[np.where(pixCheck[TGPIX])[0]]]) + np.sum(fullGalaProb[FSPIX[np.where(pixCheck[FSPIX])[0]]])))

    # Write out the calibrations.
    caliTable = Table()
    caliTable['STAR'] = starAdjustments
    caliTable['GALA'] = galaAdjustments
    caliTable.write(calibrationFile, overwrite = True)