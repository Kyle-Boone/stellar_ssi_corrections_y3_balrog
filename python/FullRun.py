'''
This is a full run generating everything needed for corrections.
'''

import numpy as np
import matplotlib.pyplot as plt
import fitsio
import astropy.io.fits as fits
from astropy.table import Table
import healpy as hp
import healsparse as hsp
import skyproj
from os import listdir
import astropy.units as u
from astropy.coordinates import SkyCoord
import Config
import StellarConfig as strConfig
from CropSurveyProperties import *
from GetObjects import *
from Classification import *
from Detection import *

# Cycle limit for relative detection rate and classification probability calculations respectively.
detIndLim = 300
claIndLim = 150

# Define hyperparameters.
res = strConfig.res
numMagBins = strConfig.numMagBins
numBins = strConfig.numBins
classCutoff = strConfig.classCutoff
goldCols = strConfig.goldCols
gCut = strConfig.gCut
magBins = strConfig.magBins
cutOffPercent = strConfig.cutOffPercent
binNum = strConfig.binNum

# Columns necessary from the deep fields.
deepCols = strConfig.deepCols

# Isochrone configuration.
path = strConfig.path
mu = strConfig.mu

# Original balrog data files.
matBalrGalaFile = strConfig.matBalrGalaFile
detBalrGalaFile = strConfig.detBalrGalaFile
matBalrStarFile = strConfig.matBalrStarFile
detBalrStarFile = strConfig.detBalrStarFile

# Galaxy information files.
matGalaFile = strConfig.matGalaFile
detGalaAllPosFile = strConfig.detGalaAllPosFile

# Star information files.
matStarFile = strConfig.matStarFile
detStarAllPosFile = strConfig.detStarAllPosFile

# Originl deep field data.
deepFiles = strConfig.deepFiles

# Original survey property files.
origCondFiles = strConfig.origCondFiles
stelFile = strConfig.stelFile

# Valid pixel and updated survey property files.
pixFile = strConfig.pixFile
condFiles = strConfig.condFiles

# Galaxy correction files.
galaExtrFiles = strConfig.galaExtrFiles
galaTrainFiles = strConfig.galaTrainFiles
galaProbFiles = strConfig.galaProbFiles

galaDetAsStarExtrFiles = strConfig.galaDetAsStarExtrFiles
galaDetAsStarTrainFiles = strConfig.galaDetAsStarTrainFiles
galaDetAsStarProbFiles = strConfig.galaDetAsStarProbFiles

galaDetAsGalaExtrFiles = strConfig.galaDetAsGalaExtrFiles
galaDetAsGalaTrainFiles = strConfig.galaDetAsGalaTrainFiles
galaDetAsGalaProbFiles = strConfig.galaDetAsGalaProbFiles

galaDetAsAnyExtrFiles = strConfig.galaDetAsAnyExtrFiles
galaDetAsAnyTrainFiles = strConfig.galaDetAsAnyTrainFiles
galaDetAsAnyProbFiles = strConfig.galaDetAsAnyProbFiles

# Star correction files.
starExtrFiles = strConfig.starExtrFiles
starTrainFiles = strConfig.starTrainFiles
starProbFiles = strConfig.starProbFiles

starDetAsStarExtrFiles = strConfig.starDetAsStarExtrFiles
starDetAsStarTrainFiles = strConfig.starDetAsStarTrainFiles
starDetAsStarProbFiles = strConfig.starDetAsStarProbFiles

starDetAsGalaExtrFiles = strConfig.starDetAsGalaExtrFiles
starDetAsGalaTrainFiles = strConfig.starDetAsGalaTrainFiles
starDetAsGalaProbFiles = strConfig.starDetAsGalaProbFiles

starDetAsAnyExtrFiles = strConfig.starDetAsAnyExtrFiles
starDetAsAnyTrainFiles = strConfig.starDetAsAnyTrainFiles
starDetAsAnyProbFiles = strConfig.starDetAsAnyProbFiles

# Files with information on Y3 Gold objects.
goldStarFiles = strConfig.goldStarFiles
goldGalaFiles = strConfig.goldGalaFiles

goldObjectsDir = strConfig.goldObjectsDir
goldObjectsFiles = strConfig.goldObjectsFiles

goldMoreInfoStarFiles = strConfig.goldMoreInfoStarFiles
goldMoreInfoGalaFiles = strConfig.goldMoreInfoGalaFiles

# Deep field classification rate calibration file.
calibrationFile = strConfig.calibrationFile

# Get valid pixels and cropped survey properties.
validPixCropData(origCondFiles, stelFile, pixFile, condFiles)

validPix = fitsio.read(pixFile)['PIXEL']
pixCheck = np.full(12*(res**2), False, dtype = bool)
pixCheck[validPix] = True

# Get Stars:
getMatStars(path, mu, res, matBalrStarFile, detBalrStarFile, pixFile, matStarFile, gCut, classCutoff)
getDetStarPositions(res, detBalrStarFile, pixFile, detStarAllPosFile)

# Get Galaxies:
getMatGalas(path, mu, res, deepFiles, matBalrGalaFile, detBalrGalaFile, pixFile, matGalaFile, gCut, classCutoff)
getDetGalaPositions(res, deepFiles, detBalrGalaFile, pixFile, detGalaAllPosFile)

# Start and end inds to hopefully guarantee no crashing occurs
startInds = 2 * np.arange(50)
endInds = (2 * np.arange(50)) + 1

# Star classification

singleCorrectionTrain(matStarFile, condFiles, pixFile, magBins, starTrainFiles, starProbFiles, starExtrFiles, numBins, res, True, classCutoff, binNum, cutOffPercent, doFullSky = False, indLenLim = claIndLim)

fullSkyBool = [True, True, True]
for i in range(len(starTrainFiles)):
    if loadtxt(starTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
        fullSkyBool[i] = False

for i in range(3):
    if fullSkyBool[i]:
        for j in range(50):
            fullSky(pixFile, condFiles, np.array([starTrainFiles[i]]), np.array([starProbFiles[i]]), np.array([starExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
    else:
        aveAcc = loadtxt(starTrainFiles[i][0:-5] + '_Ave_Acc.csv', delimiter=',')
        aveAcc = 1 * aveAcc

        prob_table = Table()
        prob_table['SIGNAL'] = aveAcc * np.ones(len(validPix))
        prob_table.write(starProbFiles[i], overwrite = True) 

        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
        ext_table.write(starExtrFiles[i], overwrite = True)
        
# Galaxy classification
singleCorrectionTrain(matGalaFile, condFiles, pixFile, magBins, galaTrainFiles, galaProbFiles, galaExtrFiles, numBins, res, False, classCutoff, binNum, cutOffPercent, doFullSky = False, indLenLim = claIndLim)

fullSkyBool = [True, True, True]
for i in range(len(galaTrainFiles)):
    if loadtxt(galaTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
        fullSkyBool[i] = False

for i in range(3):
    print(i)
    if fullSkyBool[i]:
        for j in range(50):
            fullSky(pixFile, condFiles, np.array([galaTrainFiles[i]]), np.array([galaProbFiles[i]]), np.array([galaExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
    else:
        aveAcc = loadtxt(galaTrainFiles[i][0:-5] + '_Ave_Acc.csv', delimiter=',')
        aveAcc = 1 * aveAcc

        prob_table = Table()
        prob_table['SIGNAL'] = aveAcc * np.ones(len(validPix))
        prob_table.write(galaProbFiles[i], overwrite = True) 

        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
        ext_table.write(galaExtrFiles[i], overwrite = True)
        
# Star det as star
singleCorrectionTrainDet(detStarAllPosFile, matStarFile, condFiles, pixFile, magBins, starDetAsStarTrainFiles, starDetAsStarProbFiles, starDetAsStarExtrFiles, numBins, res, True, classCutoff, binNum, cutOffPercent, doFullSky = False, indLenLim = detIndLim)

fullSkyBool = [True, True, True]
for i in range(len(starDetAsStarTrainFiles)):
    if loadtxt(starDetAsStarTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
        fullSkyBool[i] = False

for i in range(3):
    if fullSkyBool[i]:
        for j in range(50):
            fullSkyDet(pixFile, condFiles, np.array([starDetAsStarTrainFiles[i]]), np.array([starDetAsStarProbFiles[i]]), np.array([starDetAsStarExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
    else:
        aveDet = loadtxt(starDetAsStarTrainFiles[i][0:-5] + '_Ave_Det.csv', delimiter=',')
        aveDet = 1 * aveDet

        prob_table = Table()
        prob_table['SIGNAL'] = aveDet * np.ones(len(validPix))
        prob_table.write(starDetAsStarProbFiles[i], overwrite = True) 

        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
        ext_table.write(starDetAsStarExtrFiles[i], overwrite = True)
        
# Gala det as star
singleCorrectionTrainDet(detGalaAllPosFile, matGalaFile, condFiles, pixFile, magBins, galaDetAsStarTrainFiles, galaDetAsStarProbFiles, galaDetAsStarExtrFiles, numBins, res, True, classCutoff, binNum, cutOffPercent, doFullSky = False, indLenLim = detIndLim)

fullSkyBool = [True, True, True]
for i in range(len(galaDetAsStarTrainFiles)):
    if loadtxt(galaDetAsStarTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
        fullSkyBool[i] = False

for i in range(3):
    if fullSkyBool[i]:
        for j in range(50):
            fullSkyDet(pixFile, condFiles, np.array([galaDetAsStarTrainFiles[i]]), np.array([galaDetAsStarProbFiles[i]]), np.array([galaDetAsStarExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
    else:
        aveDet = loadtxt(galaDetAsStarTrainFiles[i][0:-5] + '_Ave_Det.csv', delimiter=',')
        aveDet = 1 * aveDet

        prob_table = Table()
        prob_table['SIGNAL'] = aveDet * np.ones(len(validPix))
        prob_table.write(galaDetAsStarProbFiles[i], overwrite = True) 

        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
        ext_table.write(galaDetAsStarExtrFiles[i], overwrite = True)
        
# Star det as gala
singleCorrectionTrainDet(detStarAllPosFile, matStarFile, condFiles, pixFile, magBins, starDetAsGalaTrainFiles, starDetAsGalaProbFiles, starDetAsGalaExtrFiles, numBins, res, False, classCutoff, binNum, cutOffPercent, doFullSky = False, indLenLim = detIndLim)

fullSkyBool = [True, True, True]
for i in range(len(starDetAsGalaTrainFiles)):
    if loadtxt(starDetAsGalaTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
        fullSkyBool[i] = False

for i in range(3):
    if fullSkyBool[i]:
        for j in range(50):
            fullSkyDet(pixFile, condFiles, np.array([starDetAsGalaTrainFiles[i]]), np.array([starDetAsGalaProbFiles[i]]), np.array([starDetAsGalaExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
    else:
        aveDet = loadtxt(starDetAsGalaTrainFiles[i][0:-5] + '_Ave_Det.csv', delimiter=',')
        aveDet = 1 * aveDet

        prob_table = Table()
        prob_table['SIGNAL'] = aveDet * np.ones(len(validPix))
        prob_table.write(starDetAsGalaProbFiles[i], overwrite = True) 

        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
        ext_table.write(starDetAsGalaExtrFiles[i], overwrite = True)
        
# Gala det as gala
singleCorrectionTrainDet(detGalaAllPosFile, matGalaFile, condFiles, pixFile, magBins, galaDetAsGalaTrainFiles, galaDetAsGalaProbFiles, galaDetAsGalaExtrFiles, numBins, res, False, classCutoff, binNum, cutOffPercent, doFullSky = False, indLenLim = detIndLim)

fullSkyBool = [True, True, True]
for i in range(len(galaDetAsGalaTrainFiles)):
    if loadtxt(galaDetAsGalaTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
        fullSkyBool[i] = False

for i in range(3):
    if fullSkyBool[i]:
        for j in range(50):
            fullSkyDet(pixFile, condFiles, np.array([galaDetAsGalaTrainFiles[i]]), np.array([galaDetAsGalaProbFiles[i]]), np.array([galaDetAsGalaExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
    else:
        aveDet = loadtxt(galaDetAsGalaTrainFiles[i][0:-5] + '_Ave_Det.csv', delimiter=',')
        aveDet = 1 * aveDet

        prob_table = Table()
        prob_table['SIGNAL'] = aveDet * np.ones(len(validPix))
        prob_table.write(galaDetAsGalaProbFiles[i], overwrite = True) 

        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
        ext_table.write(galaDetAsGalaExtrFiles[i], overwrite = True)

# The following is unnecessary for the pipeline but was used in producing a plot for the DES MWWG group meeting.
# Star det as any
# singleCorrectionTrainDet(detStarAllPosFile, matStarFile, condFiles, pixFile, magBins, starDetAsAnyTrainFiles, starDetAsAnyProbFiles, starDetAsAnyExtrFiles, numBins, res, True, 3.5, binNum, cutOffPercent, doFullSky = False, indLenLim = detIndLim)

# fullSkyBool = [True, True, True]
# for i in range(len(starDetAsAnyTrainFiles)):
#     if loadtxt(starDetAsAnyTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
#         fullSkyBool[i] = False

# for i in range(3):
#     if fullSkyBool[i]:
#         for j in range(50):
#             fullSkyDet(pixFile, condFiles, np.array([starDetAsAnyTrainFiles[i]]), np.array([starDetAsAnyProbFiles[i]]), np.array([starDetAsAnyExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
#     else:
#         aveDet = loadtxt(starDetAsAnyTrainFiles[i][0:-5] + '_Ave_Det.csv', delimiter=',')
#         aveDet = 1 * aveDet

#         prob_table = Table()
#         prob_table['SIGNAL'] = aveDet * np.ones(len(validPix))
#         prob_table.write(starDetAsAnyProbFiles[i], overwrite = True) 

#         ext_table = Table()
#         ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
#         ext_table.write(starDetAsAnyExtrFiles[i], overwrite = True)
        
# Gala det as any
# singleCorrectionTrainDet(detGalaAllPosFile, matGalaFile, condFiles, pixFile, magBins, galaDetAsAnyTrainFiles, galaDetAsAnyProbFiles, galaDetAsAnyExtrFiles, numBins, res, True, 3.5, binNum, cutOffPercent, doFullSky = False, indLenLim = detIndLim)

# fullSkyBool = [True, True, True]
# for i in range(len(galaDetAsAnyTrainFiles)):
#     if loadtxt(galaDetAsAnyTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
#         fullSkyBool[i] = False

# for i in range(3):
#     if fullSkyBool[i]:
#         for j in range(50):
#             fullSkyDet(pixFile, condFiles, np.array([galaDetAsAnyTrainFiles[i]]), np.array([galaDetAsAnyProbFiles[i]]), np.array([galaDetAsAnyExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
#     else:
#         aveDet = loadtxt(galaDetAsAnyTrainFiles[i][0:-5] + '_Ave_Det.csv', delimiter=',')
#         aveDet = 1 * aveDet

#         prob_table = Table()
#         prob_table['SIGNAL'] = aveDet * np.ones(len(validPix))
#         prob_table.write(galaDetAsAnyProbFiles[i], overwrite = True) 

#         ext_table = Table()
#         ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
#         ext_table.write(galaDetAsAnyExtrFiles[i], overwrite = True)
        
# Deep field calibrations.

# Spatial matching.
def findMatches(angleCutoff, RASource, DECSource, RAMatchCatalog, DECMatchCatalog, nthneighbor=1):
    c = SkyCoord(ra=RASource*u.degree, dec=DECSource*u.degree)
    catalog = SkyCoord(ra=RAMatchCatalog*u.degree, dec=DECMatchCatalog*u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog, nthneighbor=nthneighbor)
    matches = d2d < angleCutoff
    return matches, d2d

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

deepFlagCuts = np.where((deepFlag == 0) &
                        (deepFlagNir == 0) &
                        (deepRA < 120) &
                        (deepClass > 0) &
                        (deepClass <= 3))[0]

deepRA = deepRA[deepFlagCuts]
deepDEC = deepDEC[deepFlagCuts]
deepClass = deepClass[deepFlagCuts]

if len(np.where(deepClass == 3)[0]) != 0:
    print('WARNING: Objects with no class are present in this deep field selection. ' + str(len(np.where(deepClass == 3)[0])) + ' object(s) out of ' + str(len(deepClass)) + ' have an ambiguous classification.')

deepPix = np.unique(hp.ang2pix(res, deepRA, deepDEC, lonlat = True, nest = True))

deepPixCheck = np.full(12*(res**2), False, dtype = bool)
deepPixCheck[deepPix] = True

starAdjustments = []
galaAdjustments = []

for i in np.arange(len(goldMoreInfoStarFiles)):
    allStarData = fitsio.read(goldMoreInfoStarFiles[i])
    allStarRA = allStarData['RA']
    allStarDEC = allStarData['DEC']
    allStarPIX = hp.ang2pix(res, allStarRA, allStarDEC, lonlat = True, nest = True)
    allStarRA = allStarRA[np.where(deepPixCheck[allStarPIX])[0]]
    allStarDEC = allStarDEC[np.where(deepPixCheck[allStarPIX])[0]]
    print(len(allStarRA))

    allGalaData = fitsio.read(goldMoreInfoGalaFiles[i])
    allGalaRA = allGalaData['RA']
    allGalaDEC = allGalaData['DEC']
    allGalaPIX = hp.ang2pix(res, allGalaRA, allGalaDEC, lonlat = True, nest = True)
    allGalaRA = allGalaRA[np.where(deepPixCheck[allGalaPIX])[0]]
    allGalaDEC = allGalaDEC[np.where(deepPixCheck[allGalaPIX])[0]]
    print(len(allGalaRA))
    
    deepStarMatches, _ = findMatches(0.5*u.arcsec, deepRA, deepDEC, allStarRA, allStarDEC)
    deepGalaMatches, _ = findMatches(0.5*u.arcsec, deepRA, deepDEC, allGalaRA, allGalaDEC)

    matchedDeepStarRA = deepRA[deepStarMatches]
    matchedDeepStarDEC = deepDEC[deepStarMatches]
    matchedDeepStarClass = deepClass[deepStarMatches]

    matchedDeepGalaRA = deepRA[deepGalaMatches]
    matchedDeepGalaDEC = deepDEC[deepGalaMatches]
    matchedDeepGalaClass = deepClass[deepGalaMatches]
    
    TSPIX = hp.ang2pix(res, matchedDeepStarRA[np.where(matchedDeepStarClass == 2)[0]], matchedDeepStarDEC[np.where(matchedDeepStarClass == 2)[0]], lonlat = True, nest = True)
    FSPIX = hp.ang2pix(res, matchedDeepStarRA[np.where(matchedDeepStarClass == 1)[0]], matchedDeepStarDEC[np.where(matchedDeepStarClass == 1)[0]], lonlat = True, nest = True)

    TGPIX = hp.ang2pix(res, matchedDeepGalaRA[np.where(matchedDeepGalaClass == 1)[0]], matchedDeepGalaDEC[np.where(matchedDeepGalaClass == 1)[0]], lonlat = True, nest = True)
    FGPIX = hp.ang2pix(res, matchedDeepGalaRA[np.where(matchedDeepGalaClass == 2)[0]], matchedDeepGalaDEC[np.where(matchedDeepGalaClass == 2)[0]], lonlat = True, nest = True)
    
    starCorrProb = np.clip(fitsio.read(starProbFiles[i])['SIGNAL'], 0, 1)
    fullStarProb = np.full(12*(res**2), hp.UNSEEN)
    fullStarProb[validPix] = starCorrProb

    galaCorrProb = np.clip(fitsio.read(galaProbFiles[i])['SIGNAL'], 0, 1)
    fullGalaProb = np.full(12*(res**2), hp.UNSEEN)
    fullGalaProb[validPix] = galaCorrProb
    
    starAdjustments.append(len(TSPIX) / (np.sum(fullStarProb[TSPIX[np.where(pixCheck[TSPIX])[0]]]) + np.sum(fullStarProb[FGPIX[np.where(pixCheck[FGPIX])[0]]])))   
    galaAdjustments.append(len(TGPIX) / (np.sum(fullGalaProb[TGPIX[np.where(pixCheck[TGPIX])[0]]]) + np.sum(fullGalaProb[FSPIX[np.where(pixCheck[FSPIX])[0]]])))
    
caliTable = Table()
caliTable['STAR'] = starAdjustments
caliTable['GALA'] = galaAdjustments
caliTable.write(calibrationFile, overwrite = True)