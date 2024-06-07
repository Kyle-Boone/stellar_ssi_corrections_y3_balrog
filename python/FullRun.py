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
from Y3Objects import *
from Calibration import *

# Cycle limit for relative detection rate and classification probability calculations respectively.
detIndLim = 300
claIndLim = 150

# Define hyperparameters.
res = strConfig.res
numMagBins = strConfig.numMagBins
numBins = strConfig.numBins
classCutoff = strConfig.classCutoff
gCut = strConfig.gCut
magBins = strConfig.magBins
cutOffPercent = strConfig.cutOffPercent
binNum = strConfig.binNum
matchDist = strConfig.matchDist

# Isochrone configuration.
path = strConfig.path
mu = strConfig.mu

# Original balrog data files.
matBalrGalaFile = Config.matBalrGalaFile
detBalrGalaFile = Config.detBalrGalaFile
matBalrStarFile = Config.matBalrStarFile
detBalrStarFile = Config.detBalrStarFile

# Galaxy information files.
matGalaFile = strConfig.matGalaFile
detGalaAllPosFile = strConfig.detGalaAllPosFile

# Star information files.
matStarFile = strConfig.matStarFile
detStarAllPosFile = strConfig.detStarAllPosFile

# Originl deep field data.
deepFiles = Config.deepFiles
deepCols = Config.deepCols

# Original survey property files.
origCondFiles = Config.origCondFiles
stelFile = Config.stelFile

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

# Files with information on Y3 Gold objects.
goldObjectsDir = Config.goldObjectsDir
goldObjectsFiles = Config.goldObjectsFiles
goldCols = Config.goldCols

goldStarFiles = strConfig.goldStarFiles
goldGalaFiles = strConfig.goldGalaFiles

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
        
# Get Y3 Objects

getY3Objects(pixFile, goldObjectsDir, goldObjectsFiles, goldCols, goldMoreInfoStarFiles, goldMoreInfoGalaFiles, goldStarFiles, goldGalaFiles, res, magBins, numMagBins, classCutoff, gCut, path, mu)
        
# Deep field calibrations.

calibrations(pixFile, calibrationFile, starProbFiles, galaProbFiles, goldMoreInfoStarFiles, goldMoreInfoGalaFiles, deepFiles, deepCols, res, matchDist)