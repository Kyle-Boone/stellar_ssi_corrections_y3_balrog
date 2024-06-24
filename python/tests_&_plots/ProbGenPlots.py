detIndLim = 300
claIndLim = 150

stellarDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/FinalPipeline/Tests/ProbPlots/80/'

import sys
sys.path.insert(1, '/afs/hep.wisc.edu/home/kkboone/software/StarWeights/FinalPipeline')
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

res = strConfig.res
numMagBins = 1
numBins = strConfig.numBins
classCutoff = strConfig.classCutoff
goldCols = Config.goldCols
gCut = strConfig.gCut
magBins = [23.9, 24.5]
cutOffPercent = strConfig.cutOffPercent
binNum = strConfig.binNum

path = strConfig.path
mu = strConfig.mu

matBalrStarFile = Config.matBalrStarFile
detBalrStarFile = Config.detBalrStarFile

pixFile = strConfig.pixFile
condFiles = strConfig.condFiles

validPix = fitsio.read(pixFile)['PIXEL']
pixCheck = np.full(12*(res**2), False, dtype = bool)
pixCheck[validPix] = True

injDir = stellarDir + 'InjectionData/'

valIDDir = injDir + 'ValidPix/'

matStarFile = injDir + 'Mat_Stars.fits'
detStarAllPosFile = injDir + 'Det_Stars_All_Position_Data.fits'

starDir = stellarDir + 'Stars/'

starExtrFiles = []
starTrainFiles =  []
starProbFiles = []
for i in np.arange(numMagBins):
    starExtrFiles.append(starDir + 'Star_Extr_Bin' + str(i+1) + '.fits')
    starTrainFiles.append(starDir + 'Star_Train_Bin' + str(i+1) + '.fits')
    starProbFiles.append(starDir + 'Star_Prob_Bin' + str(i+1) + '.fits')
    
starDetAsStarExtrFiles = []
starDetAsStarTrainFiles =  []
starDetAsStarProbFiles = []
for i in np.arange(numMagBins):
    starDetAsStarExtrFiles.append(starDir + 'Star_Det_As_Star_Extr_Bin' + str(i+1) + '.fits')
    starDetAsStarTrainFiles.append(starDir + 'Star_Det_As_Star_Train_Bin' + str(i+1) + '.fits')
    starDetAsStarProbFiles.append(starDir + 'Star_Det_As_Star_Prob_Bin' + str(i+1) + '.fits')
    
startInds = 2 * np.arange(50)
endInds = (2 * np.arange(50)) + 1

np.random.seed(0)

# starData = fitsio.read(detBalrStarFile, columns = ['true_ra', 'true_dec', 'bal_id'])
# starID = starData['bal_id']
# starRA = starData['true_ra']
# starDEC = starData['true_dec']

# starPix = hp.ang2pix(res, starRA, starDEC, lonlat = True, nest = True)

# uniqueStarPix = np.unique(starPix)

# includeInds = np.full(len(uniqueStarPix), False, dtype = bool)
# includeInds[0:int(0.8 * len(includeInds))] = True
# np.random.shuffle(includeInds)

# includePix = uniqueStarPix[includeInds]
# includeID = starID[np.isin(starPix, includePix)]

valIDFile = valIDDir + 'Val_ID.fits'
# my_table = Table()
# my_table['ID'] = includeID
# my_table.write(valIDFile, overwrite = True)

# Get Stars
# getMatStars(path, mu, res, matBalrStarFile, detBalrStarFile, pixFile, matStarFile, gCut, classCutoff, cutID = True, valIDFile = valIDFile)
# getDetStarPositions(res, detBalrStarFile, pixFile, detStarAllPosFile, cutID = True, valIDFile = valIDFile)

# Star classification
# singleCorrectionTrain(matStarFile, condFiles, pixFile, magBins, starTrainFiles, starProbFiles, starExtrFiles, numBins, res, True, classCutoff, binNum, cutOffPercent, doFullSky = False, indLenLim = claIndLim)

# fullSkyBool = [True, True, True]
# for i in range(len(starTrainFiles)):
#     if loadtxt(starTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
#         fullSkyBool[i] = False

# for i in range(3):
#     if fullSkyBool[i]:
#         for j in range(50):
#             fullSky(pixFile, condFiles, np.array([starTrainFiles[i]]), np.array([starProbFiles[i]]), np.array([starExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
#     else:
#         aveAcc = loadtxt(starTrainFiles[i][0:-5] + '_Ave_Acc.csv', delimiter=',')
#         aveAcc = 1 * aveAcc

#         prob_table = Table()
#         prob_table['SIGNAL'] = aveAcc * np.ones(len(validPix))
#         prob_table.write(starProbFiles[i], overwrite = True) 

#         ext_table = Table()
#         ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
#         ext_table.write(starExtrFiles[i], overwrite = True)
        
# Star det as star
# singleCorrectionTrainDet(detStarAllPosFile, matStarFile, condFiles, pixFile, magBins, starDetAsStarTrainFiles, starDetAsStarProbFiles, starDetAsStarExtrFiles, numBins, res, True, classCutoff, binNum, cutOffPercent, doFullSky = False, indLenLim = detIndLim)

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