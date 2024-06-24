perObjects = int(input("Enter percent of objects to use: "))
seed = int(input("Enter random seed (enter a negative seed to have the seed be 0 but stored in a different file): "))

stellarDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/FinalPipeline/Tests/Percent_Used/' + str(perObjects) + '/'

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
numMagBins = strConfig.numMagBins
numBins = strConfig.numBins
classCutoff = strConfig.classCutoff
goldCols = Config.goldCols
gCut = strConfig.gCut
magBins = strConfig.magBins
cutOffPercent = strConfig.cutOffPercent
binNum = strConfig.binNum

deepCols = Config.deepCols

path = strConfig.path
mu = strConfig.mu

matBalrGalaFile = Config.matBalrGalaFile
detBalrGalaFile = Config.detBalrGalaFile
matBalrStarFile = Config.matBalrStarFile
detBalrStarFile = Config.detBalrStarFile

deepFiles = Config.deepFiles
pixFile = strConfig.pixFile
condFiles = strConfig.condFiles

validPix = fitsio.read(pixFile)['PIXEL']
pixCheck = np.full(12*(res**2), False, dtype = bool)
pixCheck[validPix] = True

goldMoreInfoStarFiles = strConfig.goldMoreInfoStarFiles
goldMoreInfoGalaFiles = strConfig.goldMoreInfoGalaFiles

matStarFile = stellarDir + 'InjectionData/Mat_Stars_'+str(seed)+'.fits'
detStarAllPosFile = stellarDir + 'InjectionData/Det_Stars_All_Position_Data_'+str(seed)+'.fits'

matGalaFile = stellarDir + 'InjectionData/Mat_Galaxies_'+str(seed)+'.fits'
detGalaAllPosFile = stellarDir + 'InjectionData/Det_Galaxies_All_Position_Data_'+str(seed)+'.fits'

calibrationFile = stellarDir + 'Calibration/Calibrations_'+str(seed)+'.fits'

starDir = stellarDir + 'Stars/'

starExtrFiles = []
starTrainFiles =  []
starProbFiles = []
for i in np.arange(numMagBins):
    starExtrFiles.append(starDir + 'Star_Extr_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    starTrainFiles.append(starDir + 'Star_Train_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    starProbFiles.append(starDir + 'Star_Prob_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    
starDetAsStarExtrFiles = []
starDetAsStarTrainFiles =  []
starDetAsStarProbFiles = []
for i in np.arange(numMagBins):
    starDetAsStarExtrFiles.append(starDir + 'Star_Det_As_Star_Extr_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    starDetAsStarTrainFiles.append(starDir + 'Star_Det_As_Star_Train_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    starDetAsStarProbFiles.append(starDir + 'Star_Det_As_Star_Prob_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    
galaDir = stellarDir + 'Galaxies/'

galaExtrFiles = []
galaTrainFiles =  []
galaProbFiles = []
for i in np.arange(numMagBins):
    galaExtrFiles.append(galaDir + 'Gala_Extr_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    galaTrainFiles.append(galaDir + 'Gala_Train_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    galaProbFiles.append(galaDir + 'Gala_Prob_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    
galaDetAsStarExtrFiles = []
galaDetAsStarTrainFiles =  []
galaDetAsStarProbFiles = []
for i in np.arange(numMagBins):
    galaDetAsStarExtrFiles.append(galaDir + 'Gala_Det_As_Star_Extr_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    galaDetAsStarTrainFiles.append(galaDir + 'Gala_Det_As_Star_Train_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    galaDetAsStarProbFiles.append(galaDir + 'Gala_Det_As_Star_Prob_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    
# # Generate random id's, this has already been done
    
# galaID = fitsio.read(detBalrGalaFile, columns = ['bal_id'])['bal_id']

# np.random.seed(np.max(np.array([seed, 0])))

# includeInds = np.full(len(galaID), False, dtype = bool)
# includeInds[0:int((float(perObjects)*len(includeInds)) / 100)] = True
# np.random.shuffle(includeInds)

# valIDFile = stellarDir + 'InjectionData/Val_ID_'+str(seed)+'.fits'
# my_table = Table()
# my_table['ID'] = galaID[includeInds]
# my_table.write(valIDFile, overwrite = True)

# # Get Stars: This has already been done
# getMatStars(path, mu, res, matBalrStarFile, detBalrStarFile, pixFile, matStarFile, gCut, classCutoff, cutID = True, valIDFile = valIDFile)
# getDetStarPositions(res, detBalrStarFile, pixFile, detStarAllPosFile, cutID = True, valIDFile = valIDFile)

# # Get Galaxies: This has already been done
# getMatGalas(path, mu, res, deepFiles, matBalrGalaFile, detBalrGalaFile, pixFile, matGalaFile, gCut, classCutoff, cutID = True, valIDFile = valIDFile)
# getDetGalaPositions(res, deepFiles, detBalrGalaFile, pixFile, detGalaAllPosFile, cutID = True, valIDFile = valIDFile)

# Start and end inds to hopefully guarantee no crashing occurs
startInds = 2 * np.arange(50)
endInds = (2 * np.arange(50)) + 1

# # Star classification: This has already been done
# singleCorrectionTrain(matStarFile, condFiles, pixFile, magBins, starTrainFiles, starProbFiles, starExtrFiles, numBins, res, True, classCutoff, binNum, cutOffPercent, doFullSky = False)

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
        
# # Galaxy classification
# singleCorrectionTrain(matGalaFile, condFiles, pixFile, magBins, galaTrainFiles, galaProbFiles, galaExtrFiles, numBins, res, False, classCutoff, binNum, cutOffPercent, doFullSky = False)

# fullSkyBool = [True, True, True]
# for i in range(len(galaTrainFiles)):
#     if loadtxt(galaTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
#         fullSkyBool[i] = False

# for i in range(3):
#     print(i)
#     if fullSkyBool[i]:
#         for j in range(50):
#             fullSky(pixFile, condFiles, np.array([galaTrainFiles[i]]), np.array([galaProbFiles[i]]), np.array([galaExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
#     else:
#         aveAcc = loadtxt(galaTrainFiles[i][0:-5] + '_Ave_Acc.csv', delimiter=',')
#         aveAcc = 1 * aveAcc

#         prob_table = Table()
#         prob_table['SIGNAL'] = aveAcc * np.ones(len(validPix))
#         prob_table.write(galaProbFiles[i], overwrite = True) 

#         ext_table = Table()
#         ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
#         ext_table.write(galaExtrFiles[i], overwrite = True)
        
# # Star det as star
# singleCorrectionTrainDet(detStarAllPosFile, matStarFile, condFiles, pixFile, magBins, starDetAsStarTrainFiles, starDetAsStarProbFiles, starDetAsStarExtrFiles, numBins, res, True, classCutoff, binNum, cutOffPercent, doFullSky = False)

# fullSkyBool = [True, True, True]
# for i in range(len(starDetAsStarTrainFiles)):
#     if loadtxt(starDetAsStarTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
#         fullSkyBool[i] = False

# for i in range(3):
#     if fullSkyBool[i]:
#         for j in range(50):
#             fullSkyDet(pixFile, condFiles, np.array([starDetAsStarTrainFiles[i]]), np.array([starDetAsStarProbFiles[i]]), np.array([starDetAsStarExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
#     else:
#         aveDet = loadtxt(starDetAsStarTrainFiles[i][0:-5] + '_Ave_Det.csv', delimiter=',')
#         aveDet = 1 * aveDet

#         prob_table = Table()
#         prob_table['SIGNAL'] = aveDet * np.ones(len(validPix))
#         prob_table.write(starDetAsStarProbFiles[i], overwrite = True) 

#         ext_table = Table()
#         ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
#         ext_table.write(starDetAsStarExtrFiles[i], overwrite = True)
        
# # Gala det as star
# singleCorrectionTrainDet(detGalaAllPosFile, matGalaFile, condFiles, pixFile, magBins, galaDetAsStarTrainFiles, galaDetAsStarProbFiles, galaDetAsStarExtrFiles, numBins, res, True, classCutoff, binNum, cutOffPercent, doFullSky = False)

# fullSkyBool = [True, True, True]
# for i in range(len(galaDetAsStarTrainFiles)):
#     if loadtxt(galaDetAsStarTrainFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int).size == 0:
#         fullSkyBool[i] = False

# for i in range(3):
#     if fullSkyBool[i]:
#         for j in range(50):
#             fullSkyDet(pixFile, condFiles, np.array([galaDetAsStarTrainFiles[i]]), np.array([galaDetAsStarProbFiles[i]]), np.array([galaDetAsStarExtrFiles[i]]), res, numBins, startInd = startInds[j], endInd = endInds[j])
#     else:
#         aveDet = loadtxt(galaDetAsStarTrainFiles[i][0:-5] + '_Ave_Det.csv', delimiter=',')
#         aveDet = 1 * aveDet

#         prob_table = Table()
#         prob_table['SIGNAL'] = aveDet * np.ones(len(validPix))
#         prob_table.write(galaDetAsStarProbFiles[i], overwrite = True) 

#         ext_table = Table()
#         ext_table['EXTRAPOLATIONS'] = np.zeros(len(validPix))
#         ext_table.write(galaDetAsStarExtrFiles[i], overwrite = True)
        
# Calibration steps
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
    
    print(len(TSPIX))
    print(len(FGPIX))
    print(len(TGPIX))
    print(len(FSPIX))
    print(' ')
    
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