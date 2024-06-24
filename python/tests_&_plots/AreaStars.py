import sys
sys.path.insert(1, '/afs/hep.wisc.edu/home/kkboone/software/StarWeights/FinalPipeline')
import fitsio
import numpy as np
import healpy as hp
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib
from scipy import interpolate as inter
from astropy.table import Table
import StellarConfig as strConfig
import Config
# from matplotlib.path import Path Cut has already been applied
matplotlib.style.use('des_dr1')

directory = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/FinalPipeline/Tests/Object_Counts/'

maxCycles = 300
numAreas = 3
areaFile = directory + 'Three_Areas_Pixels.fits'
areaData = fitsio.read(areaFile)
areaPix = []
for i in np.arange(numAreas):
    areaPix.append(areaData[str(i)])
    
for i in np.arange(len(areaPix)):
    area512 = areaPix[i]
    fullArea512 = np.zeros(12*(512**2))
    fullArea512[area512] = 1
    fullArea4096 = hp.ud_grade(fullArea512, 4096, order_in = 'NESTED', order_out = 'NESTED')
    areaPix[i] = np.where(fullArea4096 > 0.5)[0]

def mostSigInd(y):
    maxSquaredDiff = 0
    index = -1
    
    maxSingError = np.max(np.abs(y - 1))
    
    if maxSingError <= cutOffPercent:
        return index, maxSingError
    
    for i in range(len(y)):
        yi = y[i]
        
        diff = np.sum((yi - 1)**2)
        
        if diff > maxSquaredDiff:
            maxSquaredDiff = diff
            index = i
            
    return index, maxSingError

cutOffPercent = .01
res = 4096
binNum = 10
classCut = 1.5
# path = strConfig.path Cut has already been applied
# mu = strConfig.mu
rMagCut = [23.9, 24.5]
conditions = Config.conditions

validPixFile = strConfig.detStarAllPosFile

validPix = np.unique(fitsio.read(validPixFile)['PIXEL'])

for i in np.arange(len(areaPix)):
    areaPix[i] = areaPix[i][np.isin(areaPix[i], validPix)]

oldValidPixFile = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/BalrogTests/Test1/ValidPix.fits'

oldValidPix = fitsio.read(oldValidPixFile)['PIXEL']

# Already has color and quality cuts applied.
matStarFile = strConfig.matStarFile

matStarData = fitsio.read(matStarFile)

matStarRA = matStarData['RA']
matStarDEC = matStarData['DEC']
# matStarGMAG = matStarData['GMAG'] Unnecessary, cuts are based on RMAG
matStarRMAG = matStarData['RMAG']
matStarCLASS = matStarData['CLASS']

# Naming conventions changing to match original file
matPix = hp.ang2pix(res, matStarRA, matStarDEC, nest = True, lonlat = True)

pixCut = np.isin(matPix, validPix)
matPix = matPix[pixCut]
matRmag = matStarRMAG[pixCut]
matClass = matStarCLASS[pixCut]

magCut = np.where((matRmag <= rMagCut[1]) & (matRmag > rMagCut[0]))[0]
matPix = matPix[magCut]
matClass = matClass[magCut]

classCuts = np.where((matClass >= 0) & (matClass <= classCut))[0]
matPix = matPix[classCuts]

origDetPix = np.copy(matPix)
origDetPix = np.sort(origDetPix)

origAllDetPix, origAllDetPixCounts = np.unique(np.append(validPix, origDetPix), return_counts = True)
origAllDetPixCounts = origAllDetPixCounts - 1

ave = np.average(origAllDetPixCounts)
testPix = []

for i in np.arange(numAreas):
    pixOfInterest = np.isin(origAllDetPix, areaPix[i])
    detOfInterest = origAllDetPixCounts[pixOfInterest]
    areaAve = np.average(detOfInterest)
    
    includeInds = np.full(len(detOfInterest), False, dtype = bool)
    includeInds[0:int((float(20)*len(includeInds)) / 100)] = True

    np.random.shuffle(includeInds)
    cropDetOfInterest = detOfInterest[includeInds]

    cropAve = np.average(cropDetOfInterest)

    while (np.abs(areaAve - cropAve) / areaAve) > 0.001:
        np.random.shuffle(includeInds)
        cropDetOfInterest = detOfInterest[includeInds]
        cropAve = np.average(cropDetOfInterest)
        
    testPix.append(origAllDetPix[pixOfInterest][includeInds])
    
allTestPix = []
for i in np.arange(len(testPix)):
    allTestPix.extend(testPix[i])
allTestPix = np.array(allTestPix)

aveTest = np.average(origAllDetPixCounts[np.isin(origAllDetPix, allTestPix)])

writeFile = directory + 'Stars/Three_Area_Errors/0.0_Percent_' + str(rMagCut[0]) + '_' + str(rMagCut[1]) + '.fits'

my_table = Table()

for i in np.arange(len(testPix)):
    cropDetOfInterest = origAllDetPixCounts[np.isin(origAllDetPix, testPix[i])]
#     newDetOfInterest = np.hstack((cropDetOfInterest,) * 5)
    
#     bootExcessPers = []
#     includeInds = np.full(len(newDetOfInterest), False, dtype = bool)
#     includeInds[0:int((float(20)*len(includeInds)) / 100)] = True

#     for _ in range(10000):
#         np.random.shuffle(includeInds)
#         bootExcessPers.append(np.average(newDetOfInterest[includeInds]) / aveTest)
        
#     bootExcessPers = np.array(bootExcessPers)
    
    my_table[str(i)] = np.array([np.average(cropDetOfInterest) / aveTest])#bootExcessPers

my_table.write(writeFile, overwrite = True)

origCondFiles = []
for cond in conditions:
    origCondFiles.append('/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/BalrogTests/Test1/Conds/' + cond + '.fits')
origCondFiles = np.array(origCondFiles)

origCondMaps = []
newPixCrop = np.isin(oldValidPix, validPix)

# This loops over every condition file
for condFile in origCondFiles:
    condData = fitsio.read(condFile) # This reads in the data
    origCondMaps.append(condData['SIGNAL'][newPixCrop]) # Only stores the values that are in pixels with injections

origCondMaps = np.array(origCondMaps)

persToUse = np.array([5, 10, 15, 20, 40, 60, 80, 100]) # np.linspace(100, 20, 5)

allPixFile = strConfig.detStarAllPosFile

origInjData = fitsio.read(allPixFile)

origInjPix = origInjData['PIXEL']
origValidPix = np.unique(origInjPix)
origInjPix = origInjPix[~np.isin(origInjPix, allTestPix)]

origInjPix = np.sort(origInjPix)

# Everything from here until the main loop is to generate matchInds.
# These are only necessary in the train data where crops are made.

origInjPixUnique, origInjPixCounts = np.unique(origInjPix, return_counts = True)

trainOrigDetPix = origDetPix[~np.isin(origDetPix, allTestPix)]
matchInds = np.zeros(len(trainOrigDetPix), dtype = int)

startInjInds = np.append(np.array([0]), np.cumsum(origInjPixCounts)[:-1])

trainOrigAllDetPixCounts = origAllDetPixCounts[~np.isin(origAllDetPix, allTestPix)]
# origAllDetPix counts needs this continuous crop since it wasn't cropped immediately
startDetInds = np.append(np.array([0]), np.cumsum(trainOrigAllDetPixCounts)[:-1])

for i in np.arange(len(trainOrigAllDetPixCounts)):
    if trainOrigAllDetPixCounts[i] == 0:
        continue
    matchInds[startDetInds[i]: startDetInds[i] + trainOrigAllDetPixCounts[i]] = np.arange(trainOrigAllDetPixCounts[i]).astype(int) + startInjInds[i]
    
# STILL NEEDS WORK FROM HERE ON
for perObjectsToUse in persToUse:
    
    includeInds = np.full(len(origInjPix), False, dtype = bool)
    includeInds[0:int((float(perObjectsToUse)*len(includeInds)) / 100)] = True
    np.random.shuffle(includeInds)
    
    detPix = trainOrigDetPix[includeInds[matchInds]]
    injPix = origInjPix[includeInds]
    
    detPix = np.append(detPix, origDetPix[np.isin(origDetPix, allTestPix)])
    injPix = np.append(injPix, allTestPix)

    validPix =  np.unique(injPix)
    
    condCrop = np.isin(origValidPix, validPix)
    
    detPixIndicator, origDetPixCounts = np.unique(np.append(validPix, detPix), return_counts = True)
    origDetPixCounts = origDetPixCounts - 1
    
    condMaps = []

    # This loops over every condition file
    for origCondMap in origCondMaps:
        condMaps.append(origCondMap[condCrop]) # Only stores the values that are in pixels with injections

    condMaps = np.array(condMaps)
    
    trainInds = ~np.isin(detPixIndicator, allTestPix)
    
    aveDetTrain = np.average(origDetPixCounts[trainInds])

    sortInds = []
    for i in range(len(condMaps)):
        sortInds.append(condMaps[i][trainInds].argsort())
    sortInds = np.array(sortInds)
    
    binIndLims = [0]

    for j in range(binNum):
        binIndLims.append(int((np.sum(trainInds) - binIndLims[-1]) / (binNum - j)) + (binIndLims[-1]))
        
    xBins = []

    for i in range(len(condMaps)):
        cond_Map_Sort = condMaps[i][trainInds][sortInds[i][::1]]
        condBins = []
        for j in range(binNum):
            condBins.append(cond_Map_Sort[binIndLims[j]:binIndLims[j+1]])
        indXBin = []

        for j in range(binNum):
            indXBin.append(np.average(condBins[j]))

        xBins.append(np.array(indXBin))

    xBins = np.array(xBins)
    
    yBinsOrig = []
    for i in range(len(condMaps)):
        detSort = origDetPixCounts[trainInds][sortInds[i][::1]]
        detBins = []
        for j in range(binNum):
            detBins.append(detSort[binIndLims[j]:binIndLims[j+1]])
        indYBinOrig = []

        for j in range(binNum):
            indYBinOrig.append(np.average(detBins[j]) / aveDetTrain)

        yBinsOrig.append(np.array(indYBinOrig))

    yBinsOrig = np.array(yBinsOrig)
    
    detPixCounts = np.copy(origDetPixCounts)
    
    allErrors = []
    
    numCycles = 0

    while(True):
        
        if numCycles >= maxCycles:
            break

        yBins = []
        for i in range(len(condMaps)):
            detSort = detPixCounts[trainInds][sortInds[i][::1]]
            detBins = []
            for j in range(binNum):
                detBins.append(detSort[binIndLims[j]:binIndLims[j+1]])
            indYBin = []

            for j in range(binNum):
                indYBin.append(np.average(detBins[j]) / aveDetTrain)

            yBins.append(np.array(indYBin))

        yBins = np.array(yBins)

        index, maxErr = mostSigInd(yBins)
        if index == -1:
            break

        allErrors.append(maxErr)

        corrFunc = inter.interp1d(xBins[index], yBins[index], bounds_error = False, fill_value = (yBins[index][0], yBins[index][-1]))

        detPixCounts = detPixCounts / (corrFunc(condMaps[index]))

        detPixCounts = detPixCounts * aveDetTrain / (np.average(detPixCounts[trainInds]))
        
        numCycles += 1
    
    aveDetTest = np.average(detPixCounts[~trainInds])
    
    writeFile = directory + 'Stars/Three_Area_Errors/' + str(perObjectsToUse) + '_Percent_' + str(rMagCut[0]) + '_' + str(rMagCut[1]) + '.fits'
    my_table = Table()
    
    for i in np.arange(len(testPix)):
        
        cropDetOfInterest = detPixCounts[np.isin(detPixIndicator, testPix[i])]
#         newDetOfInterest = np.hstack((cropDetOfInterest,) * 5)

#         bootExcessPers = []
#         includeInds = np.full(len(newDetOfInterest), False, dtype = bool)
#         includeInds[0:int((float(20)*len(includeInds)) / 100)] = True

#         for _ in range(10000):
#             np.random.shuffle(includeInds)
#             bootExcessPers.append(np.average(newDetOfInterest[includeInds]) / aveDetTest)

#         bootExcessPers = np.array(bootExcessPers)

        my_table[str(i)] = np.array([np.average(cropDetOfInterest) / aveDetTest])#bootExcessPers
        
    my_table.write(writeFile, overwrite = True)