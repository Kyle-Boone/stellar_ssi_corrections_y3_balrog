'''
This file contains three methods related to finding classification probabilities as a function of survey properties and then applying this function to all valid pixels.
The first method "mostSigInd" is used to figure out which survey property has the largest impact on classification rates throughout training.
The second method "singleCorrectionTrain" is used to find classification probabilities as a function of survey properties and call the final method "fullSky" to apply this function to all valid pixels.
'''

import numpy as np
from numpy import savetxt
from numpy import loadtxt
import fitsio
import healpy as hp
import Config
from scipy import interpolate as inter
from astropy.table import Table


def mostSigInd(y, cutOffPercent):
    '''
    This should not be called directly by the user.
    It is used within "singleCorrectionTrain" to find which survey property effect to train for next.
    '''
    
    maxSquaredDiff = 0
    index = -1
    
    maxSingError = np.max(np.abs(y - 1))
    
    # If all values are within the cutOffPercent, a default value is returned.
    if maxSingError <= cutOffPercent:
        return index
    
    # Otherwise, a least squares error determines which survey property to correct for next.
    for i in range(len(y)):
        yi = y[i]
        
        diff = np.sum((yi - 1)**2)
        
        if diff > maxSquaredDiff:
            maxSquaredDiff = diff
            index = i
            
    return index


def singleCorrectionTrain(objectFile, condFiles, pixFile, magBins, trainDataFiles, fullSkyProbFiles, extrFiles, numBins, res, isStar, classCut, binNum, cutOffPercent, doFullSky = True, indLenLim = -1):
    '''
    This function attempts to find correct classification probabilities as a function of survey properties.
    All files sent to this function should be .fits files.
    objectFile contains information on the detected objects.
    condFiles is a list of survey property files.
    pixFile contains the valid pixels.
    magBins is a list of the magnitude bins.
    The number of magnitude bins is one less than the length of magBins.
    trainDataFiles is where all the information necessary for performing the correction is stored (should be the length of the number of magnitude bins passed in).
    fullSkyProbFiles is where the actual correct classification probabilities will be stored (should be the length of the number of magnitude bins passed in).
    extrFiles is where counts on the number of extrapolations per pixel will be stored (should be the length of the number of magnitude bins passed in).
    numBins is an integer used for applying the function to the full sky.
    The full sky will be split binNum many times to limit how much information is read in at once.
    Larger numbers will be easier on a computer but marginally slower.
    res is the healpixel resolution.
    isStar is a boolean corresponding to whether we are training stars (True) or galaxies (False).
    classCut is the class cutoff used to distinguish classified stars and galaxies.
    binNum is the number of data points used to calculate out interpolation functions for each survey property.
    cutOffPercent is an accuracy threshold to determine when training should stop.
    doFullSky is a boolean deciding whether or not to do the full sky application.
    In my runs, this led to some issues with memory so I often would set it to false and just directly call "doFullSky" after training.
    indLenLim, if a different value is passed in, will limit the number of cycles that can be done in training even if we don't converge to the desired accuracy.
    '''
    
    # Classes corresponding to a correct classification.
    if isStar:
        lowClass = 0
        highClass = classCut
    else:
        lowClass = classCut
        highClass = 3
        
    # Getting valid pixels.
    validPix = fitsio.read(pixFile)['PIXEL']
    
    # Read object data.
    objectData = fitsio.read(objectFile)
    
    origRA = objectData['RA']
    origDEC = objectData['DEC']
    RMAG = objectData['RMAG']
    EXTENDED_CLASS = objectData['CLASS']
    
    # Switching to a 0 or 1 class indicator
    origCLASS = np.zeros_like(EXTENDED_CLASS)
    origCLASS[np.where((EXTENDED_CLASS >= lowClass) & (EXTENDED_CLASS <= highClass))[0]] = 1
    
    for i in range(len(magBins) - 1):
        
        # This defines magnitude cuts in accordance with the magnitude bins.
        minRMAG = magBins[i]
        maxRMAG = magBins[i + 1]
        magCut = np.where(((RMAG <= maxRMAG) & (RMAG > minRMAG)))[0]
            
        RA = origRA[magCut]
        DEC = origDEC[magCut]
        CLASSRepeats = origCLASS[magCut]
        
        # Converting to healpixels for all objects.
        PIXRepeats = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)
        
        # Healpixels of correctly classified objects.
        CORPIXRepeats = PIXRepeats[np.where(CLASSRepeats == 1)[0]]
        
        # PIX signifies the pixels, TOT the total number of detections, and COR the number of correct classifications.
        PIX, TOT = np.unique(PIXRepeats, return_counts = True)
        _, COR = np.unique(np.append(PIX, CORPIXRepeats), return_counts = True)
        COR = COR - 1
        
        # Read in survey properties on PIX.
        condMaps = []

        for condFile in condFiles:
            condData = fitsio.read(condFile)
            condSigExt = np.full(12*(res**2), -1.6375e+30)
            condSigExt[validPix] = condData['SIGNAL']
            condMaps.append(condSigExt[PIX])

        condMaps = np.array(condMaps, dtype = object)
        
        # Average correct classification probability.
        aveAcc = np.sum(COR) / np.sum(TOT)

        # Relative classification rates for each cycle of training.
        yValues = []
        
        # Indices of the survey properties used for each cycle of training.
        corrIndices = []
        
        # Indices to sort each survey property map.
        sortInds = []
        for j in range(len(condMaps)):
            sortInds.append(condMaps[j].argsort())
        sortInds = np.array(sortInds)

        # This is used to split the pixels up into binNum partitions.
        binIndLims = [0]

        for j in range(binNum):
            binIndLims.append(int((len(condMaps[0]) - binIndLims[-1]) / (binNum - j)) + (binIndLims[-1]))

        # Average value of survey property among 10 bins for each survey property.
        xBins = []

        for j in range(len(condMaps)):
            cond_Map_Sort = condMaps[j][sortInds[j][::1]]
            condBins = []
            for k in range(binNum):
                condBins.append(cond_Map_Sort[binIndLims[k]:binIndLims[k+1]])
            indXBin = []

            for k in range(binNum):
                indXBin.append(np.sum(condBins[k]) / len(condBins[k]))

            xBins.append(np.array(indXBin))

        xBins = np.array(xBins)
        
        # Training the correct classification probability function.
        while(True):
            
            # Break loop if we exceed the limit of cycles.
            if indLenLim >= 0:
                if len(corrIndices) >= indLenLim:
                    break

            # Calculating relative classification rates for each survey property bin.
            yBins = []
            for j in range(len(condMaps)):
                corSort = COR[sortInds[j][::1]]
                totSort = TOT[sortInds[j][::1]]
                corBins = []
                totBins = []
                for k in range(binNum):
                    corBins.append(corSort[binIndLims[k]:binIndLims[k+1]])
                    totBins.append(totSort[binIndLims[k]:binIndLims[k+1]])
                indYBin = []

                for k in range(binNum):
                    indYBin.append((np.sum(corBins[k]) / np.sum(totBins[k])) / aveAcc)

                yBins.append(np.array(indYBin))

            yBins = np.array(yBins)

            # Determine which survey property has the largest impact on relative classification rates.
            index = mostSigInd(yBins, cutOffPercent)
            
            # Break if we've hit accuracy threshold, otherwise store relative classification rate information.
            if index == -1:
                break
            else:
                corrIndices.append(index)
                yValues.append(yBins[index])

            # Interpolation function for relative classification rate as a function of the chosen survey property.
            corrFunc = inter.interp1d(xBins[index], yBins[index], bounds_error = False, fill_value = (yBins[index][0], yBins[index][-1]))

            # Correct the COR count to train for this chosen survey property.
            COR = COR / (corrFunc(condMaps[index].astype('float')))

            # Calibrate accuracy.
            COR = COR * aveAcc / (np.sum(COR) / np.sum(TOT))

        # New indices used for easier storage.
        storeCorrIndices = []
        
        numConds = len(condMaps)
        
        for j in range(len(corrIndices)):
            storeCorrIndices.append(corrIndices[j] + numConds*j)

        storeCorrIndices = np.array(storeCorrIndices)

        # Store indices and average accuracy.
        savetxt(trainDataFiles[i][0:-5] + '_Indices.csv', storeCorrIndices, delimiter=',')
        savetxt(trainDataFiles[i][0:-5] + '_Ave_Acc.csv', np.array([aveAcc]), delimiter=',')
        
        # If necessary, the fits file for relative classification rates at each step of the training cycle is split up into multiple fits files. 
        # This won't happen with cycle limits.
        if len(corrIndices) > 999:
            j = 0
            for fits_file_count in np.arange(int(np.ceil(len(corrIndices) / 999))):
                y_table = Table()
                
                max_j = np.min([j + 999, len(storeCorrIndices)]) # maximum for this file
                
                while j < max_j:
                    y_table[str(storeCorrIndices[j])] = yValues[j]
                    j += 1
                    
                y_table.write(trainDataFiles[i][0:-5] + '_' + str(fits_file_count) + '.fits', overwrite = True)
        else:
            y_table = Table()
            for j in np.arange(len(storeCorrIndices)):
                y_table[str(storeCorrIndices[j])] = yValues[j]
            y_table.write(trainDataFiles[i], overwrite = True)

        # Save survey property bins.
        x_table = Table()
        for j in np.arange(len(xBins)):
            x_table[str(j)] = xBins[j]
        x_table.write(trainDataFiles[i][0:-5] + '_X_Values.fits', overwrite = True)
        
    # Extend this to the full sky.
    if doFullSky:
        fullSky(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins)
    
    
def fullSky(pixFile, condFiles, trainDataFiles, fullSkyProbFiles, extrFiles, res, numBins, startInd = 0, endInd = -1, showProgress = False):
    '''
    This function finds correct classification probabilities across all valid pixels.
    It is critical that the above function is run before this one.
    All files sent to this function should be .fits files.
    pixFile contains the valid pixels.
    condFiles is a list of survey property files.
    trainDataFiles is where all the information necessary for performing the correction is stored.
    fullSkyProbFiles is where the actual correct classification probabilities will be stored.
    extrFiles is where counts on the number of extrapolations per pixel will be stored.
    res is the healpixel resolution.
    
    All remaining arguments do not change the functionality of the code but simply help it run.
    
    numBins is an integer used for applying the function to the full sky.
    The full sky will be split binNum many times to limit how much information is read in at once.
    Larger numbers will be easier on a computer but marginally slower.
    
    startInd and endInd correspond to numBins by limiting how many of the bins are calculated in any method call.
    If numBins was 100, startInd was 2, and endInd was 4, of the 100 spatial bins of the full sky, only bins 2-4 would be calculted in the run.
    If default values are used, every bin will be calculated.
    
    showProgress will print the bin number before it is calculated to check on progress and get quick estimates on time demands.
    '''
    
    if endInd == -1:
        endInd = numBins - 1 # Actual default.
    
    # Read in valid healpixels.
    validPix = fitsio.read(pixFile)['PIXEL']
    
    # Loops over the magnitude bins sent in implicitly through file list length.
    for i in range(len(trainDataFiles)):
        # This reads in the training data.
        xValues = fitsio.read(trainDataFiles[i][0:-5] + '_X_Values.fits')
        aveAcc = loadtxt(trainDataFiles[i][0:-5] + '_Ave_Acc.csv', delimiter=',')
        # print(aveAcc) # This was used to ensure the code has made it to this point.
        indices = loadtxt(trainDataFiles[i][0:-5] + '_Indices.csv', delimiter=',').astype(int)
        
        # Reformatting indices so that the following will work for any size of indices.
        # If the size is zero, workarounds are used instead of this method.
        if indices.size == 1:
            indices = np.array([indices[()]])
        
        # Reading in relative classification rates.
        if len(indices) <= 999:
            yValues = fitsio.read(trainDataFiles[i])
        else:
            yValues = []
            for fits_file_count in np.arange(int(np.ceil(len(indices) / 999))):
                yValues.append(fitsio.read(trainDataFiles[i][0:-5] + '_' + str(fits_file_count) + '.fits'))
        
        # Getting spatial bins to split the valid pixels.
        binLims = [0]
        for j in range(numBins):
            binLims.append(int((len(validPix) - binLims[-1]) / (numBins - j)) + (binLims[-1]))
            
        # This will be used to record extrapolations.
        extrMap = []
        for j in range(len(indices)):
            extrMap.append([])
        
        # This will be the probabilities map.
        probMap = []

        for j in range(len(binLims) - 1):
            if j < startInd:
                continue
            if j > endInd:
                continue
            if showProgress:
                print(j)

            condMaps = []
            for condFile in condFiles:
                condMaps.append(fitsio.read(condFile, rows = np.arange(binLims[j], binLims[j + 1]))['SIGNAL'])
            condMaps = np.array(condMaps, dtype = object)
            
            numConds = len(condMaps)

            sectProb = np.ones(len(condMaps[0]))
            # This generates the corrections for this specific section of pixels read in.

            for k in range(len(indices)):

                # Will be used to mark all areas where extrapolations were performed.
                extrapolation = np.zeros(len(condMaps[indices[k] % numConds]))
                
                # Get average survey property and relative classification rate bins.
                x = xValues[str(indices[k] % numConds)]
                
                if len(indices) <= 999:
                    y = yValues[str(indices[k])]
                else:
                    y = yValues[int(k / 999)][str(indices[k])]
                
                extrapolation[np.where((condMaps[indices[k] % numConds] > x[-1]) | (condMaps[indices[k] % numConds] < x[0]))[0]] = 1

                # Generates the function via interpolation
                f = inter.interp1d(x, y, bounds_error = False, fill_value = (y[0], y[-1]))

                # Generates the relative probability
                corr = f(condMaps[indices[k] % numConds].astype('float'))

                # Applies the probability to this section.
                sectProb = sectProb * corr

                # For this index, expands out the extrapolation map.
                extrMap[k].extend(extrapolation)

            # Multiplies the relative probability by the average probability.
            probMap.extend(sectProb * aveAcc)

        probMap = np.array(probMap)
        
        # This collapses all of the extrapolations down into one map.
        fullExtrMap = np.zeros_like(probMap)
        for j in range(len(extrMap)):
            fullExtrMap = fullExtrMap + np.array((extrMap[j]))
            
        if startInd > 0:     
            oldProbs = fitsio.read(fullSkyProbFiles[i])['SIGNAL']
            oldExtrs = fitsio.read(extrFiles[i])['EXTRAPOLATIONS']

            probMap = np.append(oldProbs, probMap)
            fullExtrMap = np.append(oldExtrs, fullExtrMap)
        
        # This stores the probabilities and extrapolations.
        my_table = Table()
        my_table['SIGNAL'] = probMap
        my_table.write(fullSkyProbFiles[i], overwrite = True) 
        
        ext_table = Table()
        ext_table['EXTRAPOLATIONS'] = fullExtrMap
        ext_table.write(extrFiles[i], overwrite = True)
