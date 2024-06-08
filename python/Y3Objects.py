'''
This file contains a single method, "getY3Objects" which is designed to get objects for the purposes of original object counts and calibrations to the deep fields.
Adjustments to this will be necessary if different styles of cuts are desired.
Currently I perform a maximum g magnitude cut, an isochrone cut in gr color vs absolute g magnitude space, flag cuts, and r magnitude bins.
'''

import numpy as np
import fitsio
import astropy.io.fits as fits
from astropy.table import Table
import healpy as hp
from matplotlib.path import Path

def getY3Objects(pixFile, goldObjectsDir, goldObjectsFiles, goldCols, goldMoreInfoStarFiles, goldMoreInfoGalaFiles, goldStarFiles, goldGalaFiles, res, magBins, numMagBins, classCutoff, gCut, path, mu):
    '''
    This gets Y3 Gold object properties into a more concise way for quicker calibrations and final corrections.
    pixFile is where the valid pixels are stored.
    goldObjectsDir is the directory where original Y3 gold data is stored.
    goldObjectsFiles is a list of the files within goldObjectsDir.
    goldCols are the columns I read in from the goldObjectsFiles.
    goldMoreInfoStar/GalaFiles are lists of files which will contain g and r magnitudes as well as RA and DEC values for all objects that pass our cuts.
    goldStar/GalaFiles are lists of files which will contain the number of stars and galaxies that pass our cuts on each of the valid pixels.
    For the two above file types, the length of the lists of files should match the number of magnitude bins.
    res is the full resolution of calculations.
    magBins is a list of magnitude bins in order (ex: [0, 22, 23, 24]).
    numMagBins is the number of magnitude bins (for the above example it would be 3).
    classCutoff is the class cut used to distinguish stars from galaxies.
    gCut is a maximum g magnitude used for cuts.
    path and mu are used to perform an isochrone cut.
    '''
    
    # Read in valid pixels
    validPix = fitsio.read(pixFile)['PIXEL']
    pixCheck = np.full(12*(res**2), False, dtype = bool)
    pixCheck[validPix] = True
    
    StarPIX = []
    GalaPIX = []

    StarRA = []
    StarDEC = []

    GalaRA = []
    GalaDEC = []

    StarRMAG = []
    StarGMAG = []

    GalaRMAG = []
    GalaGMAG = []

    # Each magnitude bin is stored separately
    for _ in range(numMagBins):

        StarPIX.append(np.array([]))
        StarRA.append(np.array([]))
        StarDEC.append(np.array([]))
        StarRMAG.append(np.array([]))
        StarGMAG.append(np.array([]))

        GalaPIX.append(np.array([]))
        GalaRA.append(np.array([]))
        GalaDEC.append(np.array([]))
        GalaRMAG.append(np.array([]))
        GalaGMAG.append(np.array([]))

    for j in range(len(goldObjectsFiles)):
        if j%100 == 0:
            print(j)
        # Read in real gold data
        obsData = fitsio.read(goldObjectsDir + goldObjectsFiles[j], columns = goldCols)
        FOREGROUND = obsData[goldCols[0]]
        BADREGIONS = obsData[goldCols[1]]
        FOOTPRINT = obsData[goldCols[2]]
        CLASS = obsData[goldCols[3]]
        GMAG = obsData[goldCols[4]]
        RMAG = obsData[goldCols[5]]
        GMAG_GALA = obsData[goldCols[6]]
        RMAG_GALA = obsData[goldCols[7]]
        RA = obsData[goldCols[8]]
        DEC = obsData[goldCols[9]]
        PIX = hp.ang2pix(res, RA, DEC, lonlat = True, nest = True)

        # General Quality Cuts
        qualityCut = np.where((FOREGROUND == 0) &
                              (BADREGIONS < 2) &
                              (FOOTPRINT == 1) &
                              (pixCheck[PIX]))[0]

        CLASS = CLASS[qualityCut]
        GMAG = GMAG[qualityCut]
        RMAG = RMAG[qualityCut]
        GMAG_GALA = GMAG_GALA[qualityCut]
        RMAG_GALA = RMAG_GALA[qualityCut]
        PIX = PIX[qualityCut]
        RA = RA[qualityCut]
        DEC = DEC[qualityCut]

        # Classified as Star Objects

        blueStarCut = np.where((CLASS <= classCutoff) & 
                          (CLASS >= 0) &
                          (GMAG < gCut))[0]

        STARRA = RA[blueStarCut]
        STARDEC = DEC[blueStarCut]
        STARPIX = PIX[blueStarCut]
        STARGMAG = GMAG[blueStarCut]
        STARRMAG = RMAG[blueStarCut]

        for i in np.arange(numMagBins):
            minRMAG = magBins[i]
            maxRMAG = magBins[i + 1]
            magCut = np.where((STARRMAG <= maxRMAG) & (STARRMAG > minRMAG))[0]

            magStarRa = STARRA[magCut]
            magStarDec = STARDEC[magCut]
            magStarGmag = STARGMAG[magCut]
            magStarRmag = STARRMAG[magCut]
            magStarPix = STARPIX[magCut]

            isoStarCut = Path.contains_points(path,np.vstack((magStarGmag - magStarRmag, magStarGmag - mu)).T)

            StarRA[i] = np.append(StarRA[i], magStarRa[isoStarCut])
            StarDEC[i] = np.append(StarDEC[i], magStarDec[isoStarCut])
            StarPIX[i] = np.append(StarPIX[i], magStarPix[isoStarCut])
            StarRMAG[i] = np.append(StarRMAG[i], magStarRmag[isoStarCut])
            StarGMAG[i] = np.append(StarGMAG[i], magStarGmag[isoStarCut])

        # Classified as Galaxy Objects

        blueGalaCut = np.where((CLASS <= 3) & 
                          (CLASS >= classCutoff) &
                          (GMAG_GALA < gCut))[0]

        GALARA = RA[blueGalaCut]
        GALADEC = DEC[blueGalaCut]
        GALAPIX = PIX[blueGalaCut]
        GALAGMAG = GMAG_GALA[blueGalaCut]
        GALARMAG = RMAG_GALA[blueGalaCut]

        for i in np.arange(numMagBins):

            minRMAG = magBins[i]
            maxRMAG = magBins[i + 1]
            magCut = np.where((GALARMAG <= maxRMAG) & (GALARMAG > minRMAG))[0]

            magGalaRa = GALARA[magCut]
            magGalaDec = GALADEC[magCut]
            magGalaGmag = GALAGMAG[magCut]
            magGalaRmag = GALARMAG[magCut]
            magGalaPix = GALAPIX[magCut]

            isoGalaCut = Path.contains_points(path,np.vstack((magGalaGmag - magGalaRmag, magGalaGmag - mu)).T)

            GalaRA[i] = np.append(GalaRA[i], magGalaRa[isoGalaCut])
            GalaDEC[i] = np.append(GalaDEC[i], magGalaDec[isoGalaCut])
            GalaPIX[i] = np.append(GalaPIX[i], magGalaPix[isoGalaCut])
            GalaRMAG[i] = np.append(GalaRMAG[i], magGalaRmag[isoGalaCut])
            GalaGMAG[i] = np.append(GalaGMAG[i], magGalaGmag[isoGalaCut])

    # Storing with more information
    for i in range(numMagBins):
        my_table = Table()
        my_table['RA'] = StarRA[i].astype(float)
        my_table['DEC'] = StarDEC[i].astype(float)
        my_table['RMAG'] = StarRMAG[i].astype(float)
        my_table['GMAG'] = StarGMAG[i].astype(float)
        my_table.write(goldMoreInfoStarFiles[i], overwrite = True)

        my_table = Table()
        my_table['RA'] = GalaRA[i].astype(float)
        my_table['DEC'] = GalaDEC[i].astype(float)
        my_table['RMAG'] = GalaRMAG[i].astype(float)
        my_table['GMAG'] = GalaGMAG[i].astype(float)
        my_table.write(goldMoreInfoGalaFiles[i], overwrite = True)
        
    for j in np.arange(len(StarPIX)):

        starPixRepeats = StarPIX[j]

        # This will be used to store the number of stars at each pixel.
        starPix, starDet = np.unique(starPixRepeats, return_counts = True) # The unique pixels, with no repeats.

        fullSkyStars = np.full(12*(res**2), 0.0)
        fullSkyStars[starPix.astype(int)] = starDet

        my_table = Table()
        my_table['SIGNAL'] = fullSkyStars[validPix]
        my_table.write(goldStarFiles[j], overwrite = True)
        
    for j in np.arange(len(GalaPIX)):

        galaPixRepeats = GalaPIX[j]

        # This will be used to store the number of stars at each pixel.
        galaPix, galaDet = np.unique(galaPixRepeats, return_counts = True) # The unique pixels, with no repeats.

        fullSkyGalas = np.full(12*(res**2), 0.0)
        fullSkyGalas[galaPix.astype(int)] = galaDet

        my_table = Table()
        my_table['SIGNAL'] = fullSkyGalas[validPix]
        my_table.write(goldGalaFiles[j], overwrite = True)