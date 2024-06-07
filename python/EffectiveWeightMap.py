import numpy as np
import fitsio
import astropy.io.fits as fits
from astropy.table import Table
import healpy as hp
import StellarConfig as strConfig

# Where final correction info will be stored
starCorrectionFile = strConfig.starCorrectionFile
galaCorrectionFile = strConfig.galaCorrectionFile

# Hyperparameters
res = strConfig.res
nsideCourse = strConfig.nsideCourse
numMagBins = strConfig.numMagBins

# Valid pixels
pixFile = strConfig.pixFile

validPix = fitsio.read(pixFile)['PIXEL']
pixCheck = np.full(12*(res**2), False, dtype = bool)
pixCheck[validPix] = True

# Star probabilities
starProbFiles = strConfig.starProbFiles
starDetAsStarProbFiles = strConfig.starDetAsStarProbFiles
starDetAsGalaProbFiles = strConfig.starDetAsGalaProbFiles
    
# Galaxy probabilities
galaProbFiles = strConfig.galaProbFiles
galaDetAsStarProbFiles = strConfig.galaDetAsStarProbFiles
galaDetAsGalaProbFiles = strConfig.galaDetAsGalaProbFiles

# FracDet file
fracFile = Config.fracFile

# Calibration information
calibrationFile = strConfig.calibrationFile
    
caliData = fitsio.read(calibrationFile)
starAdjustments = caliData['STAR']
galaAdjustments = caliData['GALA']

# Y3 Object counts
goldStarFiles = strConfig.goldStarFiles
goldGalaFiles = strConfig.goldGalaFiles

claStar = []
    
for goldStarFile in goldStarFiles:
    claStar.append(fitsio.read(goldStarFile)['SIGNAL'])
    
claStar = np.array(claStar, dtype = object)

claGala = []

for goldGalaFile in goldGalaFiles:
    claGala.append(fitsio.read(goldGalaFile)['SIGNAL'])
    
claGala = np.array(claGala, dtype = object)

# Reading in probabilities
starCorrProb = []
for i in range(len(goldStarFiles)):
    starCorrProb.append(np.clip(starAdjustments[i] * fitsio.read(starProbFiles[i])['SIGNAL'], 0, 1))
starCorrProb = np.array(starCorrProb, dtype = object)

starDetAsStarProb = []
for i in range(len(goldStarFiles)):
    nextProb = fitsio.read(starDetAsStarProbFiles[i])['SIGNAL']
    nextProb[np.where(nextProb < 0)[0]] = 0
    starDetAsStarProb.append(nextProb)
starDetAsStarProb = np.array(starDetAsStarProb, dtype = object)

starDetAsGalaProb = []
for i in range(len(goldStarFiles)):
    nextProb = fitsio.read(starDetAsGalaProbFiles[i])['SIGNAL']
    nextProb[np.where(nextProb < 0)[0]] = 0
    starDetAsGalaProb.append(nextProb)
starDetAsGalaProb = np.array(starDetAsGalaProb, dtype = object)

galaCorrProb = []
for i in range(len(goldGalaFiles)):
    galaCorrProb.append(np.clip(galaAdjustments[i] * fitsio.read(galaProbFiles[i])['SIGNAL'], 0, 1))
galaCorrProb = np.array(galaCorrProb, dtype = object)

galaDetAsStarProb = []
for i in range(len(goldGalaFiles)):
    nextProb = fitsio.read(galaDetAsStarProbFiles[i])['SIGNAL']
    nextProb[np.where(nextProb < 0)[0]] = 0
    galaDetAsStarProb.append(nextProb)
galaDetAsStarProb = np.array(galaDetAsStarProb, dtype = object)

galaDetAsGalaProb = []
for i in range(len(goldGalaFiles)):
    nextProb = fitsio.read(galaDetAsGalaProbFiles[i])['SIGNAL']
    nextProb[np.where(nextProb < 0)[0]] = 0
    galaDetAsGalaProb.append(nextProb)
galaDetAsGalaProb = np.array(galaDetAsGalaProb, dtype = object)

# This generates the fracDet data.
fracData = fitsio.read(fracFile)

# This degrades it to a lower resolution and applies a cut to where there is at least 50% coverage.
fracPix = fracData['PIXEL']
fracDet = fracData['SIGNAL']
origFracMap = np.full(12*(4096**2), 0.0)
origFracMap[fracPix] = fracDet
origFracMap[~pixCheck] = 0.0 # If we aren't looking at the pixel, effective coverage of 0%
fracMap = hp.ud_grade(origFracMap, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
fracPix = np.where(fracMap >= 0.5)[0]

# Degrading counts and probabilities
deClaStar = []

for i in range(len(claStar)):
    
    fullClaStar = np.zeros(12*(res**2))
    fullClaStar[validPix] = claStar[i]
    deClaStarInd = hp.ud_grade(fullClaStar, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    
    deClaStar.append((deClaStarInd[fracPix] / fracMap[fracPix]) * ((res / nsideCourse)**2))
    
deClaStar = np.array(deClaStar, dtype = object)

deClaGala = []

for i in range(len(claGala)):
    
    fullClaGala = np.zeros(12*(res**2))
    fullClaGala[validPix] = claGala[i]
    deClaGalaInd = hp.ud_grade(fullClaGala, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    
    deClaGala.append((deClaGalaInd[fracPix] / fracMap[fracPix]) * ((res / nsideCourse)**2))
    
deClaGala = np.array(deClaGala, dtype = object)

deStarDetAsStarProb = []
for i in range(len(starDetAsStarProb)):
    
    fullProb = np.zeros(12*(res**2))
    fullProb[validPix] = starDetAsStarProb[i] * origFracMap[validPix]
    
    deFullProb = hp.ud_grade(fullProb, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    deStarDetAsStarProb.append(deFullProb[fracPix] / fracMap[fracPix])
    
deStarDetAsStarProb = np.array(deStarDetAsStarProb, dtype = object)

deStarDetAsGalaProb = []
for i in range(len(starDetAsGalaProb)):
    
    fullProb = np.zeros(12*(res**2))
    fullProb[validPix] = starDetAsGalaProb[i] * origFracMap[validPix]
    
    deFullProb = hp.ud_grade(fullProb, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    deStarDetAsGalaProb.append(deFullProb[fracPix] / fracMap[fracPix])
    
deStarDetAsGalaProb = np.array(deStarDetAsGalaProb, dtype = object)

deStarCorrProb = []
for i in range(len(starCorrProb)):
    
    fullProb = np.zeros(12*(res**2))
    fullProb[validPix] = starCorrProb[i] * origFracMap[validPix]
    
    deFullProb = hp.ud_grade(fullProb, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    deStarCorrProb.append(deFullProb[fracPix] / fracMap[fracPix])
    
deStarCorrProb = np.array(deStarCorrProb, dtype = object)

deGalaDetAsStarProb = []
for i in range(len(galaDetAsStarProb)):
    
    fullProb = np.zeros(12*(res**2))
    fullProb[validPix] = galaDetAsStarProb[i] * origFracMap[validPix]
    
    deFullProb = hp.ud_grade(fullProb, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    deGalaDetAsStarProb.append(deFullProb[fracPix] / fracMap[fracPix])
    
deGalaDetAsStarProb = np.array(deGalaDetAsStarProb, dtype = object)

deGalaDetAsGalaProb = []
for i in range(len(galaDetAsGalaProb)):
    
    fullProb = np.zeros(12*(res**2))
    fullProb[validPix] = galaDetAsGalaProb[i] * origFracMap[validPix]
    
    deFullProb = hp.ud_grade(fullProb, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    deGalaDetAsGalaProb.append(deFullProb[fracPix] / fracMap[fracPix])
    
deGalaDetAsGalaProb = np.array(deGalaDetAsGalaProb, dtype = object)

deGalaCorrProb = []
for i in range(len(galaCorrProb)):
    
    fullProb = np.zeros(12*(res**2))
    fullProb[validPix] = galaCorrProb[i] * origFracMap[validPix]
    
    deFullProb = hp.ud_grade(fullProb, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    deGalaCorrProb.append(deFullProb[fracPix] / fracMap[fracPix])
    
deGalaCorrProb = np.array(deGalaCorrProb, dtype = object)

# Original counts
origStar = np.full(len(fracPix), 0.0)
for i in np.arange(numMagBins):
    origStar = origStar + deClaStar[i]
    
origGala = np.full(len(fracPix), 0.0)
for i in np.arange(numMagBins):
    origGala = origGala + deClaGala[i]
    
# Correction algorithm
corrStarBins = []
corrGalaBins = []
for i in np.arange(numMagBins):
    
    obsStars = (((deGalaCorrProb[i] * deClaStar[i]) + ((deGalaCorrProb[i] - 1) * deClaGala[i])) / ((deStarCorrProb[i] + deGalaCorrProb[i] - 1))).astype(float)
    obsStars[np.where(obsStars < 0)] = 0
    obsStars[np.where(obsStars >= deClaStar[i] + deClaGala[i])] = deClaStar[i][np.where(obsStars >= deClaStar[i] + deClaGala[i])] + deClaGala[i][np.where(obsStars >= deClaStar[i] + deClaGala[i])]
    
    obsGalas = deClaStar[i] + deClaGala[i] - obsStars
    
    CsfOs = obsStars * deStarCorrProb[i]
    CsfOg = obsGalas * (1 - deGalaCorrProb[i])
    
    CgfOs = obsStars * (1 - deStarCorrProb[i])
    CgfOg = obsGalas * deGalaCorrProb[i]
    
    CsfOsCorr = CsfOs / (deStarDetAsStarProb[i].astype(float))
    CsfOgCorr = CsfOg / (deGalaDetAsStarProb[i].astype(float))
    
    CgfOsCorr = CgfOs / (deStarDetAsGalaProb[i].astype(float))
    CgfOgCorr = CgfOg / (deGalaDetAsGalaProb[i].astype(float))
    
    corrStarBins.append(CsfOsCorr + CsfOgCorr)
    corrGalaBins.append(CgfOsCorr + CgfOgCorr)
    
corrStar = np.sum(corrStarBins, axis = 0)
corrGala = np.sum(corrGalaBins, axis = 0)

# Reordering these counts into a different format
fullOrigStar = np.full(12*(nsideCourse**2), hp.UNSEEN)
fullOrigStar[fracPix] = origStar
fullOrigStar[np.where(fullOrigStar <= 0)[0]] = hp.UNSEEN

fullCorrStar = np.full(12*(nsideCourse**2), hp.UNSEEN)
fullCorrStar[fracPix] = corrStar
fullCorrStar[np.where(fullOrigStar <= 0)[0]] = hp.UNSEEN

starRatio = np.full(12*(nsideCourse**2), hp.UNSEEN)
starRatio[np.where(fullOrigStar > 0)[0]] = fullCorrStar[np.where(fullOrigStar > 0)[0]] / fullOrigStar[np.where(fullOrigStar > 0)[0]]

fullOrigGala = np.full(12*(nsideCourse**2), hp.UNSEEN)
fullOrigGala[fracPix] = origGala
fullOrigGala[np.where(fullOrigGala <= 0)[0]] = hp.UNSEEN

fullCorrGala = np.full(12*(nsideCourse**2), hp.UNSEEN)
fullCorrGala[fracPix] = corrGala
fullCorrGala[np.where(fullOrigGala <= 0)[0]] = hp.UNSEEN

galaRatio = np.full(12*(nsideCourse**2), hp.UNSEEN)
galaRatio[np.where(fullOrigGala > 0)[0]] = fullCorrGala[np.where(fullOrigGala > 0)[0]] / fullOrigGala[np.where(fullOrigGala > 0)[0]]

# Saving correction information
my_table = Table()
my_table['PIX'] = np.where(starRatio > 0)[0]
my_table['WEIGHT'] = starRatio[np.where(starRatio > 0)[0]]
my_table['ORIG'] = fullOrigStar[np.where(starRatio > 0)[0]]
my_table['CORR'] = fullCorrStar[np.where(starRatio > 0)[0]]
my_table.write(starCorrectionFile, overwrite = True)

my_table = Table()
my_table['PIX'] = np.where(galaRatio > 0)[0]
my_table['WEIGHT'] = galaRatio[np.where(galaRatio > 0)[0]]
my_table['ORIG'] = fullOrigGala[np.where(galaRatio > 0)[0]]
my_table['CORR'] = fullCorrGala[np.where(galaRatio > 0)[0]]
my_table.write(galaCorrectionFile, overwrite = True)