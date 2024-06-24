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
import Config
import StellarConfig as strConfig
import matplotlib.style
import matplotlib
matplotlib.style.use('des_dr1')

res = strConfig.res
nsideCourse = strConfig.nsideCourse
fracPer = strConfig.fracPer
numMagBins = strConfig.numMagBins
numBins = strConfig.numBins
classCutoff = strConfig.classCutoff
goldCols = Config.goldCols
gCut = strConfig.gCut
magBins = strConfig.magBins
cutOffPercent = strConfig.cutOffPercent
binNum = strConfig.binNum

path = strConfig.path
mu = strConfig.mu

pixFile = strConfig.pixFile

validPix = fitsio.read(pixFile)['PIXEL']
pixCheck = np.full(12*(res**2), False, dtype = bool)
pixCheck[validPix] = True

starDir = stellarDir + 'Stars/'

starProbFiles = []
starDetAsStarProbFiles = []
for i in np.arange(numMagBins):
    starProbFiles.append(starDir + 'Star_Prob_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    starDetAsStarProbFiles.append(starDir + 'Star_Det_As_Star_Prob_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    
galaDir = stellarDir + 'Galaxies/'

galaProbFiles = []
galaDetAsStarProbFiles = []
for i in np.arange(numMagBins):
    galaProbFiles.append(galaDir + 'Gala_Prob_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    galaDetAsStarProbFiles.append(galaDir + 'Gala_Det_As_Star_Prob_Bin' + str(i+1) + '_'+str(seed)+'.fits')
    
calibrationFile = stellarDir + 'Calibration/Calibrations_'+str(seed)+'.fits'
    
caliData = fitsio.read(calibrationFile)
starAdjustments = caliData['STAR']
galaAdjustments = caliData['GALA']

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

# This generates the fracDet data.
fracFile = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/fracdet/y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz'
fracData = fitsio.read(fracFile)

# This degrades it to nsideCourse resolution and applies a cut to where there is at least 75% coverage.
fracPix = fracData['PIXEL']
fracDet = fracData['SIGNAL']
origFracMap = np.full(12*(4096**2), 0.0)
origFracMap[fracPix] = fracDet
if res != 4096:
    origFracMap = hp.ud_grade(origFracMap, res, order_in = 'NESTED', order_out = 'NESTED')
origFracMap[~pixCheck] = 0.0 # If we aren't looking at the pixel, effective cover of 0%
fracMap = hp.ud_grade(origFracMap, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
fracPix = np.where(fracMap >= fracPer)[0]

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

deGalaCorrProb = []
for i in range(len(galaCorrProb)):
    
    fullProb = np.zeros(12*(res**2))
    fullProb[validPix] = galaCorrProb[i] * origFracMap[validPix]
    
    deFullProb = hp.ud_grade(fullProb, nsideCourse, order_in = 'NESTED', order_out = 'NESTED')
    deGalaCorrProb.append(deFullProb[fracPix] / fracMap[fracPix])
    
deGalaCorrProb = np.array(deGalaCorrProb, dtype = object)

origStar = np.full(len(fracPix), 0.0)
for i in np.arange(numMagBins):
    origStar = origStar + deClaStar[i]
    
origGala = np.full(len(fracPix), 0.0)
for i in np.arange(numMagBins):
    origGala = origGala + deClaGala[i]
    
corrStarBins = []
corrGalaBins = []
for i in np.arange(numMagBins):
    
    obsStars = (((deGalaCorrProb[i] * deClaStar[i]) + ((deGalaCorrProb[i] - 1) * deClaGala[i])) / ((deStarCorrProb[i] + deGalaCorrProb[i] - 1))).astype(float)
    obsStars[np.where(obsStars < 0)] = 0
    obsStars[np.where(obsStars >= deClaStar[i] + deClaGala[i])] = deClaStar[i][np.where(obsStars >= deClaStar[i] + deClaGala[i])] + deClaGala[i][np.where(obsStars >= deClaStar[i] + deClaGala[i])]
    
    obsGalas = deClaStar[i] + deClaGala[i] - obsStars
    
    CsfOs = obsStars * deStarCorrProb[i]
    CsfOg = obsGalas * (1 - deGalaCorrProb[i])
    
    CsfOsCorr = CsfOs / (deStarDetAsStarProb[i].astype(float))
    CsfOgCorr = CsfOg / (deGalaDetAsStarProb[i].astype(float))
    
    corrStarBins.append(CsfOsCorr + CsfOgCorr)
    
corrStar = np.sum(corrStarBins, axis = 0)

fullOrigStar = np.full(12*(nsideCourse**2), hp.UNSEEN)
fullOrigStar[fracPix] = origStar
fullOrigStar[np.where(fullOrigStar <= 0)[0]] = hp.UNSEEN

fullCorrStar = np.full(12*(nsideCourse**2), hp.UNSEEN)
fullCorrStar[fracPix] = corrStar
fullCorrStar[np.where(fullOrigStar <= 0)[0]] = hp.UNSEEN

starRatio = np.full(12*(nsideCourse**2), hp.UNSEEN)
starRatio[np.where(fullOrigStar > 0)[0]] = fullCorrStar[np.where(fullOrigStar > 0)[0]] / fullOrigStar[np.where(fullOrigStar > 0)[0]]

weightFile = stellarDir + 'Effective_Weights_'+str(seed)+'.fits'
my_table = Table()
my_table['PIX'] = np.where(starRatio > 0)[0]
my_table['WEIGHT'] = starRatio[np.where(starRatio > 0)[0]]
my_table.write(weightFile, overwrite = True)

nside_coverage = 32

hspStarRatio = hsp.HealSparseMap(nside_coverage=nside_coverage, healpix_map=starRatio)

# The default DES projection is a McBrydeSkymap.
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111)
sp = skyproj.DESSkyproj(ax=ax)
sp.draw_hspmap(hspStarRatio, cmap = 'viridis')
plt.clim(0.67, 1.5)
plt.colorbar(location = 'right', label = 'Effective Weight', fraction = 0.0267)
fig.suptitle(r'Corrected / Original', y = 0.9)
plt.savefig(stellarDir + 'Effective_Weight_Map_' + str(seed))

print('Done')