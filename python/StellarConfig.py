import fitsio
import numpy as np
import Config
from os import listdir
from ugali.analysis.isochrone import factory as isochrone_factory
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

# Methods for isochrone calculation
def mkpol(mu, age=12., z=0.0004, dmu=0.5, C=[0.05, 0.05], E=4., err=None, survey='DECaLS', clip=None):
    if err == None:
        print('Using DES err!')
        err = surveys.surveys['DES_DR1']['err']
    """ Builds ordered polygon for masking """

    iso = isochrone_factory('Bressan2012', survey='des',
                            age=age, distance_modulus=mu, z=z)
    c = iso.color
    m = iso.mag

    # clip=4
    # clip = 3.4
    if clip is not None:
        # Clip for plotting, use gmin otherwise
        # clip abs mag
        cut = (m > clip) & ((m + mu) < 240) & (c > 0) & (c < 1)
        c = c[cut]
        m = m[cut]

    mnear = m + mu - dmu / 2.
    mfar = m + mu + dmu / 2.
    C = np.r_[c + E * err(mfar) + C[1], c[::-1] -  E * err(mnear[::-1]) - C[0]]
    M = np.r_[m, m[::-1]]
    return np.c_[C, M],iso
err=lambda x: (0.0010908679647672335 + np.exp((x - 27.091072029215375) / 1.0904624484538419))

def feh2z( feh):
        # Section 3 of Dotter et al. 2008
        Y_p     = 0.245            # Primordial He abundance (WMAP, 2003)
        c       = 1.54             # He enrichment ratio 

        # This is not well defined...
        #Z_solar/X_solar = 0.0229  # Solar metal fraction (Grevesse 1998)
        ZX_solar = 0.0229
        return (1 - Y_p)/( (1 + c) + (1/ZX_solar) * 10**(-feh))
    

'''
Hyperparameters:
-----------
res           : Healpixel nside resolution for calculations. Upper limit should be 4096.
perCovered    : If res < 4096, this determines what percent of the healpixel nees to be covered by valid pixels at 4096 resolution to be considered valid.
nsideCourse   : Healpixel nside resolution for final corrections. Lower resolutions improve stability of maximum likelihood step.
fracPer       : FracDet cutoff for a nsideCourse resolution healpixel to be included.
classCutoff   : Class cut between stars and galaxies.
gCut          : Upper limit for g band magnitudes.
numMagBins    : Number of magnitude bins.
magBins       : r Band magnitude bins, should have length one more than numMagBins.
binNum        : Number of bins used to find interpolation functions.
cutOffPercent : Accuracy threshold to terminate training.
detIndLim     : Cycle limit if accuracy threshold isn't met in detection rate training.
claIndLim     : Cycle limit if accuracy threshold isn't met in classification rate training.
numBins       : Number of spatial splits of full sky when calculating probabilities on full sky (larger values help prevent memory issues, no impact on final results).
matchDist     : Arcsecond matching distance for deep field to wide field object matching.
mu            : Stellar stream distance modulus.
age           : Stellar stream age (Gyr).
feh           : Stellar stream metallicity.
'''
res = 4096 
perCovered = 0.5 
nsideCourse = 512 
fracPer = 0.5
classCutoff = 1.5
gCut = 27
numMagBins = 3
magBins = [0, 22.9, 23.9, 24.5]
binNum = 10
cutOffPercent = 0.01
detIndLim = 300
claIndLim = 150
numBins = 100
matchDist = 0.5
mu = 16.2
age = 12.8
feh = -2.5

z=feh2z(feh)

mk,iso=mkpol(mu,age,z,dmu=0.5,C=[0.01,0.1],E=2,err=err, survey="DES_Y3A2")
path=Path(mk)

# This will be the new directory with any new files in it.
stellarDir = '/hdfs/bechtol/balrog/y3/y3a2_survey_conditions_maps/Kyle_Stuff/MaximumLikelihood/FinalPipeline/Phoenix_Class_' + str(classCutoff) + '/'

# Valid pixels file and files for cropped survey properties.
conditions = Config.conditions
pixFile = stellarDir + 'PixAndConds/Valid_Pixels.fits'
condFiles = []
for cond in conditions:
    condFiles.append(stellarDir + 'PixAndConds/' + cond + '.fits')
condFiles = np.array(condFiles)

# Balrog object information files.
matStarFile = stellarDir + 'InjectionData/Mat_Stars.fits'
detStarAllPosFile = stellarDir + 'InjectionData/Det_Stars_All_Position_Data.fits'

matGalaFile = stellarDir + 'InjectionData/Mat_Galaxies.fits'
detGalaAllPosFile = stellarDir + 'InjectionData/Det_Galaxies_All_Position_Data.fits'

# The following is a directory for training and probability data for the Balrog galaxies.
galaDir = stellarDir + 'Galaxies/'

# Extrapolation, training, and probability files.
galaExtrFiles = []
galaTrainFiles =  []
galaProbFiles = []
for i in np.arange(numMagBins):
    galaExtrFiles.append(galaDir + 'Gala_Extr_Bin' + str(i+1) + '.fits')
    galaTrainFiles.append(galaDir + 'Gala_Train_Bin' + str(i+1) + '.fits')
    galaProbFiles.append(galaDir + 'Gala_Prob_Bin' + str(i+1) + '.fits')
    
galaDetAsStarExtrFiles = []
galaDetAsStarTrainFiles =  []
galaDetAsStarProbFiles = []
for i in np.arange(numMagBins):
    galaDetAsStarExtrFiles.append(galaDir + 'Gala_Det_As_Star_Extr_Bin' + str(i+1) + '.fits')
    galaDetAsStarTrainFiles.append(galaDir + 'Gala_Det_As_Star_Train_Bin' + str(i+1) + '.fits')
    galaDetAsStarProbFiles.append(galaDir + 'Gala_Det_As_Star_Prob_Bin' + str(i+1) + '.fits')
    
galaDetAsGalaExtrFiles = []
galaDetAsGalaTrainFiles =  []
galaDetAsGalaProbFiles = []
for i in np.arange(numMagBins):
    galaDetAsGalaExtrFiles.append(galaDir + 'Gala_Det_As_Gala_Extr_Bin' + str(i+1) + '.fits')
    galaDetAsGalaTrainFiles.append(galaDir + 'Gala_Det_As_Gala_Train_Bin' + str(i+1) + '.fits')
    galaDetAsGalaProbFiles.append(galaDir + 'Gala_Det_As_Gala_Prob_Bin' + str(i+1) + '.fits')
    
# The following is a directory for training and probability data for the Balrog delta stars.
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
    
starDetAsGalaExtrFiles = []
starDetAsGalaTrainFiles =  []
starDetAsGalaProbFiles = []
for i in np.arange(numMagBins):
    starDetAsGalaExtrFiles.append(starDir + 'Star_Det_As_Gala_Extr_Bin' + str(i+1) + '.fits')
    starDetAsGalaTrainFiles.append(starDir + 'Star_Det_As_Gala_Train_Bin' + str(i+1) + '.fits')
    starDetAsGalaProbFiles.append(starDir + 'Star_Det_As_Gala_Prob_Bin' + str(i+1) + '.fits')
    
# Files with information on gold objects.
goldStarDir = stellarDir + 'GoldObjects/Stars/'
goldGalaDir = stellarDir + 'GoldObjects/Galaxies/'

# These just contain counts at the valid pix for the different magnitude bins for number of objects.
goldStarFiles = []
goldGalaFiles = []
for i in np.arange(numMagBins):
    goldStarFiles.append(goldStarDir + 'Bin' + str(i+1) + '.fits')
    goldGalaFiles.append(goldGalaDir + 'Bin' + str(i+1) + '.fits')
    
# These include color and magnitude information necessary for calibrations.
goldMoreInfoStarFiles = []
goldMoreInfoGalaFiles = []
for i in np.arange(numMagBins):
    goldMoreInfoStarFiles.append(goldStarDir + 'More_Info_Bin' + str(i+1) + '.fits')
    goldMoreInfoGalaFiles.append(goldGalaDir + 'More_Info_Bin' + str(i+1) + '.fits')
    
# Overall multiplicative corrections on valid pix.
starCorrectionFile = stellarDir + 'Correction/StarCorrections.fits'
galaCorrectionFile = stellarDir + 'Correction/GalaxyCorrections.fits'

# Calibration information for different magnitude bins.
calibrationFile = stellarDir + 'Calibration/Calibrations.fits'

# Information on phoenix position, used for testing.
phoenixFile = stellarDir + 'Phoenix_Pix.fits'
backgroundFile = stellarDir + 'Background_Pix.fits'
