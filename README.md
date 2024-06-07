# stellar_ssi_corrections_y3_balrog

This repository contains code that learns classification probabilities and relative detection rates for stars and galaxies using synthetic sources.
This software has been written for use with data from the Dark Energy Survey (DES) Data Release 1, in particular the Y3 Gold catalog for physical objects and DES Y3 Balrog catalogs for synthetic sources.

## Configuration Files

There are two configuration files in this pipeline: Config.py and StellarConfig.py.
Config.py primarily consists of original data files from DES that are read from.
These files include DES Y3 Gold object catalogs, Balrog Y3 object catalogs, deep field object catalogs, a fracDet file (contains the percentage of each healpixel that was observed), and survey property maps.
Config.py also contains a list of the name of each survey property used.
Finally, Config.py contains lists of relevant columns for some of the different files which share column names.

If adding additional survey properties or removing some, adjustments may need to be made to the CropSurveyProperties.py program.
Currently there are two varieties of file structures which have to be treated separately and then converted to a similar format for further use.
CropSurveyProperties.py will work as long as all but one file (the stellar density survey property map) have the same format.
This program also assumes that these maps are being stored at a healpixel resolution of 4096.
With this being said, adding in new survey properties that just have new names but the same format as all the non stellar density survey properties is as easy as just adding extra files to the origCondFiles list and adding the name of the survey property to the conditions list in Config.py.

Besides this and adjusting for your own file locations, Config.py should remain the same if you wish to apply the same cuts as described in the corresponding paper to this repository.
To apply any other cuts, bin on different magnitude bands, or anything of that sort, the methods throughout this repository will have to be adjusted accordingly.

The second configuration file is StellarConfig.py.
This contains files to write information to as well as hyperparameters for the run.
These can all be adjusted without any other modifications to the other programs to perform a run with different parameter values.

## Running the Pipeline

To generate all of the probabilities and calibrations necessary for the full correction, run FullRun.py.
To apply these corrections and get correction maps, run EffectiveWeightMap.py.