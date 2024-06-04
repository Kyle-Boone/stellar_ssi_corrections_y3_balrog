# stellar_ssi_corrections_y3_balrog

This repository contains code that learns classification probabilities and relative detection rates for stars and galaxies using synthetic sources.
This software has been written for use with data from the Dark Energy Survey (DES) Data Release 1, in particular the Y3 Gold catalog for physical objects and DES Y3 Balrog catalogs for synthetic sources.

## Potential Adjustments Before Running

Before running, there are a few potential adjustments that should be made to best suit individual science cases.
First, it should be checked that Config.py contains the desired survey property files for use.
Second, files should be adjusted within StellarConfig.py to the desired locations. 
These files are split with the top files being written to by our software and the bottom files being original data files that will not be written to.
Third, potential adjustments in object selection should be made to GetObjects.py and Y3Objects.py.
Currently these are set up for isochrone and flag cuts.
Current magnitude bins are decided by the r band magnitude.
To change this, adjustments must be made to Classification.py, Detection,py, and Y3Objects.py.

## Running the Pipeline

To generate all of the probabilities and calibrations necessary for the full correction, run FullRun.py.
To apply these corrections and get correction maps, run EffectiveWeightMap.py.