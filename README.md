# ADB Graphics Creation for UPP Model Output

This repository houses a Python-based implementation of the graphics package
that is responsible for generating maps for the RAP/HRRR/FV3 data. It is
eventually meant to replace the NCL Graphics suite currently populating the
real-time pages (https://rapidrefresh.noaa.gov/hrrr/).

# Overview

The ADB_Grapics Python package contains all the necessary modules for generating
the graphics.

# Getting Started

## Python environment

The Python environment required to generate the graphics is defined by the
environment.yml file. This requires access to an implementation of conda. To
install this conda environment named adb_graphics, do the following:

```
conda env create -f environment.yml
conda activate adb_graphics
```

## Run the package

An example script has been included for generating a single figure, and serves
as an example for how to call the package for an upper air map. To run the
example script in the top-level repo directory, type: 

```
python plot_example.py
```

# Contact

Christina Holt, christina.holt@noaa.gov
Brian Jamison, brian.jamison@noaa.gov
Craig Hartsough, craig.hartsough@hoaa.gov
