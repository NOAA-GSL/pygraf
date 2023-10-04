#!/bin/bash

module purge

module use -a /contrib/miniconda3/modulefiles
module load miniconda3/4.5.12
conda activate pygraf

module load intel/2022.1.2
module load wgrib2/2.0.8

module list
