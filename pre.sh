#!/bin/bash

module purge

module use -a /contrib/miniconda3/modulefiles
module load miniconda3/4.5.12
conda activate pygraf

module load wgrib2/0.1.9.6a

module list
