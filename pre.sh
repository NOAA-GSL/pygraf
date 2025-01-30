#!/bin/bash

module purge

module use -a /contrib/miniconda3/modulefiles
module load miniconda3/4.12.0
conda activate pygraf

module list
