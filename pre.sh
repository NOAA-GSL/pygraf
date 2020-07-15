#!/bin/bash

module purge

module use -a /contrib/miniconda3/modulefiles
module load miniconda3
conda activate pygraf

module list
