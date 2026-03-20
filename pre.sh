#!/bin/bash

module purge

if [[ $(hostname) == u* ]] ; then
  module use -a /contrib/miniconda/modulefiles
  module load miniconda/25.3.1
else
  module use -a /contrib/miniconda3/modulefiles
  module load miniconda3/25.11.0
fi
conda activate pygraf

module list
