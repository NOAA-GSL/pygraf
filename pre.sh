#!/bin/bash

module purge


if [[ $(hostname) == u* ]] ; then
  module use -a /contrib/miniconda/modulefiles
  module load miniconda/25.3.1
  conda activate pygraf_rap
else
  module use -a /contrib/miniconda3/modulefiles
  module load miniconda3/4.12.0
  conda activate pygraf
fi

module list
