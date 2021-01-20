#!/bin/bash

#SBATCH --account=nrtrr
#SBATCH --qos=batch
#SBATCH --nodes=1-1
#SBATCH --partition=xjet
##SBATCH --partition=kjet,xjet,tjet,ujet
#SBATCH -t 02:15:00
#SBATCH --job-name=skew_t
#SBATCH -o /lfs1/BMC/wrfruc/cholt/adb_graphics/run_skewt-%j.log



source pre.sh

set -x

echo $(pwd)

scripts=/lfs1/BMC/wrfruc/cholt/adb_graphics

cdate=2021019012

cday=${cdate:0:8}
chour=${cdate:8:2}

data_dir="/lfs1/BMC/nrtrr/HRRR/run/${cdate}/postprd"
data_dir="/lfs4/BMC/nrtrr/NCO_dirs/ptmp/com/RRFS_CONUS/para/RRFS_dev1.${cday}/${chour}"


python create_graphics.py \
  skewts \
  -d $data_dir \
  -f 7 \
  --sites ${scripts}/static/conus_raobs5.txt  \
  --max_plev 100 \
  -o ${scripts}/skew_ts \
  -s ${cdate} \
  -n ${SLURM_CPUS_ON_NODE:-1} \
  --file_tmpl "RRFS_CONUS.t${chour}z.bgrd3df{FCST_TIME:03d}.tm${chour}.grib2"
#  -z $scripts/skew_ts
