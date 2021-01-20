#!/bin/bash

#SBATCH --account=nrtrr
#SBATCH --qos=batch
#SBATCH --nodes=1-1
#SBATCH --exclusive
#SBATCH --partition=kjet,xjet
##SBATCH --partition=kjet,xjet,tjet,ujet
#SBATCH -t 02:15:00
#SBATCH --job-name=maps
#SBATCH -o /lfs1/BMC/wrfruc/cholt/adb_graphics/run_graphics_zip-%j.log


model='hrrr'


source pre.sh

set -x

echo $(pwd)
which python

scripts=/lfs1/BMC/wrfruc/cholt/adb_graphics
fhr=6
cdate=2021011812

cday=${cdate:0:8}
chour=${cdate:8:2}

if [[ $model == 'hrrr' ]] ; then

  dataroot=/lfs1/BMC/nrtrr/HRRR/run/${cdate}/postprd
  file_tmpl='wrfprs_hrconus_{FCST_TIME:02d}.grib2'
  file_type=prs
  images="image_lists/hrrr_subset.yml hourly"
  output=${scripts}/hrrr_maps
  tiles='full NC'
  tiles='full'
  tiles=$1
  tiles='full GreatLakes ATL SEA-POR'

elif [[ $model == 'rapfull' ]] ; then

  dataroot=/whome/rtrr/rap_databasedir/cycle/${cdate}/postprd
  file_tmpl='wrfprs_rr_{FCST_TIME:02d}.grib2'
  file_type=prs
  images="image_lists/rap.yml hourly"
  output=${scripts}/rap_maps
  tiles='full HI'

elif [[ $model == 'rapsub' ]] ; then

  dataroot=/whome/rtrr/rap_databasedir/cycle/${cdate}/postprd
  file_tmpl='wrfprs_130_{FCST_TIME:02d}.grib2'
  file_type=prs
  images="image_lists/rap.yml hourly"
  output=${scripts}/rap_maps
  tiles='conus NC NE NW SC SE SW'

elif [[ $model == 'rapak' ]] ; then

  dataroot=/whome/rtrr/rap_databasedir/cycle/${cdate}/postprd
  file_tmpl='wrfprs_242_{FCST_TIME:02d}.grib2'
  file_type=prs
  images="image_lists/rap.yml hourly"
  output=${scripts}/rap_maps
  tiles='AK AKZoom'

elif [[ $model == 'rrfs' ]] ; then

  dataroot=/lfs4/BMC/nrtrr/NCO_dirs/ptmp/com/RRFS_CONUS/para/RRFS_dev1.${cday}/${chour}
  file_tmpl="RRFS_CONUS.t${chour}z.bgdawpf{FCST_TIME:03d}.tm${chour}.grib2"
  file_type=prs
  images="image_lists/rrfs_subset.yml hourly"
  output=${scripts}/rrfs_maps
  tiles='full'

else 
  echo "No model chosen!"
  exit
fi

python create_graphics.py \
  maps \
  -d ${dataroot} \
  -f 7 \
  --file_tmpl ${file_tmpl} \
  --file_type ${file_type} \
  --images ${images} \
  -n ${SLURM_CPUS_ON_NODE:-12} \
  -o ${output} \
  -s ${cdate} \
  --tiles ${tiles}
#  -z ${output}
