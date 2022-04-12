#!/bin/bash

#SBATCH --account=nrtrr
#SBATCH --qos=debug
#SBATCH --nodes=1-1
#SBATCH --exclusive
#SBATCH --partition=bigmem
##SBATCH --partition=kjet,xjet
##SBATCH --partition=kjet,xjet,tjet,ujet
#SBATCH -t 0:30:00
##SBATCH -t 02:15:00
#SBATCH --job-name=maps
##SBATCH -o /lfs1/BMC/wrfruc/cholt/repos/pygraf_dev/run_graphics_zip_kjet_%j.log
#SBATCH -o /lfs1/BMC/wrfruc/cholt/repos/pygraf_dev/run_graphics_zip_%j.log


model=$1

partition=kjet


#source pre.sh

set -x

echo $(pwd)
which python

scripts=/lfs1/BMC/wrfruc/cholt/repos/pygraf_dev
fhr=
cdate=2021082512
cdate=2022040618

START_DATE=`echo "${cdate}" | sed 's/\([[:digit:]]\{2\}\)$/ \1/'`

cday=${cdate:0:8}
chour=${cdate:8:2}
cjul=`date +%y%j%H -d "${START_DATE}"`

tiles=full,SE,NE,SC,NC,SW,NW
tiles=SE
tiles=SE,SC,SW
tiles=full
file_type=prs
output=${scripts}/ens_panel

model=rrfs

case $model in

  "rrfs")
      chour=${cdate:8:2}
      dataroot="/mnt/lfs4/BMC/wrfruc/RRFSE/NCO_dirs/ptmp/com/RRFSE_CONUS/para/RRFS_ens.${cday}/${chour}/mem{mem}/hrrr_grid"
      dataroot="/mnt/lfs1/BMC/wrfruc/cholt/data/rrfse/RRFS_conus_3km.$cday/$chour/mem{mem:04d}/hrrr_grid"
      file_tmpl="RRFSE_CONUS.t${chour}z.bgdawpf{FCST_TIME:03d}.tm00.grib2"
      file_tmpl="RRFS_CONUS.t${chour}z.bgdawpf{FCST_TIME:03d}.tm00.grib2"
      images="image_lists/rrfs_ens.yml hourly"
      name='RRFSE'
      ;;

  *)

    echo "No model chosen!"
    exit 1
    ;;

esac

python -u create_graphics.py \
  enspanel \
  -d ${dataroot} \
  -f 0 \
  --file_tmpl ${file_tmpl} \
  --file_type ${file_type} \
  --images ${images} \
  -m "${name}" \
  -n ${SLURM_CPUS_ON_NODE:-6} \
  -o ${output} \
  -s ${cdate} \
  --tiles ${tiles} \
  --ens_size 10
#  -z ${output}
