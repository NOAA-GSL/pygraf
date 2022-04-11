# ADB Graphics Creation for UPP Model Output

> Note: This repository is under heavy development. Use at your own risk!

This repository houses a Python-based implementation of the graphics package
that is responsible for generating maps for the RAP/HRRR/FV3/RRFS data. It has
replaced NCL as the real-time graphics creation package at NOAA GSL for maps and
SkewT diagrams.

# Overview

The adb_grapics Python package currently includes tools to create SkewT diagrams
and the total plan-view maps created for real-time experimental HRRR runs
available on the [HRRR Page](https://rapidrefresh.noaa.gov/hrrr/).

# Getting Started

## Download the source code

The repo contains large files totaling 1.2 Gb or more and they are managed by
GIT LFS that are used for testing purposes. To avoid downloading these files
when you clone, use the following command:

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NOAA-GSL/pygraf
```

If you have the disk space, internet data, and/or would like to run the test
suite included with the package, clone the standard way:

```
git clone https://github.com/NOAA-GSL/pygraf
```


## Python environment

A Python environment is available on NOAA RDHPCS Platforms ready to use. To
activate this environment, do the following:

```
module use -a /contrib/miniconda3/modulefiles
module load miniconda3
conda activate pygraf
```


## Stage data


There are several real-time data locations on RDHPCS Platforms, but they store only a short
rolling window. If your desired data is available currently, copy it to your
space.

Otherwise, you can retrieve it from HPSS or even the NOAA Big Data Project cloud
buckets.

An example of pulling a wgrib file for HRRR from BDP:

```
wget https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20220214/conus/hrrr.t00z.wrfprsf06.grib2
```

Check out the list of available NOAA BDP data sets here at [this link](
https://www.noaa.gov/organization/information-technology/list-of-big-data-program-datasets)

When you are browsing the AWS buckets, you can find the appropriate file path to
use with `wget` by "right clicking" the file name, and copying the link address. 

## Run the package

### Creating maps

#### Configure the list of fields

The real-time graphics produce more than 100 maps for each model at each
forecast lead time. The list of maps is configured in the pygraf subdirectory
`image_lists/` where you will find a set of yaml files for a variety of
supported NWP systems run at GSL.

Start with the one that matches the model you'd like to plot, and remove or
comment (add a `#` at the beginning of the line) out the fields you are not
interested in.

You will provide this file path when you run the graphics in the next step.

#### Submitting the run script

See a full list of command line arguments by running the following command:

```
python create_graphics.py -h
```

If you are creating only a couple of maps using RRFS data as an example you can
run on the front-end nodes (or on your laptop) with a command like this:

```
python create_graphics.py \
         maps \
         --all_leads \
         -d /path/to/input/data \
         -f 0 6 \
         --file_type prs \
         --file_tmpl "RRFS_NA_3km.t15z.bgdawpf{FCST_TIME:03d}.tm00.grib2" \
         --images ./image_lists/rrfs_subset.yml hourly \
         -m "My RRFS Retro" \
         -n 4 \
         -o /path/to/output/images \
         -s 2021052315 \
         --tiles full
```

If you have a larger set of maps to create, you might use a Slurm batch script
that looks like this, and is submitted from the pygraf directory:

```
#!/bin/bash

#SBATCH --account=my_jet_account
#SBATCH --qos=batch
#SBATCH --nodes=1-1
#SBATCH --exclusive
#SBATCH --partition=kjet,xjet,tjet,ujet
#SBATCH -t 1:30:00
#SBATCH --job-name=maps

source pre.sh

python create_graphics.py \
         maps \
         --all_leads \
         -d /path/to/input/data \
         -f 0 12 \
         --file_type prs \
         --file_tmpl "RRFS_NA_3km.t15z.bgdawpf{FCST_TIME:03d}.tm00.grib2" \
         --images ./image_lists/rrfs_subset.yml hourly \
         -m "My RRFS Retro" \
         -n ${SLURM_CPUS_ON_NODE:-12} \
         -o /path/to/output/images \
         -s 2021052315 \
         --tiles full,ATL,CA-NV,CentralCA

```
NOTE: The graphics already run as a workflow step in the RRFS Retros! They may be
zipped by default, so you can unzip those files to see your images on disk.

### Creating Skew-T Diagrams

#### Configure the list of locations

The real-time graphics produce a set of about 100 Skew Ts at predefined
locations. You can use those locations, or create your own. The standard
locations are defined in the `static` directory under `pygraf` and each line
in the file represents a single location.

The format of each line is crucial for the columns leading up to the Site Name.
The Site Name can be any string, but must start on or after position 37 of the
line.

Here is an example for Las Vegas, NV:

``` VEF   3120 72388  36.05 115.18  693 Las Vegas, NV```

Station ID: VEF
Column 2: Unused number
Site Number: 72388
Lat: 36.05
Lon: 115.18
Column 6: Unused number
Site Name: Las Vegas, NV


#### Submitting the run script

See a full list of command line arguments by running the following command:

```
python create_graphics.py -h
```

If you are creating only a couple of maps using RRFS data as an example you can
run on the front-end nodes (or on your laptop) with a command like this:

```
python create_graphics.py \
         skewts \
         -d /path/to/input/data \
         -f 6 \
         --file_type nat \
         --file_tmpl "RRFS_NA_3km.t15z.bgrd3df{FCST_TIME:03d}.tm00.grib2" \
         --max_plev 100 \
         -m "My RRFS Retro" \
         -n 4 \
         -o /path/to/output/images \
         -s 2021052315 \
         --sites ./static/sites_file.txt \
```

If you are creating many Skew-Ts, please submit a batch job. You can modify the
maps example above to run this command.


# Troubleshooting

- Getting an error like this?

```
      File "create_graphics.py", line 41
        LOG_BREAK = f"{('-' * 80)}\n{('-' * 80)}"
                                                ^
    SyntaxError: invalid syntax
```
You probably don't have the conda environment loaded, and the system default
Python 2 is trying to run Python 3 code.  You may also see an error like this
when you've loaded the module, but haven't activated the pygraf environment:

```
    Traceback (most recent call last):
      File "create_graphics.py", line 7, in <module>
        import matplotlib as mpl
    ModuleNotFoundError: No module named 'matplotlib'
```


# View the output files

You have several options to view the resulting PNG files that are created on
RDHPCS platforms. You can do the standard data transfer to a local machine, use
the Linux utility `display` with the appropriate ssh connection flags, or you
can spin up a Jupyter Notebook through ssh tunneling.

Follow along with RDHPCS Docs for [using Jupyter notebooks over
SSH](https://rdhpcs-common-docs.rdhpcs.noaa.gov/wiki/index.php/Anaconda#Jupyter_Notebooks_over_SSH_and_front-end_nodes)
to set up your connection with *one slight modification*.

Once you setup your first window, you will spin up your Jupyter Notebook
instance in the scratch space where your PNG file lives by doing the following:

```
cd /my/scratch/space
module use -a /contrib/miniconda3/modulefiles
module load miniconda3
conda activate pygraf
jupyter notebook --no-browser --port=8809
```

The `jupyter notebook` command will provide you with a secure URL that you can
drop into your web browser to view your notebook.

Now you can continue on with the instructions for Window 2.

## Troubleshooting

- For the tunneling to work, you cannot have any other connections open on the
platform where you are attempting to tunnel. When you are opening your "Window
1" in the RDHPCS docs, and see the following message when you log on, you won't
be able to connect to your notebook. Shut down all connections and try again.

```
bind: Address already in use
channel_setup_fwd_listener_tcpip: cannot listen to port: 21952
Could not request local forwarding.
```


# Contributing

When contributing code to this repo, please keep in mind the following style
guidelines:

- All Python code must pass the linter with 10/10.
- All code must pass tests, and tests must be updated to accommodate new code.
- Style beyond linting:
  - Alphabetize lists (anywhere another order is not more obvious to everyone)
  - A single white space line before and after comments.
  - A single white space after each method/function. Two after classes.
  - Lists are maintained with each item on a single line followed by a comma,
  even the last item.

This repository is using a minor variation on GitLab flow, requiring new work be
contributed via Pull Request from a branch with reviewers (required). Releases
will be handled with tags (as opposed to branches, in the original GitLab flow),
and will be marked as versions with v[major].[minor].[update].

# Contact

| Name | Email |
| ---- | :---- |
| Christina Holt  | christina.holt@noaa.gov   |
| Brian Jamison   | brian.d.jamison@noaa.gov  |
| Craig Hartsough | craig.hartsough@noaa.gov  |
