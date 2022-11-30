# Creating Difference Plots using pygraf

The following attempts to explain how to plot your own difference maps using the
pygraf package. It is set up to work in much the same way as regular field maps
with a few exceptions:

- Variables accumulated in pygraf are not supported for difference maps, but
those that are accumulated in UPP and provided as a single variable are
supported.
- Pygraf will create a value range that is consistent with the data being plotted,
  so no two maps should be expected to have the same color scale.

Plotting values from two different models (e.g. HRRR and RRFS) is not tested,
and not supported at this time. Try at your own risk.


# Getting Started

Please see README.md for more detailed information on setting up the environment
for pygraf and running create_graphics.py


## Run the package

### Create difference maps

#### Configure the list of feilds

The real-time graphics produce more than 100 maps for each model at each
forecast lead time. The list of maps is configured in the pygraf subdirectory
image_lists/ where you will find a set of yaml files for a variety of supported
NWP systems run at GSL.

Start with the one that matches the model you'd like to plot, and remove or
comment out (add a `#` at the beginning of the line) the fields you are not
interested in.

You will provide this file path when you run the graphics in the next step.

#### Submitting the run script

See a full list of command line arguments by running the following command:

```
python create_graphics.py -h
```

Please see the README.md for an example of a Slurm batch script for running
pygraf in a batch job (recommended for large sets of graphics).


#### Examples

To create a set of difference plots between two experiments, you will use the
following arguments.

```
python -u create_graphics.py \
         diff \
         -d /path/to/input/data \
         --data_root2 /path/to/second/experiment \
         -f 0 12 \
         --file_type prs \
         --file_tmpl "RRFS_CONUS.t15z.bgdawpf{FCST_TIME:03d}.tm00.grib2" \
         --file_tmpl2 "RRFS_CONUS.t15z.bgdawpf{FCST_TIME:03d}.tm00.grib2" \
         --images ./image_lists/rrfs_subset.yml hourly \
         -m "RRFS Expt A - RRFS Expt B" \
         -n ${SLURM_CPUS_ON_NODE:-12} \
         -o /path/to/output/images \
         -s 2021052315 \
         --tiles full,ATL,CA-NV,CentralCA
```

Note: The `-u` flag on python allows the stdout/stderr to be unbuffered, so you
can watch the output stream into your log file when in batch mode.


To get a time-lagged difference, only one set of times can be done at once. Just
make sure you have removed templates from the paths and file templates, like
this:


```
python -u create_graphics.py \
         diff \
         -d /path/to/input/data \
         --data_root2 /path/to/second/experiment \
         -f 0 12 \
         --file_type prs \
         --file_tmpl "RRFS_CONUS.t15z.bgdawpf012.tm00.grib2" \
         --file_tmpl2 "RRFS_CONUS.t16z.bgdawpf011.tm00.grib2" \
         --images ./image_lists/rrfs_subset.yml hourly \
         -m "RRFS 15 Z - 16 Z" \
         -n ${SLURM_CPUS_ON_NODE:-12} \
         -o /path/to/output/images \
         -s 2021052315 \
         --tiles full,ATL,CA-NV,CentralCA
```






- 
