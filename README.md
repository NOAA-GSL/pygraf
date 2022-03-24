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

Check out list of available NOAA BDP data sets here at [this link](
https://www.noaa.gov/organization/information-technology/list-of-big-data-program-datasets)

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
