# ADB Graphics Creation for UPP Model Output

> Note: This repository is under heavy development. Use at your own risk!

This repository houses a Python-based implementation of the graphics package
that is responsible for generating maps for the RAP/HRRR/FV3 data. It is
eventually meant to replace the NCL Graphics suite currently populating the
real-time pages (https://rapidrefresh.noaa.gov/hrrr/).

# Overview

The adb_grapics Python package currently includes tools to create SkewT diagrams
and a subset of the total plan-view maps created for real-time experimental HRRR
runs available on the [HRRR Page](https://rapidrefresh.noaa.gov/hrrr/). These
graphics are not yet fully operational, so will vary sligthly from the graphics
on the HRRR Page.


# Getting Started

In addition to the information below, checkout the the ADB Python Graphics
[Google Doc](https://docs.google.com/document/d/1mlLSmFZ-gkNXuF7HmD58WEwJgJVHNcKsicrWXpryFEU/edit#)
for more information.

## Python environment

A Python environment is available on NOAA RDHPCS Platforms ready to use. To
activate this environment, to the following:

```
module use -a /contrib/miniconda3/modulefiles
module load miniconda3
conda activate pygraf
```

## Run the package

An example script has been included for generating a single figure, and serves
as an example for how to call the package for an upper air map. To run the
example script in the top-level repo directory, type:

```
conda activate pygraf
python plot_example.py
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
