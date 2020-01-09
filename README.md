# ADB Graphics Creation for UPP Model Output

This repository houses a Python-based implementation of the graphics package
that is responsible for generating maps for the RAP/HRRR/FV3 data. It is
eventually meant to replace the NCL Graphics suite currently populating the
real-time pages (https://rapidrefresh.noaa.gov/hrrr/).

# Overview

The adb_grapics Python package contains all the necessary modules for generating
the graphics.

# Getting Started

In addition to the information below, checkout the the ADB Python Graphics [Google Doc](https://docs.google.com/document/d/1mlLSmFZ-gkNXuF7HmD58WEwJgJVHNcKsicrWXpryFEU/edit#) for more information.

## Python environment

The Python environment required to generate the graphics is defined by the
environment.yml file. This requires access to an implementation of conda. To
install this conda environment named adb_graphics, do the following:

```
conda env create -f environment.yml
conda activate adb_graphics
```

You only need to create the environment once (or when it's updated in the repo).
You will need to activate it each time you'd like to run any of the adb_graphics
package.

## Run the package

An example script has been included for generating a single figure, and serves
as an example for how to call the package for an upper air map. To run the
example script in the top-level repo directory, type: 

```
conda activate adb_graphics
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

This repository is using a minor variation on GitLab flow, requiring new work be contributed via
Pull Request from a branch with reviewers (required). Releases will be handled with tags
(as opposed to branches, in the original GitLab flow), and will be marked as
versions with v[major].[minor].[update].

# Contact

| Name | Email |
| ---- | :---- |
| Christina Holt  | christina.holt@noaa.gov   |
| Brian Jamison   | brian.d.jamison@noaa.gov  |
| Craig Hartsough | craig.hartsough@noaa.gov  |
