What is SAGIT
=======

SAGIT stands for Selective Automated Group Integrated Tractography. 
It's a Python library that automates the generation of tractography for Neuroimaging research. 

It's developed by David Qixiang Chen as part of his PhD thesis with Dr. Mojgan Hodaie at the University of Toronto. 

It supports popular tractography methods:
* Single tensor tractography (DTI)
* Extended Streamline Tractography (XST)
* Constrained Spherical Convolutional tractogrpahy (CST)

See (https://sinkpoint.gitbooks.io/sagit/content/) for full documentation. 

# Setup

## Requirements

SAGIT is developed and tested on Ubuntu 14.04 LTS. 

SAGIT relies on a number of neuroimaging software packages for its functionality:
* 3D Slicer 4 (DTI)
* Hodaie TEEM (XST)
* MRTrix 3 (CST)
* Freesurfer
* Anatomical Normalization Tools (ANTs)

As well as these python libraries
* nibabel
* dipy
* scipy
* numpy
* pyparsing
* jsoncomment

A full list of packages can be found in *requirement.txt*

## Install


### environment
First make sure python>2.6

In .bashrc, create entries for SLICER_4HOME:
```
export SLICER4_HOME="/path/to/slice4"
```


### SAGIT setup
It's recommended to have an isolated python environment for SAGIT
This can be done with **virtualenvwrapper**
```
pip intall virtualenvwrapper
```

Then create a new environment for SAGIT
```
mkvirtualenv sagit
```

Then in the future, to return to the python environment, use
```
workon sagit
```

Install the requirements with 
```
pip install -r requirements.txt
```

Then install the library
```
python setup.py install 
```