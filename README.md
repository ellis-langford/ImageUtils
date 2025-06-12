<div align="center">
  <img src="./assets/image_utils_logo.png" width="700">
  <br><br>
  <p align="center"><strong>An image processing toolbox</strong></p>
</div>

<div align="center" style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
  <a href="https://profiles.ucl.ac.uk/101480-ellis-langford"><img src="https://custom-icon-badges.demolab.com/badge/UCL Profile-purple?logo=ucl" alt="UCL Profile"></a>
  <a href="https://orcid.org/0009-0006-1269-2632"><img src="https://img.shields.io/badge/ORCiD-green?logo=orcid&logoColor=white" alt="ORCiD"></a>
  <a href="https://github.com/ellis-langford"><img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://uk.linkedin.com/in/ellis-langford-8333441ab"><img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff" alt="LinkedIn"></a>
</div>

## Introduction

ImageUtils is an image processing toolbox for neuroimaging data.


## Requirements

To successfully run the ImageUtils pipeline, please ensure the following requirements are met:

**Ubuntu 22.04 + Python 3.10**<br>
*(other versions may be compatible but have not been tested)*


## Installation & Quick Start

To install the necessary components for ImageUtils, please follow the steps below:

- Clone the ImageUtils docker image from the GitHub repo:
  
  ```bash
  git clone https://github.com/ellis-langford/ImageUtils.git utils
  ```
  
- Add the toolbox directory to your pythonpath:
  
  ```python
  sys.path.append(os.path.abspath("/path/to/code/utils"))

  ```

- Import the toolbox classes to your notebook
  
  ```python
  from helpers import Helpers
  from utils import Utils
  ```

- Instantiate the class object:
  
  ```bash
  helpers = Helpers()
  utils = Utils()

- Use functions inside your notebook:
  
  ```bash
  helpers.plugin_log("Hello World")


## Classes and Functions

The available classes include:
1. `base_cog.py`

   > *BaseCog:* sets up the core functionality of the toolbox and parses parameters

2. `helpers.py`

   > *Path:* functions relating to filepaths, e.g. parse folder name, file name, file extension
   > *Helpers:* general helper functions, e.g. general logging, error logging

3. `utils.py`

   > *Utils:* general image processing functions, e.g. upsample, downsample, origin reset, binning, image compression
   > *Convert:* functions which allow conversion between different file types
   > *Plotting:* functions relating to image and data plotting

This project is being continually updated.


## Citation

If you find this work useful to you, feel free to cite the following reference:

```
@ARTICLE{xxxxxxxx,
  author={Langford E},
  journal={}, 
  title={}, 
  year={},
  volume={},
  number={},
  pages={},
  doi={}}
```