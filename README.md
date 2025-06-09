<p align="center">
  <img src="./assets/image_utils_logo.png" width="700"/>
</p>

## ImageUtils: An Image Processing Toolbox

### 1. Requirements

To successfully run the ImagePrep pipeline, please ensure the following requirements are met:

**Ubuntu 22.04 + Python 3.10**<br>
*(other versions may be compatible but have not been tested)*

### 2. Installation & Quick Start

To install the necessary components for ImageUtils, please follow the steps below:

- Clone the ImageUtils docker image from the GitHub repo:
  
  ```bash
  git clone https://github.com/ellis-langford/ImageUtils.git
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

### 3. Classes and Functions

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

### 5. Citation

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

### Useful links

<!-- [![](https://img.shields.io/badge/Software-ANTsPy-orange)](https://antspy.readthedocs.io/en/latest/index.html) -->

### Acknowledgements

This pipeline was developed whilst completing my PhD research at University College London.

### References

<!-- 1. *B. B. Avants, N. Tustison, P. A. Cook, et al., “ANTs: Advanced Normalization Tools in Python (ANTsPy),” Insight Journal, 2019.* -->