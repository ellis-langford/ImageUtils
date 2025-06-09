"""
Generic utility methods
"""
__author__   = ["ellis.langford.19@ucl.ac.uk"]
__modified__ = "21-Feb-2025"

# Imports
import os
import sys
import shutil
import numpy as np
import nibabel as nib
from PIL import Image
from scipy.ndimage import zoom
from pathlib import Path
import tifffile
import imagecodecs
import json
import nilearn.plotting as plotting

# Add the utils directory to the Python path
utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
sys.path.append(utils_path)

# Custom imports
from base_cog import BaseCog
from helpers import Helpers, Path

class Utils(BaseCog):
    """
    Generic utility methods
    """
    def __init__(self, **kwargs):
        """
        Instantiate the Utils class.
        """
        super().__init__(**kwargs)

        # Instantiate custom modules
        self.helpers = Helpers()
        self.path = Path()

    def copy(self, source, dest, contents_only=False):
        """
        Copy a file or directory from source to destination.

        Parameters:
        source (str)         : Path to the source file or directory.
        dest (str)           : Path to the destination file or directory.
        contents_only (bool) : If True, the contents of source directory will be copied
                             : If False, the entire source directory will be copied
        """
        try:
            if not os.path.exists(source):
                print(f"Source path '{source}' does not exist.")
                return

            if os.path.isfile(source):
                # Handle if destination is a directory
                if os.path.isdir(dest):
                    dest = os.path.join(dest, os.path.basename(source))

                # Create the destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(source, dest)

            elif os.path.isdir(source):
                # If the destination directory does not exist, create it
                if not os.path.exists(dest):
                    os.makedirs(dest, exist_ok=True)

                # Perform the directory copy
                if contents_only:
                    shutil.copytree(source, dest, dirs_exist_ok=True)
                else:
                    dest_dir = os.path.join(dest, os.path.basename(source))
                    shutil.copytree(source, dest_dir, dirs_exist_ok=True)

            else:
                self.helpers.errors(f"Source '{source}' is not a valid file or directory.")
        except Exception as e:
            self.helpers.errors(f"An error occurred while copying: {e}")

    def compress_ims(self, image):
        """
        Compress .nii images to .nii.gz

        Parameters:
        image (str): Path to image to compress
        """
        im = nib.load(image)
        data = im.get_fdata()
        img_nii = np.asarray(data).astype(np.float32)
        result_image = nib.Nifti1Image(img_nii, im.affine, im.header)
        nib.save(result_image, image.replace(".nii", ".nii.gz"))

    def compare_nifti_images(self, nii_file1: str, nii_file2: str) -> bool:
        """
        Compare two NIfTI (.nii.gz) images to check if they are identical.
    
        Parameters:
        nii_file1 (str): Path to the first NIfTI file.
        nii_file2 (str): Path to the second NIfTI file.
    
        Returns:
        bool: True if the images are identical, False otherwise.
        """
        # Load the NIfTI files
        image1 = nib.load(nii_file1).get_fdata()
        image2 = nib.load(nii_file2).get_fdata()
    
        # Compare the data arrays
        if not np.array_equal(image1, image2):
            self.helpers.plugin_log("Data arrays are different")
            return False
    
        # Compare the headers
        header1 = nib.load(nii_file1).header
        header2 = nib.load(nii_file2).header
    
        if header1 != header2:
            self.helpers.plugin_log("Headers are different")
            return False
    
        self.helpers.plugin_log("NIfTI images are identical")
        return True

    def compare_tif_images(self, image_path1, image_path2):
        """
        Compare two TIFF images to check if they are identical, including headers.

        Parameters:
        image_path1: Path to the first TIFF image.
        image_path2: Path to the second TIFF image.

        Returns:
        bool: True if images and metadata are identical, False otherwise.
        """
        try:
            # Open images
            img1 = Image.open(image_path1)
            img2 = Image.open(image_path2)
    
            # Compare image pixel data
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            if not np.array_equal(arr1, arr2):
                self.helpers.plugin_log("Image data is different")
                return False
    
            # Compare TIFF headers (metadata)
            info1 = img1.info
            info2 = img2.info
            if info1 != info2:
                self.helpers.plugin_log("TIFF headers are different")
                return False
    
            self.helpers.plugin_log("Images are identical")
            return True
    
        except Exception as e:
            self.helpers.errors(f"Error: {e}")
            return False

    def downsample_nifti(self, input_im_fpath: str, outpath: str, 
                         vox_size_target: float = None,
                         vox_size_target_max: bool = True,
                         downsample_factor: int = None):
        """
        Downsample a NIfTI image by a specified factor or target voxel size.
    
        Parameters:
        input_im_fpath (str): Path to the input NIfTI (.nii.gz) file.
        outpath (str): Path for the downsampled NIfTI file.
        vox_size_target (float, optional): Desired voxel size (e.g., 1.0 for 1mm).
        vox_size_target_max (bool, optional): If True, voxel size will not exceed target size.
                                              If False, voxel size will not be below target size.
        downsample_factor (int, optional): Integer factor to downsample by (e.g., 10 for 0.1mm → 1mm).
    
        Note: One of `downsample_factor` or `vox_size_target` **must** be specified.
        """
    
        # Load the NIfTI file
        nifti_image = nib.load(input_im_fpath)
        image_data = nifti_image.get_fdata()
        affine = nifti_image.affine
    
        # Extract current voxel size from affine matrix (diagonal of 3x3 rotation/scaling matrix)
        voxel_sizes = np.abs(np.diag(affine)[:3])  # Extract x, y, z voxel sizes
    
        if vox_size_target is not None:
            # Calculate the closest integer downsample factor
            if vox_size_target_max:
                downsample_factor = max(1, int(vox_size_target // min(voxel_sizes))) # Below target voxel size
            else:
                downsample_factor = max(1, int(np.ceil(vox_size_target / min(voxel_sizes)))) # Above target voxel size
    
        if downsample_factor is None:
            self.helpers.errors("Either 'downsample_factor' or 'voxel_size_target' must be specified.")
    
        # Compute zoom factor (inverse of downsampling factor)
        zoom_factor = 1 / downsample_factor
    
        # Apply downsampling
        downsampled_data = zoom(image_data, zoom_factor, order=3)  # order=3 for cubic interpolation
    
        # Adjust affine matrix to reflect new voxel size
        new_affine = affine.copy()
        new_affine[:3, :3] *= downsample_factor  # Scale voxel spacing
    
        # Save the downsampled NIfTI file
        new_nifti = nib.Nifti1Image(downsampled_data, new_affine, nifti_image.header)
        nib.save(new_nifti, outpath)
    
        return

    def upsample_nifti(self, input_im_fpath: str, outpath: str, 
                       reference_nii: str = None, 
                       vox_size_target: float = None, 
                       downsample_factor: int = None):
        """
        Upsample a NIfTI image back to its original voxel size.
    
        Parameters:
        input_im_fpath (str): Path to the input downsampled NIfTI (.nii.gz) file.
        outpath (str): Path for the upsampled NIfTI file.
        reference_nii (str, optional): Path to a reference NIfTI image to determine voxel size.
        vox_size_target (list, optional): Desired voxel sizes in list (e.g., [0.1, 0.1, 0.1].
        downsample_factor (int, optional): Factor the image was originally downsampled by.
    
        Note: One of `reference_nii`, `vox_size_target`, or `downsample_factor` **must** be specified.
        """
    
        # Load the downsampled NIfTI file
        nifti_image = nib.load(input_im_fpath)
        image_data = nifti_image.get_fdata()
        affine = nifti_image.affine
    
        # Extract voxel size from the downsampled image
        voxel_sizes = np.abs(np.diag(affine)[:3])  # Extract x, y, z voxel sizes
    
        # Determine upsampling factor
        if reference_nii:
            reference_image = nib.load(reference_nii)
            reference_voxel_sizes = np.abs(np.diag(reference_image.affine)[:3])
            upsample_factor = np.array(voxel_sizes) / np.array(reference_voxel_sizes)
        elif voxel_size_target:
            upsample_factor = np.array(voxel_sizes) / np.array(voxel_size_target)
        elif downsample_factor:
            upsample_factor = [downsample_factor, downsample_factor, downsample_factor]
        else:
            self.helpers.errors("One of 'reference_nii_path', 'voxel_size_target' or 'downsample_factor' must be specified.")
    
        # Apply upsampling
        upsampled_data = zoom(image_data, upsample_factor, order=3)  # order=3 for cubic interpolation
    
        # Adjust affine matrix to reflect new voxel size
        new_affine = affine.copy()
        new_affine[:3, :3] = affine[:3, :3] / upsample_factor  # Scale voxel spacing
    
        # Fix output filename
        output_path = os.path.join(self.path.basedir(outpath), f"{self.path.name(outpath)}_upsampled{self.path.ext(outpath)}")
    
        # Save the upsampled NIfTI file
        new_nifti = nib.Nifti1Image(upsampled_data, new_affine, nifti_image.header)
        nib.save(new_nifti, output_path)

        return

    def bin_image(self, input_im, output_path, max_dim, method="mean"):
        """
        Bin an image to a target shape by averaging or summing pixel groups.
    
        Parameters:
        input_im (str)    : Path to image to be binned
        output_path (str) : Path for binned image to be saved to
        max_dim (int)     : Maximum dimension for binned image
        method (str)      : "mean" for averaging values, "sum" for summing values
        """
        # Load image
        if input_im.endswith((".nii", ".nii.gz")):
            nii = nib.load(input_im)
            img = nii.get_fdata()
        elif input_im.endswith((".tif", ".tiff")):
            img = tifffile.imread(input_im)
        else:
            self.helpers.errors("Unsupported file type (.nii, .nii.gz, .tif, .tiff only)")

        # Caculate bin factor based on target dimensions
        img_shape = np.array(img.shape)
        scale_factors = np.array(img_shape) / max_dim  # Compute per-dimension scaling factors
        bin_factor = int(np.ceil(max(scale_factors)))

        # Crop slices to closest multiple of bin factor
        new_shape = [s - (s % bin_factor) for s in img.shape]  # Find nearest multiple
        crop_slices = tuple(slice(0, ns) for ns in new_shape)  # Create cropping slices
        img = img[crop_slices]
    
        # Reshape and bin
        reshaped = img.reshape(
            new_shape[0] // bin_factor, bin_factor,
            new_shape[1] // bin_factor, bin_factor,
            new_shape[2] // bin_factor, bin_factor
        )
    
        if method == "mean":
            img_binned = reshaped.mean(axis=(1, 3, 5))
        elif method == "sum":
            img_binned = reshaped.sum(axis=(1, 3, 5))
        else:
            self.helpers.errors("Method must be 'mean' or 'sum'")

        # Save the output
        if input_im.endswith((".nii", ".nii.gz")):
            new_nii = nib.Nifti1Image(img_binned, affine=nii.affine, header=nii.header)
            nib.save(new_nii, output_path)
        elif input_im.endswith(".tif"):
            tifffile.imwrite(output_path, img_binned.astype(img.dtype))
    
        return

    def fix_orientations(self, image, outpath, transposes=[], flips=[]):
        """
        Fix image orientations
    
        Parameters:
        image (str): Path to the input NIfTI (.nii.gz) file.
        outpath (str): Path to save reoriented image to.
        transposes (list, optional): List of tuples for transpose to perform, e.g. [(0, 1, 2), (2, 0, 1)].
        flips (list, optional): List of axis flips to perform, e.g. [1, 2]
        """
        # Load the image
        nii = nib.load(image)
        data = nii.get_fdata()
        affine = nii.affine
        header = nii.header
    
        # Perform transforms
        for transpose in transposes:
            reoriented_data = np.transpose(data, transpose)
        for flip in flips:
            reoriented_data = np.flip(reoriented_data, axis=flip)
        
        # Save new NIfTI file
        new_img = nib.Nifti1Image(reoriented_data, affine, header)
        nib.save(new_img, outpath)

        return

    def rescale_intensities(self, image, outpath, min_val = 0, max_val = 1000):
        """
        Rescale image intensities

        Parameters:
        image (str): Path to NIfTI image to scale intensities of
        outpath (str): Path to save rescaled image to
        min_val (int): Minimum value for image intensity range
        max_val (int): Maximum value for image intensity range
        """
        # Load images
        im = nib.load(image)
        data = im.get_fdata()
        img_nii = np.asarray(data).astype(np.float32)
        
        # Shift the intensity range if necessary
        if np.min(img_nii) < min_val:
            img_nii += (min_val - np.min(img_nii))
            
        # Rescale the intensity range
        img_range = np.max(img_nii) - np.min(img_nii)
        img_nii = (img_nii - np.min(img_nii)) / img_range * (max_val - min_val) + min_val
    
        # Save the resulting NIfTI image
        result_image = nib.Nifti1Image(img_nii, im.affine, im.header)
        nib.save(result_image, outpath)
        
        return

    def origin_reset(self, image, outpath, coords=None):
        """
        Set image origin to approx centre of brain

        Parameters:
        image (str): Path to NIfTI image to reset origin
        outpath (str): Path to save origin reset image to
        coords (list): List of x, y, z coordinates to set as the new origin
        """
        nii = nib.load(image)
        img = nii.get_fdata()
        squeezed = np.squeeze(img)
        aff = nii.affine
        
        # get affine without translation
        aff_out        = aff
        aff_out[0:3,3] = 0
        
        # Use provided coordinates or default to image center
        if coords is not None:
            origin_shift = -np.array(coords)
        else:
            origin_shift = -np.array(squeezed.shape) / 2.0
    
        # Apply shift
        aff_origin = np.identity(4)
        aff_origin[0:3, 3] = origin_shift
        aff_out = aff_out.dot(aff_origin)
        
        # create output 
        nib.Nifti1Image(np.asarray(squeezed).astype(np.float32), 
                        aff_out).to_filename(str(outpath))

        return


class Convert(BaseCog):
    """
    Image format conversion methods
    """
    def __init__(self, **kwargs):
        """
        Instantiate the Convert class.
        """
        super().__init__(**kwargs)

        # Instantiate custom modules
        self.helpers = Helpers()

    def convert_bit_depth(self, input_path, output_path, target_dtype=np.uint8, normalise=True):
        """
        Convert a TIFF image to a specified data type (e.g., 32-bit to 8-bit).
    
        Parameters:
        input_path (str or Path)    : Path to the input TIFF file
        output_path (str or Path)   : Path to save the converted TIFF file
        target_dtype (numpy dtype)  : Target data type (e.g., np.uint8, np.uint16)
        normalise (bool)            : Normalise the image to fit the new bit-depth
        """
        # Read the image
        img = tifffile.imread(input_path)
    
        # Normalise if required
        if normalise:
            img = img.astype(np.float32)  # Convert to float for safe normalisation
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:  # Avoid division by zero
                img = (img - img_min) / (img_max - img_min)  # Normalise to 0-1
            img = (img * np.iinfo(target_dtype).max).astype(target_dtype)  # Scale and convert
    
        else:
            img = img.astype(target_dtype)  # Direct conversion without normalisation
    
        # Save the new image
        tifffile.imwrite(output_path, img)

    def tiff_to_nii(self, input_path: str, header_info_path: str, output_file: str):
        """
        Convert a single 3D TIFF file or a folder of 2D slices to a compressed NIfTI (.nii.gz) file,
        using a JSON file for header metadata.
    
        Parameters:
        input_path (str)        : Path to the 3D TIFF file or folder containing TIFF slices.
        header_info_path (str)  : Path to JSON file containing manually specified header fields.
        output_file (str)       : Output NIfTI file path, including extension.
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            self.helpers.errors(f"TIFF path provided does not exist ({input_path})")
        
        # Load header info from JSON
        with open(header_info_path, 'r') as f:
            header_info = json.load(f)
    
        if os.path.isfile(input_path) and input_path.lower().endswith(('.tiff', '.tif')):
            # Handle single 3D TIFF file
            image_stack = tifffile.imread(input_path).astype(np.float32)
        
        elif os.path.isdir(input_path):
            # Handle folder of 2D TIFF slices
            tiff_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.tiff', '.tif'))])
            
            if not tiff_files:
                self.helpers.errors(f"No TIFF files found at {input_path}")
            
            # Load the first slice to determine image dimensions
            img_sample = Image.open(os.path.join(input_path, tiff_files[0]))
            width, height = img_sample.size
            
            # Allocate array for image stack
            image_stack = np.zeros((len(tiff_files), height, width), dtype=np.float32)
            
            # Load each slice into the stack
            for i, file_name in enumerate(tiff_files):
                file_path = os.path.join(input_path, file_name)
                img = Image.open(file_path)
                image_stack[i, :, :] = np.array(img, dtype=np.float32)
        
        else:
            self.helpers.errors("Input path must be a valid 3D TIFF file or a directory containing TIFF slices.")
    
        # Calculate voxel sizes and transformation values
        dimensions = image_stack.shape
        voxel_sizes = header_info.get("pixdim", [1, 1, 1])  # Extract only the spatial dimensions
        qoffset = [round((-dimensions[0] * voxel_sizes[0] / 2.0), 5),
                    round((-dimensions[1] * voxel_sizes[1] / 2.0), 5),
                    round((-dimensions[2] * voxel_sizes[2] / 2.0), 5)]
        srow_x = header_info.get("srow_x", [voxel_sizes[0], 0, 0, qoffset[0]])
        srow_y = header_info.get("srow_y", [0, voxel_sizes[1], 0, qoffset[1]])
        srow_z = header_info.get("srow_z", [0, 0, voxel_sizes[2], qoffset[2]])
        
        # Ensure no NaN or Inf values
        if not np.isfinite(image_stack).all():
            self.helpers.errors("TIFF image data contains invalid values (NaNs or Infs).")
        
        # Create an affine transformation matrix
        affine = np.eye(4)
        affine[:3, :3] = np.diag(voxel_sizes)
        affine[:3, 3] = qoffset
        
        # Create the NIfTI image
        nifti_image = nib.Nifti1Image(image_stack, affine=affine)
        header = nifti_image.header
        
        # Update header fields from JSON
        header["pixdim"][1:4] = voxel_sizes  # Voxel dimensions
        header["xyzt_units"] = header_info.get("xyzt_units", 2)  # Spatial unit (default: mm)
        header["qform_code"] = header_info.get("qform_code", 1)  # Use scanner-based coordinate system
        header["sform_code"] = header_info.get("sform_code", 1)  # Use scanner-based coordinate system
        header["qoffset_x"], header["qoffset_y"], header["qoffset_z"] = qoffset
        header["srow_x"], header["srow_y"], header["srow_z"] = srow_x, srow_y, srow_z
        
        # Save the NIfTI file
        try:
            nib.save(nifti_image, output_file)
        except Exception as e:
            self.helpers.errors(f"Compression failed, saving as uncompressed NIfTI: {str(e)}")
            uncompressed_output = output_file.replace(".nii.gz", ".nii")
            nib.save(nifti_image, uncompressed_output)
            self.helpers.errors(f"NIfTI file saved to {uncompressed_output}")
            return

    def nii_to_tiff(nii_file_path: str, output_tiff_path: str, reference_tiff: str = None):
        """
        Convert a NIfTI (.nii.gz) file to a single 3D TIFF file while matching reference TIFF metadata.
        
        Parameters:
        nii_file_path (str):    Path to the input NIfTI file.
        output_tiff_path (str): Path for the output 3D TIFF file.
        reference_tiff (str):   Path to a reference TIFF file to match metadata (optional).
        """
        # Load the NIfTI file
        nifti_image = nib.load(nii_file_path)
        image_data = nifti_image.get_fdata()
    
        # Default settings
        compression = "tiff_lzw"
        resolution = (1, 1)
    
        # Match the reference TIFF metadata if provided
        if reference_tiff:
            ref_img = Image.open(reference_tiff)
            ref_mode = ref_img.mode  # Example: "F", "I;16", "L"
    
            # Convert image data to match the reference mode
            if ref_mode == "I;16":  # 16-bit integer
                image_data = image_data.astype(np.uint16)
            elif ref_mode == "F":  # 32-bit floating point
                image_data = image_data.astype(np.float32)
            elif ref_mode == "L":  # 8-bit grayscale
                image_data = image_data.astype(np.uint8)
            else:
                self.helpers.errors(f"Unsupported TIFF mode: {ref_mode}")
    
            # Extract and match resolution from the reference TIFF
            # if "resolution" in ref_img.info:
            #     resolution = tuple(map(int, ref_img.info["resolution"]))  # Ensure it's stored as integers
    
            # Extract and match compression (default to 'raw' if not found)
            compression = ref_img.info.get("compression", "raw")
            if compression == "tiff_lzw":
                compression = "lzw"
    
        # Save as a 3D TIFF with matched metadata
        tifffile.imwrite(
            output_tiff_path, 
            image_data, 
            compression=compression
        )

        return
        
    def tiff_3d_to_2d(self, input_tiff: str, output_dir: str, prefix: str = "slice"):
        """
        Converts a 3D TIFF file into a stack of 2D TIFF images.
    
        Parameters:
        input_tiff (str): Path to the input 3D TIFF file.
        output_dir (str): Directory where 2D slices will be saved.
        prefix (str): Prefix for the output file names (default is "slice").
    
        """
        # Load the 3D TIFF image
        img_stack = tifffile.imread(input_tiff)
    
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
        # Save each slice as a separate 2D TIFF
        for i, slice_img in enumerate(img_stack):
            slice_path = os.path.join(output_dir, f"{prefix}_{i:04d}.tif")
            tifffile.imwrite(slice_path, slice_img)

        return

    def tiff_2d_to_3d(self, input_dir: str, output_tiff: str):
        """
        Converts a stack of 2D TIFF images into a single 3D TIFF file.
    
        Parameters:
        input_dir (str): Directory containing 2D TIFF slices.
        output_tiff (str): Path to save the output 3D TIFF file.
    
        """
        # Get all TIFF files in the directory, sorted in natural order (e.g., slice_0001.tif before slice_0010.tif)
        tiff_files = natsorted([f for f in os.listdir(input_dir) if f.lower().endswith(".tif")])
        
        if not tiff_files:
            self.helpers.errors(f"No TIFF files found in the directory {input_dir}")
            return
    
        # Load all 2D images into a 3D numpy array
        img_stack = [tifffile.imread(os.path.join(input_dir, f)) for f in tiff_files]
        img_stack = np.stack(img_stack, axis=0)  # Stack along the first axis
    
        # Save as a single 3D TIFF
        tifffile.imwrite(output_tiff, img_stack)

        return


class Plotting(BaseCog):
    """
    Image plotting methods
    """
    def __init__(self, **kwargs):
        """
        Instantiate the Plotting class.
        """
        super().__init__(**kwargs)

        # Instantiate custom modules
        self.helpers = Helpers()

    def plot_image(self, image, atlas, title, coords=(-10,-10,-10), reg_space=""):
        """
        Plot two images overlayed to allow assessment of registration

        Parameters:
        image (str): Path to primary image to plot
        atlas (str): Path to background image or atlas to plot
        title (str): Title for image plot
        coords (tuple): Coordinates for where to cut images
        """
        at_img = image.load_img(atlas)
        at_nii = image.get_data(at_img)
        shape = np.shape(at_nii)
        at_nii[(int(shape[0]/2)):, :, :] = 0
        at_nii[:, (int(shape[1]/2)):, :] = 0
        at_nii[:, :, (int(shape[2]/2)):] = 0
        at_overlay = image.new_img_like(at_img, at_nii)
        
        plotting.plot_stat_map(at_overlay, image, draw_cross=False, 
                               cut_coords=coords, cmap="ocean", 
                               title=title)

    def plot_image_comparison(self, img_paths, titles=["Image 1", "Image 2"]):
        """
        Plot two images overlayed to allow assessment of registration

        Parameters:
        img_paths (list): List of image filepaths
        titles (list): List of titles for plots
        """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        
        # Loop through the images and display them
        for i, img_path in enumerate(img_paths):
            plotting.plot_anat(img_path, axes=axes[i], display_mode='y', cut_coords=[0])
            axes[i].set_title(titles[i])