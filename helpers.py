"""
Generic helper methods
"""
__author__   = ["ellis.langford.19@ucl.ac.uk"]
__modified__ = "29-Oct-2024"

# Imports
import os
import sys
import inspect
import datetime
import shutil
from pathlib import Path as Pathlib

from base_cog import BaseCog

class Path(BaseCog):
    """
    Generic path methods
    """
    def __init__(self, **kwargs):
        """
        Instantiate the Path class.
        """
        super().__init__(**kwargs)

    def name(self, fpath):
        """
        Returns filename
        """
        file = os.path.basename(fpath)
        ext = self.ext(fpath)
        filename = file.replace(ext, "")
        
        return filename

    def dir(self, fpath):
        """
        Returns directory name
        """
        file = os.path.dirname(fpath)
        filename = file.split("/")[-1]
        
        return filename

    def basedir(self, fpath):
        """
        Returns full path excluding filename
        """
        basedir = os.path.dirname(fpath)
        
        return basedir
        
    def ext(self, fpath):
        """
        Returns extension
        """
        file = os.path.basename(fpath)
        if file.endswith(".gz"):
            ext = file.split(".")[-2]
            full_ext = f".{ext}.gz"
        else:
            ext = file.split(".")[-1]
            full_ext = f".{ext}"
        
        return full_ext

    def name_ext(self, fpath):
        """
        Returns name extension
        """
        file = os.path.basename(fpath)
        if file.endswith(".gz"):
            ext = file.split(".")[-3:-1]
            full_ext = f".{ext[0]}.{ext[1]}"
        else:
            ext = file.split(".")[-1]
            full_ext = f".{ext}"
        
        return full_ext

    def glob(self, fpath, criteria):
        """
        Returns name extension
        """
        fpath = Pathlib(fpath)
        match = next(fpath.parent.glob(criteria))
        
        return str(match)
        
        
class Helpers(BaseCog):
    """
    Generic helper methods
    """
    def __init__(self, **kwargs):
        """
        Instantiate the Helpers class.
        """
        super().__init__(**kwargs)

    def now_time(self):
        """
        Returns formatted current date/time
        """
        now = datetime.datetime.now()
        now_formatted = now.strftime("%d-%m-%Y %H:%M")
        return now_formatted

    def tidy_up_logs(self):
        # Tidy up log files from previous runs
        for f in [self.log_fpath, self.errors_fpath]:
            if os.path.exists(f):
                os.remove(f)

    def plugin_log(self, msg, stamp=None, **kwargs):
        """
        Records messages into log file

        Parameters:
        msg        : Message to print/log
        stamp      : Include a custom stamp      
        """
        if stamp:
            log_msg = f"[ {stamp} | {self.now_time()} ] {msg}"
        else:
            log_msg = f"[ Log | {self.now_time()} ] {msg}"

        print(log_msg, end = "\n")

        # Check if running inside a Jupyter Notebook
        in_jupyter = "ipykernel" in sys.modules

        # Add to log file unless inside a Jupyter Notebook
        if self.log_fpath is not None and not in_jupyter:
            os.makedirs(self.log_dir, exist_ok=True)
            if os.path.exists(self.log_fpath):
                append_write = "a"
            else:
                append_write = "w"
            with open(self.log_fpath, append_write) as log_file:
                print(log_msg, file = log_file)

    def verbose_log(self, msg, **kwargs):
        """
        If config.VERBOSE == True, print msg.
        Otherwise, do not
        """
        if self.config["VERBOSE"]:
            self.plugin_log(msg, **kwargs)

    def notebook_log(self, msg, stamp=None, **kwargs):
        """
        Records messages into jupyter notebook

        Parameters:
        msg        : Message to print/log
        stamp      : Include a custom stamp      
        """
        if stamp:
            log_msg = f"[ {stamp} | {self.now_time()} ] {msg}"
        else:
            log_msg = f"[ Log | {self.now_time()} ] {msg}"

        print(log_msg, end = "\n")

    def errors(self, msg, stamp=None, **kwargs):
        """
        Log an error messages and exit

        Parameters:
        msg        : Error message to print/log
        stamp      : Include a custom stamp    
        """
        # Get error info
        caller = inspect.stack()
        caller_func = caller[1][3]
        caller_file = caller[1][1]
        
        # Create timestamp
        if stamp:
            err_msg = f"[ {stamp} | {self.now_time()} | Error {caller_func}({caller_file}) ] {msg}"
        else:
            err_msg = f"[ Error | {self.now_time()} | {caller_func}({caller_file}) ] {msg}"

        # Print error
        print(err_msg)

        # Check if running inside a Jupyter Notebook
        in_jupyter = "ipykernel" in sys.modules

        # Add to error log file unless inside a Jupyter Notebook
        if self.errors_fpath is not None and not in_jupyter:
            if os.path.exists(self.errors_fpath):
                append_write = "a"
            else:
                append_write = "w"
            with open(self.errors_fpath, append_write) as error_file:
                error_file.write("%s\n" % (err_msg))

        # Kill
        sys.exit(1)

    def log_options(self, params):
        """
        Record plugin inputs and parameters in an options.txt file
        """
        # Define and save the inputs values
        options_fpath = os.path.join(self.log_dir, "options.txt")
        if os.path.exists(options_fpath): 
            os.remove(options_fpath)
        
        # Iterate through parameters
        with open(options_fpath, "w") as options_file:  # Open in write mode to create the file
            for param_name, param_value in params.items():
                # Format the output line
                output_line = f"{param_name}={param_value}"
                options_file.write(f"{output_line}\n")

    def log_success(self):
        """
        Record success in results.txt file
        """
        # Define and save the inputs values
        results_fpath = os.path.join(self.base_dir, "results.txt")
        if os.path.exists(results_fpath): 
            os.remove(results_fpath)
        
        # Iterate through parameters
        with open(results_fpath, "w") as results_file:  # Open in write mode to create the file
            results_file.write(f"Plugin has executed successfully\n")