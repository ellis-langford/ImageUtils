"""
Generic setup methods
"""
__author__   = ["ellis.langford.19@ucl.ac.uk"]
__modified__ = "29-Oct-2024"

# Imports
import os
import json
import argparse
import importlib.util

class BaseCog:
    """
    Generic setup methods
    """
    def __init__(self, **kwargs):
        """
        Instantiate the BaseCog class.
        """
        # Define working directories and log files
        self.base_dir       = os.getcwd()
        self.log_dir        = os.path.join(self.base_dir, "logs")
        self.log_fpath      = os.path.join(self.log_dir, "log.txt")
        self.errors_fpath   = os.path.join(self.log_dir, "errors.txt")
        self.config_fpath   = "/app/core/config.py"

        # Load the configuration file
        self.config = self.load_config()

        # Dictionary to store resolved parameters
        self.parameters = {}

    def load_config(self):
        """
        Load the configuration file dynamically as a module.
        """
        if os.path.exists(self.config_fpath):
            # Dynamically import the config file as a module
            spec = importlib.util.spec_from_file_location("config", self.config_fpath)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
    
            # Return the loaded module's dictionary
            return {key: getattr(config, key) for key in dir(config) if not key.startswith("__")}

    def load_parameters(self, **kwargs):
        """
        Load parameters from the command line, properties file, or defaults in the config.
        If `props_fpath` is not provided or is missing, fall back to using only CLI arguments and defaults.
        """
        # Parse command-line arguments
        cli_args = self.parse_command_line()
    
        # Attempt to load the properties file if `props_fpath` is provided
        props_fpath = cli_args.get("props_fpath") or self.config["PARAMETERS"].get("props_fpath", {}).get("default")
        properties_data = {}
        if props_fpath:
            if os.path.exists(props_fpath):
                properties_data = self.load_properties_file(props_fpath)
            else:
                raise FileNotFoundError(f"Properties file '{props_fpath}' not found. Using CLI arguments and defaults.")
    
        # Merge parameters: CLI > Properties > Config Defaults
        self.parameters = {}
        for param, param_config in self.config["PARAMETERS"].items():
            # Resolve value: CLI > Properties file > Default
            value = cli_args.get(param)
            if value is None:
                value = properties_data.get(param)
            if value is None:
                value = param_config.get("default")
    
            # Convert string-based lists if necessary
            if param_config["type"] is list and isinstance(value, str):
                try:
                    value = json.loads(value)  # Convert string representation of a list into an actual list
                except json.JSONDecodeError:
                    raise ValueError(f"Parameter '{param}' should be a valid JSON list (e.g., '[2, 2]').")
    
            # Type checking
            if value is not None and not isinstance(value, param_config["type"]):
                raise ValueError(f"Parameter '{param}' must be type {param_config['type'].__name__}.")
    
            # If no value is provided and no default exists, raise an error
            if "default" not in param_config and value is None:
                raise ValueError(f"Parameter '{param}' is required but missing.")
    
            # Save the resolved value
            self.parameters[param] = value

    def str_to_bool(self, value):
        """
        Convert a string to a boolean value.
        Accepts 'true', 'false' (case insensitive), or raises an error for invalid values.
        """
        if isinstance(value, bool):
            return value
        if value.lower() in {'true', 't', 'yes', 'y', '1'}:
            return True
        elif value.lower() in {'false', 'f', 'no', 'n', '0'}:
            return False
        else:
            raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'")

    def parse_command_line(self):
        """
        Parse command-line arguments into a dictionary.
        """
        parser = argparse.ArgumentParser()
    
        # Dynamically add arguments based on the configuration
        for param_name, param_info in self.config["PARAMETERS"].items():
            param_type = param_info.get("type", str)
    
            # Handle boolean parameters explicitly
            if param_type is bool:
                parser.add_argument(f"--{param_name}", type=self.str_to_bool, required=False)
            # Handle list parameters (expects a JSON string like "[2, 2]" or "[[2, 2], [2, 2]]")
            elif param_type is list:
                parser.add_argument(f"--{param_name}", type=str, required=False)
            else:
                parser.add_argument(f"--{param_name}", type=param_type, required=False)
    
        # Parse known arguments
        args, _ = parser.parse_known_args()
        parsed_args = vars(args)
    
        # Convert JSON-like strings into actual lists
        for param_name, param_info in self.config["PARAMETERS"].items():
            if param_info["type"] is list and param_name in parsed_args and parsed_args[param_name]:
                try:
                    # Ensure input is JSON-parsable and starts with '[' and ends with ']'
                    if isinstance(parsed_args[param_name], str) and parsed_args[param_name].startswith("[") and parsed_args[param_name].endswith("]"):
                        parsed_args[param_name] = json.loads(parsed_args[param_name])  # Convert string to list
                        
                        # Validate that it's a list or list of lists
                        if not isinstance(parsed_args[param_name], list):
                            raise ValueError(f"Parameter '{param_name}' must be a list.")
                        
                        # Optional: Enforce list-of-lists constraint (if needed)
                        if any(isinstance(item, list) for item in parsed_args[param_name]):
                            if not all(isinstance(item, list) for item in parsed_args[param_name]):
                                raise ValueError(f"Parameter '{param_name}' must be a list of lists or a flat list, not mixed.")
    
                    else:
                        raise ValueError(f"Parameter '{param_name}' must be a valid JSON list (e.g., '[2, 2]' or '[[2, 2], [2, 2]]').")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format for '{param_name}': {parsed_args[param_name]}. Error: {e}")
    
        return parsed_args

    def load_properties_file(self, properties_fpath):
        """
        Load parameters from a properties JSON file.

        Parameters:
        - properties_fpath: str, Path to the JSON properties file.

        Returns:
        - A dictionary of parameters from the properties file.
        """
        if not os.path.exists(properties_fpath):
            raise FileNotFoundError(f"Properties file not found: {properties_fpath}")

        with open(properties_fpath, "r") as json_file:
            return json.load(json_file)
        
    def get_parameter(self, param_name):
        """
        Retrieve the value of a parameter from the resolved parameters.

        Parameters:
        - param_name: str, The name of the parameter to retrieve.

        Returns:
        - The value of the parameter, or None if not found.
        """
        return self.parameters.get(param_name)