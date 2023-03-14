import argparse
import os
import sys

def check_args(args):
    """
    perform basic sanity checks on arguments
    """
    if args.root_dir == None:   
        raise ValueError("No input file specified")

    if not os.path.exists(args.root_dir):
        raise ValueError(f"Root directory {args.root_dir} not found")

    return args 

def get_args(args_in):
    """
    parse command line arguments
    """

    argparser = argparse.ArgumentParser(
        description="Utility to parse IXRF log file for raw deadtime statistics"
    )

    #--------------------------
    #set up the expected args
    #--------------------------
    argparser.add_argument(
        "-d", "--root-dir", 
        help="Specify a root directory for the project"
        "Must contain directories Input_Images and Input_Masks"
        "containing the SEM images and exemplary masks, respectively",
        type=os.path.abspath,
    )

    args = argparser.parse_args(args_in)

    args = check_args(args)

    return args