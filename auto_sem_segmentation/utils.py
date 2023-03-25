import argparse
import os
import sys
import yaml

import auto_sem_segmentation.HelperFunctions as HelperFunctions

class YamlConfig():
    """
    stores control flags and variables from config.yaml
    """    
    def __init__(self, pkgconfig: str):
        """
        initialise config from yaml file

            hardcoded as "config.yaml" in [project_root]/auto_sem_segmentation/

        """

        #get project directory from current file (ie. auto_sem*/utils.py)
        directory = os.path.dirname(os.path.realpath(__file__))
        directory=os.path.dirname(directory)    #up one directory to reach project root       

        with open(pkgconfig, 'r') as f:

            cfg = yaml.safe_load(f)

            self.INPUT_SUBDIR_MASKS = str(cfg['INPUT_SUBDIR_MASKS'])
            self.INPUT_SUBDIR_IMAGES = str(cfg['INPUT_SUBDIR_IMAGES'])
            self.INPUT_SUBDIR_BG = str(cfg['INPUT_SUBDIR_BG'])

            self.OUTPUT_SUBDIR_CYCLEGAN = str(cfg['OUTPUT_SUBDIR_CYCLEGAN'])
            self.OUTPUT_SUBDIR_UNET = str(cfg['OUTPUT_SUBDIR_UNET'])
            self.TILE_SIZE_W = int(cfg['TILE_SIZE_W'])
            self.TILE_SIZE_H = int(cfg['TILE_SIZE_H'])
            self.NUM_SIMULATED_MASKS = int(cfg['NUM_SIMULATED_MASKS'])
            self.RUN_INFERENCE_ON_WHOLE_IMAGE = bool(cfg['RUN_INFERENCE_ON_WHOLE_IMAGE'])

            # Options for training and inference on a GPU
            self.USE_GPUS_NO = cfg['USE_GPUS_NO']      
            self.USE_GPU_FOR_WHOLE_IMAGE_INFERENCE = bool(cfg['USE_GPU_FOR_WHOLE_IMAGE_INFERENCE'])
            self.ALLOW_MEMORY_GROWTH = bool(cfg['ALLOW_MEMORY_GROWTH'])

            # Options for Training the Networks
            self.WGAN_BATCH_SIZE = int(cfg['WGAN_BATCH_SIZE'])
            self.WGAN_EPOCHS = int(cfg['WGAN_EPOCHS'])
            self.CYCLEGAN_BATCH_SIZE = int(cfg['CYCLEGAN_BATCH_SIZE'])
            self.CYCLEGAN_EPOCHS = int(cfg['CYCLEGAN_EPOCHS'])
            self.UNET_BATCH_SIZE = int(cfg['UNET_BATCH_SIZE'])
            self.UNET_EPOCHS = int(cfg['UNET_EPOCHS'])
            self.UNET_CONTRAST_OPTIMIZATION_RANGE = cfg['UNET_CONTRAST_OPTIMIZATION_RANGE']
            self.UNET_FILTERS = int(cfg['UNET_FILTERS'])
            self.USE_DATALOADER = bool(cfg['USE_DATALOADER'])

            # DEFAULTS
            self.BG_DEFAULT = float(cfg['BG_DEFAULT'])
            self.MIN_PARTICLES_PER_TILE = int(cfg['MIN_PARTICLES_PER_TILE'])
            self.MAX_PARTICLES_PER_TILE = int(cfg['MAX_PARTICLES_PER_TILE'])

            #INIT NONE
            self.ROOT_DIR = None
            self.INPUT_DIR_MASKS = None
            self.INPUT_DIR_IMAGES = None
            self.INPUT_DIR_BG = None
            self.OUTPUT_DIR_CYCLEGAN = None
            self.OUTPUT_DIR_UNET = None

    def get_derived(self, args):
        """
        calculate all derived flags from config+args
            including directory structure
        """        

        self = self.allocate_dirs(args.root_dir)
        
        self.use_gpu_for_inference = not self.RUN_INFERENCE_ON_WHOLE_IMAGE or (self.USE_GPU_FOR_WHOLE_IMAGE_INFERENCE and self.RUN_INFERENCE_ON_WHOLE_IMAGE)

        #if example backgrounds are included, use them, otherwise use BG_DEFAULT
        if os.path.isdir(self.INPUT_DIR_BG) and not os.listdir(self.INPUT_DIR_BG) == [] :
            self.bg_threshold=HelperFunctions.get_background_level(input_dir_bg=self.INPUT_DIR_BG)
        else:
            self.bg_threshold=self.BG_DEFAULT

        return self
        

    def allocate_dirs(self, root_dir):
        """
        assign directories relative to root dir

        perform sanity checks
        """

        if os.path.exists(root_dir):
            self.ROOT_DIR = root_dir
        else:
            raise ValueError(f"Root directory {root_dir} not found")
            
        #directories
        self.INPUT_DIR_MASKS = os.path.join(self.ROOT_DIR, self.INPUT_SUBDIR_MASKS)
        self.INPUT_DIR_IMAGES = os.path.join(self.ROOT_DIR, self.INPUT_SUBDIR_IMAGES)
        self.INPUT_DIR_BG = os.path.join(self.ROOT_DIR, self.INPUT_SUBDIR_BG)
        self.OUTPUT_DIR_CYCLEGAN = os.path.join(self.ROOT_DIR, self.OUTPUT_SUBDIR_CYCLEGAN)
        self.OUTPUT_DIR_UNET = os.path.join(self.ROOT_DIR, self.OUTPUT_SUBDIR_UNET)

        return self



def check_args(args):
    """
    perform basic sanity checks on arguments
    """
    if args.root_dir == None:   
        raise ValueError("No input directory specified")

    if not os.path.exists(args.root_dir):
        raise ValueError(f"Root directory {args.root_dir} not found")

    return args 

def get_args(args_in):
    """
    parse command line arguments
    """

    argparser = argparse.ArgumentParser(
        description=""
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