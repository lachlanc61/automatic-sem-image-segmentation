import sys
import os

from datetime import datetime
import multiprocessing as mp

import auto_sem_segmentation.utils as utils
import auto_sem_segmentation.steps as steps

#--------------------------------------------------------------------------------
# GLOBALS
#--------------------------------------------------------------------------------
PACKAGE_CONFIG='auto_sem_segmentation/config.yaml'
#

#--------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------

def main(args_in):
    """
    MAIN 
    """

    print(f'Start: {datetime.now()}')

    MAINPATH=os.path.dirname(__file__)

    config = utils.YamlConfig(os.path.join(MAINPATH, PACKAGE_CONFIG))

    args = utils.get_args(args_in)

    config = config.get_derived(args)

    # Step 0: Configuration and setup
    mp.set_start_method('spawn')
    p0 = mp.Process(target=steps.start_step_0(config))
    p0.start()
    p0.join()
    
    # Step 1: Configure and train WGAN
    p1 = mp.Process(target=steps.start_step_1(config))
    p1.start()
    p1.join()
    
    # Step 2: Simulate masks
    p2 = mp.Process(target=steps.start_step_2(config))
    p2.start()
    p2.join()

    # Step 3: Train cycleGAN
    p3 = mp.Process(target=steps.start_step_3(config))
    p3.start()
    p3.join()

    # Step 4: Simulate fake images and segment real images with cycleGAN
    p4 = mp.Process(target=steps.start_step_4(config))
    p4.start()
    p4.join()

    # Step 5: Filter artifact particles
    p5 = mp.Process(target=steps.start_step_5(config))
    p5.start()
    p5.join()
    
    # Step 6: Train UNet and segment real images
    p6a = mp.Process(target=steps.start_step_6a(config))
    p6a.start()
    p6a.join()
 
    
    p6b = mp.Process(target=steps.start_step_6b(config))
    p6b.start()
    p6b.join()

    print(f'Finished: {datetime.now()}')


#--------------------------------------------------------------------------------
# RUN
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    main(sys.argv[1:])      

    sys.exit()




