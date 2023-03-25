import os

import auto_sem_segmentation.WassersteinGAN as WassersteinGAN
import auto_sem_segmentation.CycleGAN as CycleGAN
import auto_sem_segmentation.UNet_Segmentation as UNet_Segmentation
import auto_sem_segmentation.HelperFunctions as HelperFunctions


################################################################################################################
# Wrapper functions for starting subprocesses (workaround for problems with tensorflow not freeing GPU memory) #
################################################################################################################

def start_step_0(config):
    print('Step0: Configuring Devices, Initializing Directories, and Preparing Images...')
    HelperFunctions.initialize_directories(root_dir=config.ROOT_DIR, output_dir_cyclegan=config.OUTPUT_DIR_CYCLEGAN, output_dir_unet=config.OUTPUT_DIR_UNET)
    
    HelperFunctions.prepare_images_cycle_gan(root_dir=config.ROOT_DIR, input_dir_images=config.INPUT_DIR_IMAGES, \
            tile_size_w=config.TILE_SIZE_W, tile_size_h=config.TILE_SIZE_H, num_simulated_masks=config.NUM_SIMULATED_MASKS, \
            bg_threshold=config.bg_threshold)

def start_step_1(config):
    print('Step 1: Training WGAN...')
    wgan = WassersteinGAN.WGAN(root_dir=config.ROOT_DIR, allow_memory_growth=config.ALLOW_MEMORY_GROWTH, use_gpus_no=config.USE_GPUS_NO)
    wgan.batch_size = config.WGAN_BATCH_SIZE                       # Batch size during training
    wgan.epochs = config.WGAN_EPOCHS                               # Training epochs
    wgan.n_z = 128                                          # Noise vector size
    wgan.start_training()


def start_step_2(config):
    print('Step 2: Simulating fake masks...')
    num_masks = max(config.NUM_SIMULATED_MASKS, len(os.listdir(os.path.join(config.ROOT_DIR, '2_CycleGAN', 'data', 'trainA'))))
    w_gan = WassersteinGAN.WGAN(root_dir=config.ROOT_DIR, allow_memory_growth=config.ALLOW_MEMORY_GROWTH, use_gpus_no=config.USE_GPUS_NO)
        #new WGAN instance -> on simulate_masks, will load prior model from designated directory...
    w_gan.n_z = 128                                         # Noise vector size
    w_gan.simulate_masks(no_of_images=num_masks,            # No of fake masks to simulate
                         min_no_of_particles=config.MIN_PARTICLES_PER_TILE,   # Minimum number of particles per image tile (does not take overlaps into account)
                         max_no_of_particles=config.MAX_PARTICLES_PER_TILE,   # Maximum number of particles per image tile (does not take overlaps into account)
                         use_perlin_noise=True,             # Use Perlin Noise to simulate particle agglomeration/aggregation
                         perlin_noise_threshold=0.5,        # Threshold for Perlin Noise - higher values give smaller patches with more particle aggregation
                         perlin_noise_frequency=4,          # Determines the size and number of patches (higher values give more but smaller patches)
                         use_normal_distribution=True,      # Use a normal distribution to adjust particle size
                         use_random_rotation='DISABLE',     # 'DISABLE': Do not apply any additional rotation; 'RANDOM': Apply additional random rotation; 'PERLIN': Apply 'continuous', spatially correlated random rotations
                         img_width=config.TILE_SIZE_W,             # Width of the simulated images
                         img_height=config.TILE_SIZE_H)            # Height of the simulated images


def start_step_3(config):
    print('Step 3: Training CycleGAN...')
    cycle_gan = CycleGAN.CycleGAN(root_dir=config.ROOT_DIR, image_shape=(config.TILE_SIZE_H, config.TILE_SIZE_W, 1), \
            allow_memory_growth=config.ALLOW_MEMORY_GROWTH, use_gpus_no=config.USE_GPUS_NO)
    cycle_gan.batch_size = config.CYCLEGAN_BATCH_SIZE              # Batch size during training
    cycle_gan.epochs = config.CYCLEGAN_EPOCHS                      # Training epochs
    cycle_gan.use_data_loader = config.USE_DATALOADER              # Whether to use a dataloader
    cycle_gan.label_smoothing_factor = 0.0                  # Label smoothing factor - set to a small value (e.g., 0.1) to avoid overconfident discriminator guesses and very low discriminator losses (too strong discriminators can be problematic for generators due to the adverserial nature of GANs)
    cycle_gan.gaussian_noise_value = 0.15                   # Set to a small value (e.g., 0.15) to add Gaussian Noise to the discriminator layers (can help against mode collapse and "overtraining" the discriminator)
    cycle_gan.use_skip_connection = False                   # Add a skip connection between the input and output layer in the generator (conceptually similar to identity mapping)
    cycle_gan.start_training()


def start_step_4(config):
    print('Step 4: Generating fake training images and segmenting real images with CycleGAN...')
    # Generate fake images for training from simulated masks
    cycle_gan = CycleGAN.CycleGAN(root_dir=config.ROOT_DIR, image_shape=(config.TILE_SIZE_H, config.TILE_SIZE_W, 1), \
            allow_memory_growth=config.ALLOW_MEMORY_GROWTH, use_gpus_no=config.USE_GPUS_NO)
    cycle_gan.run_inference(files=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'data', 'trainB'),
                            output_directory=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'A'),
                            source_domain='B',
                            tile_images=False,
                            use_gpu=True)

    # Segment real images with CycleGAN
    cycle_gan.image_shape = (config.TILE_SIZE_W, config.TILE_SIZE_H)          # Size of tiles (when not running inference on the whole image)
    cycle_gan.run_inference(files=config.INPUT_DIR_IMAGES,
                            output_directory=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'B'),
                            source_domain='A',
                            tile_images=not config.RUN_INFERENCE_ON_WHOLE_IMAGE,
                            min_overlap=2,
                            manage_overlap_mode=2,
                            use_gpu=config.use_gpu_for_inference)


def start_step_5(config):
    print('Step 5: Filtering Artifacts in CycleGAN Masks...')
    HelperFunctions.filter_gan_masks(img_path=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'A'),
                                     msk_path=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'data', 'trainB'),
                                     out_path=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'Synthetic_Masks_Filtered'),
                                     do_watershed_and_four_connectivity=False)

    HelperFunctions.filter_gan_masks(img_path=config.INPUT_DIR_IMAGES,
                                     msk_path=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'B'),
                                     out_path=config.OUTPUT_DIR_CYCLEGAN,
                                     do_watershed_and_four_connectivity=True)


def start_step_6a(config):
    print('Step 6.a: Train MultiRes UNet...')
    u_net = UNet_Segmentation.UNet(root_dir=config.ROOT_DIR, image_dir=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'A'), \
            mask_dir=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'Synthetic_Masks_Filtered'), \
            allow_memory_growth=config.ALLOW_MEMORY_GROWTH, use_gpus_no=config.USE_GPUS_NO)
    u_net.batch_size = config.UNET_BATCH_SIZE                                       # Batch size during training
    u_net.epochs = config.UNET_EPOCHS                                               # Training epochs
    u_net.use_dataloader = config.USE_DATALOADER                                    # Whether to use a dataloader
    u_net.filters = config.UNET_FILTERS                                             # Filters in the first layer of the UNet
    u_net.contrast_optimization_range = config.UNET_CONTRAST_OPTIMIZATION_RANGE     # Contrast optimization range (can be used to remove "hot" and "cold" pixels)
    u_net.run_training()


def start_step_6b(config):
    print('Segment real images with UNet')
    u_net = UNet_Segmentation.UNet(root_dir=config.ROOT_DIR, image_dir=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'A'), \
                mask_dir=os.path.join(config.ROOT_DIR, '2_CycleGAN', 'generate_images', 'Synthetic_Masks_Filtered'), \
                allow_memory_growth=config.ALLOW_MEMORY_GROWTH, use_gpus_no=config.USE_GPUS_NO)
    u_net.use_dataloader = config.USE_DATALOADER                                    # Whether to use a dataloader
    u_net.filters = config.UNET_FILTERS                                             # Filters in the first layer of the UNet
    u_net.image_shape = (config.TILE_SIZE_W, config.TILE_SIZE_H)                           # Size of tiles (when not running inference on the whole image)
    u_net.contrast_optimization_range = config.UNET_CONTRAST_OPTIMIZATION_RANGE     # Contrast optimization range (can be used to remove "hot" and "cold" pixels)
    u_net.run_inference(files=config.INPUT_DIR_IMAGES,                              # Directory with images to segment
                        output_directory=config.OUTPUT_DIR_UNET,                    # Output directory for segmented images
                        tile_images=not config.RUN_INFERENCE_ON_WHOLE_IMAGE,        # Whether to tile images
                        threshold=-1,                                        # Threshold applied to segmentation masks (value between [0, 1]; if set to < 0, Otsu thresholding is used)
                        watershed_lines=True,                                # Whether watershed should be done (should usually be enabled, but can be disabled if a lot of oversegmentation occurs)
                        min_distance=9,                                      # Minimum distance that will be split by watershed lines (increase if oversegmentation occurs))
                        min_overlap=2,                                       # Minimum overlap between image tiles
                        manage_overlap_mode=2,                               # What to do in overlapping tile regions (0: Use Maximum, 1: Average, 2: Crop)
                        use_gpu=config.use_gpu_for_inference)                       # Whether to use the GPU during inference
#                        model='/home/lachlan/CODEBASE/SEM_segmentation/Testing/3_UNet/Models/2023-02-25_23-13-47')
