
import time
import random
import os
import glob
import cv2

import imageio 
import numpy as np
import tqdm
import torch
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from einops import rearrange

from configs.base_config import NetworkConfig
from model.network import NGPNeRFNetwork

from load_blender import LegoDataset
from ray_utils import get_rays
from rendering import render

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def main():
    # load GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("device is", device)
    # torch.cuda.empty_cache()
    # set seed
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set config params
    val_dir = 'results/'
    # check if val_dir exists, otherwise create it
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    # occupancy grid update configuration
    warmup_steps = 256#32#256
    update_interval = 16
    max_samples = 1024

    # load datasets (lego synthetic dataset)
    # train_dataset = LegoDataset(root_dir='./data/nerf_synthetic/lego',split='train',downsample=0.5).to(device)
    # Comment below 3 lines when running inference or testing
    train_dataset = LegoDataset(root_dir='./data/nerf_synthetic/lego',split='train').to(device)
    train_dataset.batch_size = 2048#4096#8192 #256 #4096#
    train_dataset.ray_sampling_strategy = 'all_images'#'same_image'#

    # Test dataset (uncomment when testing)
    # test_dataset = LegoDataset(root_dir='./data/nerf_synthetic/lego',split='test').to(device)

    # validation metrics
    val_psnr = PeakSignalNoiseRatio(data_range=1).to(device)
    val_ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
    
    # load model
    model_config = {
        'num_levels': NetworkConfig.num_levels,
        'coarse_resolution': NetworkConfig.coarse_resolution,
        'fine_resolution': NetworkConfig.fine_resolution,
        'table_feature_vec_size': NetworkConfig.table_feature_vec_size,
        'log2_table_size': NetworkConfig.log2_table_size,
        'aabb': NetworkConfig.aabb,
        'sh_degrees': NetworkConfig.sh_degrees,
        'sh_input_dim': NetworkConfig.sh_input_dim,            
        'scale': NetworkConfig.scale,
        'density_mlp_num_layers': NetworkConfig.density_mlp_num_layers,
        'density_mlp_layer_width': NetworkConfig.density_mlp_layer_width,
        'density_mlp_output_dim': NetworkConfig.density_mlp_output_dim,
        'color_mlp_num_layers': NetworkConfig.color_mlp_num_layers,
        'color_mlp_layer_width': NetworkConfig.color_mlp_layer_width
        }
    model = NGPNeRFNetwork(**model_config).to(device)
    

    # initiate the occupancy grid for the dataset (Not needed when running inference)
    model.mark_invisible_cells(train_dataset.K, train_dataset.poses, train_dataset.img_wh)
    

    # optimizer
    scaler = 2**19
    # grad_scaler = torch.cuda.amp.GradScaler(scaler)
    grad_scaler = torch.amp.GradScaler('cuda', init_scale=scaler)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-15)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=1e-2, eps=1e-15)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-15, fused=True) # fusedadam
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    # learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 18000, 1e-2/30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32000, 1e-2/30)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    # training loop
    num_epochs = 32001
    timer = time.time()
    for e in range(num_epochs):
        # start training mode
        model.train()

        # load a random data
        train_dataset.epoch_num=e
        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]
        

        direction = data['direction']
        pose = data['pose']
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if e % update_interval == 0:
                
                model.update_density_grid(
                    0.01 * max_samples / 3**0.5, # = 5.912
                    warmup=e < warmup_steps,
                )
                
            # get rays
            rays_o, rays_d = get_rays(direction, pose)

            # render image
            results = render(model, rays_o, rays_d)

            print("ray original rgb", data['rgb'].shape)
            loss = nn.functional.mse_loss(results['rgb'], data['rgb'])
            print("epoch: ", e)
            print("Loss: ", loss)
        

        # do backprop steps with the optimizer - 
        optimizer.zero_grad()
        # loss.backward()
        grad_scaler.scale(loss).backward()
        
        grad_scaler.step(optimizer)
        
        grad_scaler.update()
        
        scheduler.step()
        # display progress every 1000 epochs
        if e % 1000 == 0:
            elapsed_time = time.time() - timer
            with torch.no_grad():
                mse = nn.functional.mse_loss(results['rgb'], data['rgb'])
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | "
                f"step={e} | psnr={psnr:.2f} | "
                f"loss={loss:.6f} | "
                # number of rays
                f"rays={len(data['rgb'])} | "
                # ray marching samples per ray (occupied space on the ray)
                f"rm_s={results['rm_samples'] / len(data['rgb']):.1f} | "
                # volume rendering samples per ray
                # (stops marching when transmittance drops below 1e-4)
                f"vr_s={results['vr_samples'] / len(data['rgb']):.1f} | "
            )
    
    # # save model
    torch.save(
        model.state_dict(),
        os.path.join(val_dir, 'model9_final.pth'),
    )
    #################################################################################################################
    # load check point if available for testing/inference
    ckpt_path = os.path.join(val_dir, 'model9_final.pth')
    if ckpt_path:
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)
        print("Load checkpoint from %s" % ckpt_path)

    # test loop (or inference)
    progress_bar = tqdm.tqdm(total=len(test_dataset), desc=f'evaluating: ')
    with torch.no_grad():
        model.eval()
        w, h = test_dataset.img_wh
        directions = test_dataset.directions
        test_psnrs = []
        test_ssims = []
        for test_step in range(len(test_dataset)):
        # for test_step in range(1):
            progress_bar.update()
            test_data = test_dataset[test_step]

            rgb_gt = test_data['rgb']
            poses = test_data['pose']
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # get rays
                rays_o, rays_d = get_rays(directions, poses)
                
                # render image
                # results = render(model, rays_o[:1000, :], rays_d[:1000, :], test_time=True)
                results = render(model, rays_o, rays_d, test_time=True)
            
            rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
            rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
            # get psnr
            val_psnr(rgb_pred, rgb_gt)
            test_psnrs.append(val_psnr.compute())
            val_psnr.reset()
            # get ssim
            val_ssim(rgb_pred, rgb_gt)
            test_ssims.append(val_ssim.compute())
            val_ssim.reset()
            # print("saving the image", test_step)
            # save test image to disk
            # if test_step % 5 == 0:
            
            test_idx = test_data['img_idxs']
            
            rgb_pred = rearrange(results['rgb'].cpu().numpy(),'(h w) c -> h w c',h=h)
            rgb_pred = (rgb_pred * 255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(val_dir,f'{test_idx:03d}.png'),rgb_pred)
            imageio.imsave(os.path.join(val_dir,f'{test_idx:03d}_d.png'),depth)
        progress_bar.close()
        test_psnr_avg = sum(test_psnrs) / len(test_psnrs)
        test_ssim_avg = sum(test_ssims) / len(test_ssims)
        print(f"evaluation: psnr_avg={test_psnr_avg} | ssim_avg={test_ssim_avg}")
    
    
    
    # Code to create a gif and video from inference results

    # if (not hparams.no_save_test) and \
    #    hparams.dataset_name=='nsvf' and \
    #    'Synthetic' in hparams.root_dir: # save video
    # imgs = sorted(glob.glob(os.path.join(val_dir, '*.png')))
    # print(imgs[1::2])
    # imageio.mimsave(os.path.join(val_dir, 'rgb.mp4'),
    #                 [imageio.v2.imread(img) for img in imgs[::2]],
    #                 fps=30)
    # imageio.mimsave(os.path.join(val_dir, 'depth.mp4'),
    #                 [imageio.v2.imread(img) for img in imgs[1::2]],
    #                 fps=30)
    
    # w1 = imageio.v2.get_writer(os.path.join(val_dir, 'rgb.mp4'),format='FFMPEG', fps=30)
    # for img in imgs[::2]:
    #     w1.append_data(imageio.v2.imread(img))
    
    # w2 = imageio.v2.get_writer(os.path.join(val_dir, 'depth.mp4'),format='FFMPEG', fps=30)
    # for img in imgs[1::2]:
    #     w2.append_data(imageio.v2.imread(img))
    # imageio.v3.imwrite(os.path.join(val_dir, 'rgb.gif'),[imageio.v2.imread(img) for img in imgs[::2]], duration=50, loop=0)
    # imageio.v3.imwrite(os.path.join(val_dir, 'depth.gif'),[imageio.v2.imread(img) for img in imgs[1::2]], duration=50, loop=0)




if __name__ == '__main__':
    main()






