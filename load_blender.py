import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from einops import rearrange

from ray_utils import get_ray_directions


from torch.utils.data import Dataset
class LegoDataset(Dataset):
    def __init__(self, root_dir, split='train', downsample=1.0, read_meta=True):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.epoch_num = 0

        self.read_intrinsics()

        if read_meta:
            self.read_meta(split, True)
    
    def __len__(self):
        return len(self.poses)
    
    def to(self, device):
        self.rays = self.rays.to(device)
        self.poses = self.poses.to(device)
        self.K = self.K.to(device)
        self.directions = self.directions.to(device)
        return self
    
    def __getitem__(self, idx):
        if self.split.startswith('train'):
            if self.ray_sampling_strategy == 'all_images':
                img_idxs = torch.randint(0, len(self.poses), size=(self.batch_size,), device=self.rays.device)
            elif self.ray_sampling_strategy == 'same_image':
                img_idxs = [idx]
            
            # if self.epoch_num<6000:
            #     print("cropped_img",self.epoch_num)
            #     pix_idxs = torch.randint((self.img_wh[0]//3)*(self.img_wh[1]//3), (self.img_wh[0]-self.img_wh[0]//3)*(self.img_wh[1]-self.img_wh[1]//3), size=(self.batch_size,), device=self.rays.device)
            # else:
            #     pix_idxs = torch.randint(0, self.img_wh[0]*self.img_wh[1], size=(self.batch_size,), device=self.rays.device)
            #     print("full img")
            pix_idxs = torch.randint(0, self.img_wh[0]*self.img_wh[1], size=(self.batch_size,), device=self.rays.device)
            

            rays = self.rays[img_idxs, pix_idxs]
            sample = {
                'img_idxs': img_idxs,
                'pix_idxs': pix_idxs,
                'pose': self.poses[img_idxs],
                'direction': self.directions[pix_idxs],
                'rgb': rays[:, :3]
            }
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
             # if ground truth available
            if len(self.rays) > 0: 
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]

        return sample

    
    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms_{}.json").format(self.split), 'r') as f:
            meta = json.load(f)
        
        img_path = os.path.join(self.root_dir,f"{meta['frames'][0]['file_path']}.png")
        img = imageio.v2.imread(img_path).astype(np.float32) / 255.0
        w = int(img.shape[0] * self.downsample)
        h = int(img.shape[1] * self.downsample)
        self.img_wh = (w, h)

        camera_angle_x = float(meta['camera_angle_x'])
        fx = fy = .5 * w / np.tan(.5 * camera_angle_x) * self.downsample
        cx = w / 2.0
        cy = h / 2.0

        K = np.float32([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, K)

        # return meta
    
    def read_meta(self, split, blend_a):
        self.rays = []
        self.poses = []

        with open(os.path.join(self.root_dir, "transforms_{}.json").format(self.split), 'r') as f:
            frames = json.load(f)["frames"]

        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            img_path = os.path.join(self.root_dir,f"{frame['file_path']}.png")
            
            if not os.path.exists(img_path):
                continue
            
            img = imageio.v2.imread(img_path).astype(np.float32) / 255.0
            

            if img.shape[2] == 4:  # blend A to RGB
                if blend_a:
                    img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
                else:
                    img = img[..., :3] * img[..., -1:]

            img = cv2.resize(img, self.img_wh)
            img = rearrange(img, 'h w c -> (h w) c')
            self.rays += [img]

            c2w = np.array(frame['transform_matrix'])[:3, :4]
            c2w[:, 1:3] *= -1  # [right up back] to [right down front]
            self.poses += [c2w]
        
        if len(self.rays) > 0:
            self.rays = torch.FloatTensor(np.stack(self.rays))  # (N_images, hw, channels=3)
        # CHECK IF NUMPY CONVERSION IS CORRECT
        self.poses = torch.FloatTensor(np.array(self.poses)) # (N_images, 3, 4) np.array(self.poses)

    

def main():
    print("Testing")
    root_dir='./data/nerf_synthetic/lego'
    with open(os.path.join(root_dir, "transforms_{}.json").format('train'), 'r') as f:
        frames = json.load(f)["frames"]
    
    img_path = os.path.join(root_dir,f"{frames[5]['file_path']}.png")
    print(img_path)
    
    # if not os.path.exists(img_path):
    #     continue
        
    img = imageio.v2.imread(img_path).astype(np.float32) / 255.0
    blend_a=True
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]
    print(img[300:600,300:600,2])
    # img = cv2.resize(img, (400, 400))
    # cv2.imshow("image", img[100:600,150:650,:])
    cv2.imshow("image", img[300:500,300:500,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = LegoDataset(root_dir='./data/nerf_synthetic/lego',split='train').to(device)
    
    # dataset.batch_size = 8192
    # dataset.ray_sampling_strategy = 'all_images'

    # i = torch.randint(0, len(dataset), (1,)).item()
    # data = dataset[i]
    # print(data)


    # dataset.read_intrinsics()
    # print(dataset.img_wh)
    # dataset.read_meta('train', True)
    # poses = dataset.poses
    # rays = dataset.rays
    # print(rays.shape)
    # print(poses[0])
    
    # print(directions)

if __name__=='__main__':
    main()