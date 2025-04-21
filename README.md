# instant-nerf-pytorch
This is a pytorch and cuda implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp) from NVIDIA. This implementation is for synthetic NeRF Lego dataset and may not work on real world datasets.

## Steps to run
- Clone the repo and in it install all the relevant python packages
- Install pytorch extension of [tiny-cuda-nn](https://github.com/nvlabs/tiny-cuda-nn#pytorch-extension) (optional: for faster training)
- Install and build custom cuda extension:
```python3
pip install instant-ngp/
```
- Run the training on lego dataset
```python3
python train.py
```
- For running inference, load the model checkpoint, comment out the training code and run the code.
### Testing setup
- Ubuntu 22.04 with torch 2.5.1+cu124 with CUDA 12.6 on a NVIDIA GeForce GTX 1050 Ti 
## Options
- To run without tcnn, set `self.use_tcnn = False` in `model/network.py`.
## Results
Model trained for 32000 epochs on full scale images of lego dataset:
<p float="middle">
	<img src="results/rgb.gif" width="250" height="250" title="output1"/> 
	<img src="results/depth.gif" width="250" height="250" title="result1"/>
</p>

## To Do
- [ ] create a requirements.txt file with all required packages.
```
torch
cv2
numpy
imageio
einops
kornia
torch_scatter
glob
torchmetrics
tqdm
matplotlib
```
## References and Acknowledgement

- [https://github.com/ashawkey/torch-ngp](https://github.com/ashawkey/torch-ngp)

- [https://github.com/kwea123/ngp_pl](https://github.com/kwea123/ngp_pl)

- [https://github.com/taichi-dev/taichi-nerfs](https://github.com/taichi-dev/taichi-nerfs)

- [https://github.com/nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)