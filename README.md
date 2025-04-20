# instant-nerf-pytorch
This is a pytorch and cuda implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp) from NVIDIA. This implementation is for synthetic NeRF Lego dataset and may not work on real world datasets.

## Steps to run
- Clone the repo and enter it.
- Install all the relevant python packages
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
add drive link to trained model

## To DO
create a requirements.txt file with all required packages
## References and Acknowledgement

- [https://github.com/ashawkey/torch-ngp](https://github.com/ashawkey/torch-ngp)

- [https://github.com/kwea123/ngp_pl/tree/master](https://github.com/kwea123/ngp_pl/tree/master)

- [https://github.com/taichi-dev/taichi-nerfs](https://github.com/taichi-dev/taichi-nerfs)

- [https://github.com/nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)