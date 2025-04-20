import torch
import torch.nn as nn
import numpy as np



class hash_encoding(nn.Module): # since the values of the hash table at each index which are also called feature vectors are learnable parameters for that scene.
    def __init__(self, num_levels, log2_table_size, table_feature_vec_size, coarse_resolution, fine_resolution):
        super(hash_encoding, self).__init__()
        # Levels
        self.num_levels = num_levels
        self.levels = torch.arange(self.num_levels)

        # Resolution
        self.coarse_resolution = coarse_resolution
        self.fine_resolution = fine_resolution
        self.b = np.exp((np.log(self.fine_resolution) - np.log(self.coarse_resolution))/(self.num_levels - 1)) # growth factor

        self.res = torch.floor(self.coarse_resolution*self.b**self.levels) # scalings (num_levels,)
        
        
        # Hash tables at each level
        self.log2_table_size = log2_table_size
        self.table_feature_vec_size = table_feature_vec_size

        self.hash_table = torch.rand(size=(2**self.log2_table_size * self.num_levels,self.table_feature_vec_size))*2 - 1
        self.hash_table_init_scale = 0.001
        self.hash_table *= self.hash_table_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

    def get_out_dim(self) -> int:
        return self.num_levels * self.table_feature_vec_size
    
    def hash_fn(self, inp_tensor):#(num_levels, 3)
        inp_tensor = inp_tensor * torch.tensor([1, 2654435761, 805459861]).to(inp_tensor.device)
        x = torch.bitwise_xor(inp_tensor[..., 0], inp_tensor[..., 1])
        x = torch.bitwise_xor(x, inp_tensor[..., 2])

        x %= 2**self.log2_table_size

        x += (self.levels*2**self.log2_table_size).to(x.device)
        return x

    def forward(self, inp_points): # forward pass through the network
        # ray_point = (Ntotal_points, 3)
        num_points = inp_points.shape[0]
        
        inp_points = inp_points[..., None, :] # Change shape to [Ntotal_points, 1, 3]
        
        scaled = inp_points * self.res.view(-1, 1).to(inp_points.device) # self.res.view(-1, 1) - shape is (num_levels, 1) and shape of scaled is (Ntotal_points, num_levels, 3)
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)
        # print(scaled_c.shape)
        offset = scaled - scaled_f # (Ntotal_points, num_levels, 3)

        # For all 8 voxel corners
        hashed_0 = self.hash_fn(scaled_c)  # [Ntotal_points, num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [Ntotal_points, num_levels, features_per_level or 2]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        # Trilinear interpolation
        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (1 - offset[..., 2:3])  # [Ntotal_points, num_levels, features_per_level]
        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [Ntotal_points, num_levels * features_per_level]

class sh_encoding(nn.Module):
    def __init__(self, degree, input_dim):
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        self.output_dim = self.degree**2
    
    def forward(self, dirs):
        # inputs - (N_total_points)
        output = torch.empty((*dirs.shape[:-1], self.output_dim), dtype=torch.float32, device=dirs.device)
        x = dirs[..., 0]
        y = dirs[..., 1]
        z = dirs[..., 2]

        # precompute terms
        xy, xz, yz  = x*y, x*z, y*z
        x2, y2, z2 = x*x, y*y, z*z
        x4, y4, z4 = x2*x2, y2*y2, z2*z2
        x6, y6, z6 = x4*x2, y4*y2, z4*z2

        output[...,0] = 0.28209479177387814
        if self.degree > 1:
            output[..., 1] = -0.48860251190291987*y # -sqrt(3)*y/(2*sqrt(pi))
            output[..., 2] = 0.48860251190291987*z # sqrt(3)*z/(2*sqrt(pi))
            output[..., 3] = -0.48860251190291987*x # -sqrt(3)*x/(2*sqrt(pi))
        if self.degree > 2:
            output[..., 4] = 1.0925484305920792*xy # sqrt(15)*xy/(2*sqrt(pi))
            output[..., 5] = -1.0925484305920792*yz # -sqrt(15)*yz/(2*sqrt(pi))
            output[..., 6] = 0.94617469575755997*z2 - 0.31539156525251999 #  sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
            output[..., 7] = -1.0925484305920792*xz # -sqrt(15)*xz/(2*sqrt(pi))
            output[..., 8] = 0.54627421529603959*x2 - 0.54627421529603959*y2 # sqrt(15)*(x2 - y2)/(4*sqrt(pi))
        if self.degree > 3:
            output[..., 9] = 0.59004358992664352*y*(-3.0*x2 + y2) # sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
            output[..., 10] = 2.8906114426405538*xy*z # sqrt(105)*xy*z/(2*sqrt(pi))
            output[..., 11] = 0.45704579946446572*y*(1.0 - 5.0*z2) # sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
            output[..., 12] = 0.3731763325901154*z*(5.0*z2 - 3.0) # sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
            output[..., 13] = 0.45704579946446572*x*(1.0 - 5.0*z2) # sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
            output[..., 14] = 1.4453057213202769*z*(x2 - y2) # sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
            output[..., 15] = 0.59004358992664352*x*(-x2 + 3.0*y2) # sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
        
        return output
            
        








def main():
    print("\nTesting")

    h_enc = hash_encoding(16, 14, 2, (torch.tensor([-100, -100, -100]), torch.tensor([100, 100, 100])), 16, 512)

    print(h_enc.forward(torch.tensor([1, 2, 3])))

    sh_enc = sh_encoding(4, 3)
    print(sh_enc.forward(torch.tensor([1.0, 2.0, 3.0])))



if __name__ == "__main__":
    main()