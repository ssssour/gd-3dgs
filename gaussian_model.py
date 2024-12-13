#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from nadc.mlp_model import GaussianPredictorMLP
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass
#from tqdm import tqdm 
import torch.nn.functional as F
from scipy.spatial import KDTree
import random
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, cam_infos, spatial_lr_scale, mlp_path):
        # Determine the MLP model path based on the scene name
        if not os.path.exists(mlp_path):
            raise FileNotFoundError(mlp_path)

        # Load MLP model for predictions
        model = GaussianPredictorMLP(input_dim=3).cuda()  # Adjust input_dim as needed
        model.load_state_dict(torch.load(mlp_path))
        model.eval()

        # Predict positions from pcd points using the MLP
        with torch.no_grad():
            inputs = torch.tensor(pcd.points).float().cuda()  # Assuming `pcd.points` is (num_points, 3)
            predicted_positions = model(inputs).cpu().numpy()  # Shape: (num_points, 3)

        # Set spatial learning rate scale
        self.spatial_lr_scale = spatial_lr_scale

        # Use predicted positions as the first fused point cloud
        fused_point_cloud = torch.tensor(predicted_positions).float().cuda()

        # Use original PCD points as the second fused point cloud
        fused_point_cloud2 = torch.tensor(np.asarray(pcd.points)).float().cuda()

        # Combine the two point clouds
        combined_point_cloud = torch.cat([fused_point_cloud, fused_point_cloud2], dim=0)

        # Initialize colors for the two point clouds
        default_color1 =  0.3 *torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float).cuda()  # Default color
        default_color2 = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())           # Original colors
        combined_color = torch.cat([default_color1, default_color2], dim=0)

        # Compute features for the combined point cloud
        features = torch.zeros((combined_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = combined_color  # Set the DC component for combined colors
        features[:, 3:, 1:] = 0.0

        # Print combined point cloud details
        print("Number of points after combining:", combined_point_cloud.shape[0])

        # Compute scaling for the combined point cloud
        dist2 = torch.clamp_min(distCUDA2(combined_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((combined_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((combined_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # Initialize parameters with combined point cloud
        self._xyz = nn.Parameter(combined_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Exposure and camera information initialization
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, extra_dense=False, top_20_indices= None):
        """
        Densifies and prunes Gaussian points based on gradient magnitude and opacity.

        Parameters:
        - max_grad (float): Maximum gradient threshold for densification.
        - min_opacity (float): Minimum opacity threshold for pruning.
        - extent (float): Spatial extent of the scene.
        - max_screen_size (float): Maximum screen size for pruning.
        - radii (Tensor): Radii of each Gaussian point.
        - extra_dense (bool): If True, applies extra densification to high-gradient Gaussians.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii

        # Perform standard densification
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Apply extra densification if the flag is set
        if extra_dense:
            print("Applying extra densification for high-density areas.")
            self.densify_top_20_high_grad_gaussians(grads, grad_threshold=max_grad,  top_20_indices=top_20_indices,scene_extent = extent, radii=radii)

        # Pruning step based on opacity and screen size
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        self.prune_points(prune_mask)

        # Clear tmp_radii to free memory
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1



    
    def select_random_gaussians_and_compute_normals(self, batch_size=10000, k=3, subset_size=10000, chunk_size=5000):
        """
        Select a fixed subset of Gaussians for fast processing, and use GPU-based operations to compute normals.
        
        Parameters:
        - batch_size: Number of random points to select for normal computation.
        - k: Number of nearest neighbors to find for normal calculation.
        - subset_size: Subset size of `_xyz` used to limit `torch.cdist` memory usage.
        - chunk_size: Size of mini-batches within `torch.cdist` to further reduce memory load.
        """
        torch.manual_seed(42)
        with torch.no_grad():
            # Select a fixed subset of points randomly with a specific random seed
            random_indices = torch.randperm(self._xyz.shape[0], device=self._xyz.device)[:batch_size]
        
        random_gaussians = self._xyz[random_indices]

        # Sample subset from `self._xyz` for distance calculations to limit memory usage
        subset_indices = torch.randperm(self._xyz.shape[0], device=self._xyz.device)[:subset_size]
        subset_points = self._xyz[subset_indices]
        
        # Compute dist_matrix in mini-batches
        neighbors_indices = []
        for i in range(0, random_gaussians.shape[0], chunk_size):
            chunk = random_gaussians[i:i + chunk_size]
            dist_matrix_chunk = torch.cdist(chunk, subset_points, p=2)
            dist_matrix_chunk = torch.nan_to_num(dist_matrix_chunk, nan=1e6, posinf=1e6, neginf=1e6)
            _, neighbors_indices_chunk = torch.topk(dist_matrix_chunk, k=min(k, dist_matrix_chunk.shape[-1]), largest=False, dim=-1)
            neighbors_indices.append(neighbors_indices_chunk)
        
        neighbors_indices = torch.cat(neighbors_indices)
        
        # Get face points and compute normals
        face_points = subset_points[neighbors_indices]
        normals = torch.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0], dim=1)
        normals = F.normalize(normals, dim=1)

        torch.cuda.empty_cache()  # Clear unused GPU memory
        return normals, face_points, random_indices

    def select_fixed_gaussians_and_compute_normals(self, num_points=500000, k=3):
        """
        Select a fixed number of points and compute normals using their nearest neighbors.
        
        Parameters:
        - num_points: Number of points to select for processing.
        - k: Number of nearest neighbors used for normal calculation.
        """
        torch.manual_seed(42)
        with torch.no_grad():
            # Randomly select a fixed number of points
            selected_indices = torch.randperm(self._xyz.shape[0], device=self._xyz.device)[:num_points]
            selected_points = self._xyz[selected_indices]

            # Compute the distance matrix and find k-nearest neighbors
            dist_matrix = torch.cdist(selected_points, selected_points, p=2)
            dist_matrix = torch.nan_to_num(dist_matrix, nan=1e6, posinf=1e6, neginf=1e6)
            _, neighbors_indices = torch.topk(dist_matrix, k=k, largest=False, dim=-1)

            # Get the k-nearest neighbor points for each selected point
            neighbor_points = selected_points[neighbors_indices]  # Shape: (num_points, k, 3)

            # Compute normals using cross product of vectors formed by neighbors
            normals = torch.cross(
                neighbor_points[:, 1] - neighbor_points[:, 0],  # Vector 1
                neighbor_points[:, 2] - neighbor_points[:, 0],  # Vector 2
                dim=1
            )
            normals = F.normalize(normals, dim=1)  # Normalize the normal vectors

        return normals, neighbor_points, selected_indices
    
    def distance_and_orientation_loss(self, distance_weight=1.0, orientation_weight=1.0, batch_size=10000, k=3, max_batches=50):
        """
        Compute distance and orientation loss between randomly selected Gaussians and their neighbors.
        
        Parameters:
        - distance_weight: Weight for distance-based loss.
        - orientation_weight: Weight for orientation-based loss.
        - batch_size: Number of Gaussians to process per batch.
        - k: Number of nearest neighbors for orientation computation.
        - max_batches: Maximum number of batches to process to limit computation time.
        """
        total_loss = 0.0
        num_batches = min(len(self._xyz) // batch_size + int(len(self._xyz) % batch_size != 0), max_batches)

        for batch_idx in range(num_batches):
            normals, face_points, indices = self.select_random_gaussians_and_compute_normals(batch_size, k)
            gaussian_positions = self._xyz[indices]
            gaussian_normals = self._rotation[indices, :3]
            surface_positions = face_points.mean(dim=1)

            # Compute distance and orientation losses
            distance_losses = ((gaussian_positions - surface_positions) ** 2).sum(dim=1).sqrt()
            cos_similarities = F.cosine_similarity(gaussian_normals, normals)
            orientation_losses = 1 - cos_similarities

            batch_losses = distance_weight * distance_losses + orientation_weight * orientation_losses
            total_loss += batch_losses.sum()

        average_loss = total_loss / len(self._xyz)
        return average_loss
    def split_and_analyze_high_grad_gaussians(self, grad_threshold, grid_size=4):
        """
        Filter Gaussians by gradient threshold, split into a 3D grid, and calculate density ratio.

        Parameters:
        - grad_threshold (float): Threshold for filtering Gaussians by gradient magnitude.
        - grid_size (int): Number of divisions along each axis (default: 4 for a 4x4x4 grid).

        Returns:
        - density_ratio (float): Ratio of sum of Gaussians in 20 densest cells to 20 sparsest cells.
        - top_20_indices (list of int): Indices of the 20 densest cells.
        - top_20_count (int): Total count of high-gradient Gaussians in the top 20 densest cells.
        """

        # Step 1: Ensure gradients are available and calculate gradient magnitudes
        if self._xyz.grad is None:
            raise RuntimeError("Gradients not available. Ensure that backward() has been called.")
        
        grad_magnitudes = self._xyz.grad.norm(dim=1)  # L2 norm to get gradient magnitude for each Gaussian

        # Step 2: Filter Gaussians by gradient threshold
        high_grad_indices = (grad_magnitudes > grad_threshold).nonzero(as_tuple=True)[0]
        high_grad_gaussians = self._xyz[high_grad_indices]

        # If no Gaussians meet the threshold, return 0
        if high_grad_gaussians.size(0) == 0:
            print("No Gaussians with gradient above threshold.")
            return 0.0, [], 0

        # Step 3: Determine the bounds and initialize the grid
        min_coords, _ = high_grad_gaussians.min(dim=0)
        max_coords, _ = high_grad_gaussians.max(dim=0)
        grid = [[] for _ in range(grid_size ** 3)]  # Grid cells to store indices of high-grad Gaussians

        # Step 4: Compute the size of each cell
        cell_size = (max_coords - min_coords) / grid_size

        # Step 5: Place high-gradient Gaussians into grid cells
        for i, gaussian in enumerate(high_grad_gaussians):
            cell_indices = ((gaussian - min_coords) / cell_size).floor().to(torch.int32)
            cell_indices = torch.clamp(cell_indices, 0, grid_size - 1)
            linear_index = (cell_indices[0] * grid_size * grid_size) + (cell_indices[1] * grid_size) + cell_indices[2]
            grid[linear_index].append(i)

        # Step 6: Count high-gradient Gaussians in each cell
        cell_counts = [len(cell) for cell in grid]

        # Step 7: Identify 20 most and 20 least dense cells by count
        top_20_indices = sorted(range(len(cell_counts)), key=lambda x: cell_counts[x], reverse=True)[:20]
        bottom_20_indices = sorted(range(len(cell_counts)), key=lambda x: cell_counts[x])[:20]

        # Step 8: Calculate the total number of high-grad Gaussians in these cells
        top_20_sum = sum(cell_counts[i] for i in top_20_indices)
        bottom_20_sum = sum(cell_counts[i] for i in bottom_20_indices)

        # Step 9: Calculate density ratio (top 20 / bottom 20)
        density_ratio = top_20_sum / max(bottom_20_sum, 1)  # Avoid division by zero

        # Step 10: Calculate the total count of high-grad Gaussians in the top 20 cells
        top_20_count = top_20_sum

        return density_ratio, top_20_indices, top_20_count

      
    def densify_top_20_high_grad_gaussians(self, grads, grad_threshold, top_20_indices, scene_extent, radii):
        """
        Clones high-gradient Gaussians specifically in the top 20 densest sectors based on a gradient threshold and density extent.

        Parameters:
        - grads (Tensor): Gradient magnitudes for each Gaussian.
        - grad_threshold (float): Threshold for cloning based on gradient magnitude.
        - top_20_indices (list): Indices of the top 20 densest sectors.
        - scene_extent (float): Extent of the scene for density filtering.
        - radii (Tensor): Radii of each Gaussian point, used to determine spatial density.
        """

        def get_sector_mask(points, sector_id, grid_size=4):
            min_coords, _ = points.min(dim=0)
            max_coords, _ = points.max(dim=0)
            cell_size = (max_coords - min_coords) / grid_size
            x_idx, y_idx, z_idx = sector_id // (grid_size ** 2), (sector_id // grid_size) % grid_size, sector_id % grid_size
            sector_min = min_coords + cell_size * torch.tensor([x_idx, y_idx, z_idx], device=points.device)
            sector_max = sector_min + cell_size
            return ((points >= sector_min) & (points < sector_max)).all(dim=1)

        # Safeguard: Ensure grads, self._xyz, and radii have compatible sizes
        mask_size = min(self._xyz.shape[0], grads.shape[0], radii.shape[0])
        xyz = self._xyz[:mask_size]
        features_dc = self._features_dc[:mask_size]
        features_rest = self._features_rest[:mask_size]
        opacities = self._opacity[:mask_size]
        scaling = self._scaling[:mask_size]
        rotation = self._rotation[:mask_size]
        grads = grads[:mask_size]
        radii = radii[:mask_size]

        # Step 1: Mask for high gradients
        selected_pts_mask = torch.zeros(mask_size, dtype=torch.bool, device="cuda")
        grad_selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask[:grad_selected.shape[0]] = grad_selected

        # Step 2: Density filter based on scene extent
        density_filter = torch.zeros(mask_size, dtype=torch.bool, device="cuda")
        scaling_selected = torch.max(self.get_scaling[:mask_size], dim=1).values <= self.percent_dense * scene_extent
        density_filter[:scaling_selected.shape[0]] = scaling_selected
        selected_pts_mask &= density_filter

        # Step 3: Mask for top 20 densest sectors
        sector_mask = torch.zeros(mask_size, dtype=torch.bool, device="cuda")
        for sector_id in top_20_indices:
            sector_mask |= get_sector_mask(xyz, sector_id, grid_size=4)

        # Combine masks
        selected_pts_mask &= sector_mask

        # Step 4: Clone points based on the combined mask
        new_xyz = xyz[selected_pts_mask]
        new_features_dc = features_dc[selected_pts_mask]
        new_features_rest = features_rest[selected_pts_mask]
        new_opacities = opacities[selected_pts_mask]
        new_scaling = scaling[selected_pts_mask]
        new_rotation = rotation[selected_pts_mask]
        new_tmp_radii = radii[selected_pts_mask]

        # Step 5: Add cloned Gaussians to the model and optimizer
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
