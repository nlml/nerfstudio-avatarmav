# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


import os
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch import functional as F
from torch import nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.avatarmav import MLP as AvatarMavMLP
from nerfstudio.field_components.avatarmav import get_embedder
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def _so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    FROM PYTORCH3D

    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R


class HeadModule(nn.Module):
    def __init__(
        self,
        exp_dim=32,
        feature_dim=4,
        feature_res=64,
        deform_bs_res=32,
        deform_bs_dim=2,
        deform_linear_dims=[54, 128, 3],
        density_linear_dims=[140, 128, 1],
        color_linear_dims=[167, 128, 3],
        interp_level=3,
        embedding_freq=4,
        deform_bbox=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        feature_bbox=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        noise=0.0,
        deform_scale=0.1,
    ):
        super(HeadModule, self).__init__()
        self.exp_dim = exp_dim
        self.feature_dim = feature_dim
        self.feature_res = feature_res
        self.deform_bs_res = deform_bs_res
        self.deform_bs_dim = deform_bs_dim
        self.deform_linear_dims = deform_linear_dims
        self.density_linear_dims = density_linear_dims
        self.color_linear_dims = color_linear_dims
        self.interp_level = interp_level
        self.embedding_freq = embedding_freq
        self.deform_bbox = deform_bbox
        self.feature_bbox = feature_bbox
        self.noise = noise
        self.deform_scale = deform_scale

        self.deform_bs_volume = nn.Parameter(
            torch.zeros(
                [
                    self.exp_dim,
                    self.deform_bs_dim,
                    self.deform_bs_res + 1,
                    self.deform_bs_res + 1,
                    self.deform_bs_res + 1,
                ]
            )
        )
        self.deform_mean_volume = nn.Parameter(
            torch.zeros([self.deform_bs_dim, self.deform_bs_res + 1, self.deform_bs_res + 1, self.deform_bs_res + 1])
        )
        self.deform_linear = AvatarMavMLP(self.deform_linear_dims)

        self.feature_volume = nn.Parameter(
            torch.zeros([self.feature_dim, self.feature_res + 1, self.feature_res + 1, self.feature_res + 1])
        )
        self.density_linear = AvatarMavMLP(self.density_linear_dims)
        self.color_linear = AvatarMavMLP(self.color_linear_dims)

        self.interp_level = self.interp_level

        self.feat_embedding, self.feat_out_dim = get_embedder(self.embedding_freq)
        self.deform_embedding, self.deform_out_dim = get_embedder(self.embedding_freq)
        # TODO(LS): re-enable viewdir dependence!
        self.view_embedding, self.view_out_dim = get_embedder(self.embedding_freq, disable_viewdir_dependence=True)

    def _rot_trans_from_pose(self, pose):
        R = _so3_exp_map(pose[:, :3])
        T = pose[:, 3:, None]
        return R, T

    def density(self, query_pts, exp, pose=None, scale=None):
        """
        Compute density values for query points.

        Args:
            query_pts: A tensor of shape `(num_cameras, 3, num_rays_per_camera * num_samples)`
                containing the 3D points to query the density at.
            exp: A tensor of shape `(num_cameras, exp_dim)` containing the expression parameters.
            pose: A tensor of shape `(num_cameras, 6)` containing the camera pose offset parameters.
                These are in axis-angle format of [rx, ry, rz, tx, ty, tz].
            scale: A tensor of shape `(num_cameras, 1)` containing the camera scale offset parameters.

        Returns:
            A tensor of shape `(num_cameras, 1, num_rays_per_camera * num_samples)` containing
            the density values for the query points.
        """
        B, C, N = query_pts.shape
        if pose is not None:
            R, T = self._rot_trans_from_pose(pose)
            if scale is not None:
                S = scale[:, :, None]
            else:
                S = 1.0
            query_pts = torch.bmm(R.permute(0, 2, 1), (query_pts - T)) / S

        exp_expanded = exp[:, : self.exp_dim, None, None, None, None]
        # exp_expanded.shape should be [nCams, expDim, 1, 1, 1, 1]
        deform_bs_volume = self.deform_bs_volume.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1)
        # deform_bs_volume.shape should be [nCams, expDim, deformBsDim=2, deformBsRes + 1, deformBsRes + 1, deformBsRes + 1]
        deform_mean_volume = self.deform_mean_volume.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        # deform_mean_volume.shape should be [nCams, deformBsDim=2, deformBsRes + 1, deformBsRes + 1, deformBsRes + 1]
        deform_volume = (exp_expanded * deform_bs_volume).sum(1) + deform_mean_volume
        # deform_volume.shape should be [nCams, deformBsDim=2, deformBsRes + 1, deformBsRes + 1, deformBsRes + 1]
        deform = self.mult_dist_interp(query_pts, deform_volume, self.deform_bbox)

        deform_embedding = self.deform_embedding(rearrange(deform, "b c n -> (b n) c"))
        offset = rearrange(self.deform_linear(deform_embedding), "(b n) c -> b c n", b=B)
        deformed_pts = offset * self.deform_scale + query_pts
        feature = self.mult_dist_interp(
            deformed_pts, self.feature_volume.unsqueeze(0).repeat(B, 1, 1, 1, 1), self.feature_bbox
        )
        feature_embedding = self.feat_embedding(rearrange(feature, "b c n -> (b n) c"))
        # TODO(LS): Maybe shouldn't feed exp to density_linear to prevent overfitting....
        exp = rearrange((exp / 3)[:, : self.exp_dim, None].repeat(1, 1, N), "b c n -> (b n) c")
        density = self.density_linear(torch.cat([feature_embedding, exp], 1))
        if self.training and self.noise > 0.0:
            density = density + torch.randn_like(density) * self.noise
        density_first_then_feature_embedding = torch.cat([density, feature_embedding], 1)
        return density_first_then_feature_embedding

    def color(self, feature_embedding, query_viewdirs, exp, pose=None):
        """
        Compute rgb color values given feature embedding and query viewdirs.

        Args:
            feature_embedding: A tensor of shape `(num_cameras, feature_dim, num_rays_per_camera * num_samples)`
                containing the feature embedding at each sample point.
            query_viewdirs: A tensor of shape `(num_cameras, 3, num_rays_per_camera * num_samples)`
                containing the viewing directions for each ray, repeated over samples.
            exp: A tensor of shape `(num_cameras, exp_dim)` containing the expression parameters.
            pose: A tensor of shape `(num_cameras, 6)` containing the camera pose offset parameters.

        Returns:
            A tensor of shape `(num_cameras, 3, num_rays_per_camera * num_samples)` containing
                the rgb color values at the feature/viewdir/expression combination.
        """
        # feature_embedding should have shape [nCams, featureDim, nRays * nSamples]
        # query_viewdirs should have shape [nCams, 3, nRays * nSamples]
        # exp should have shape [nCams, expDim]
        B, C, N = query_viewdirs.shape
        if pose is not None:
            R, _ = self._rot_trans_from_pose(pose)
            query_viewdirs = torch.bmm(R.permute(0, 2, 1), query_viewdirs)
        query_viewdirs_embedding = self.view_embedding(rearrange(query_viewdirs, "b c n -> (b n) c"))
        # TODO(LS): Maybe shouldn't feed exp here to prevent overfitting.... but probably good at least here
        exp = rearrange((exp / 3)[:, : self.exp_dim, None].repeat(1, 1, N), "b c n -> (b n) c")
        feature_embedding = rearrange(feature_embedding, "b c n -> (b n) c")
        to_cat = [feature_embedding, query_viewdirs_embedding, exp]
        color = self.color_linear(torch.cat(to_cat, 1))
        color = rearrange(color, "(b n) c -> b c n", b=B)
        color = torch.sigmoid(color)
        return color

    def interp(self, pts, volume, bbox):
        feature_volume = volume
        u = (pts[:, 0:1] - 0.5 * (bbox[0][0] + bbox[0][1])) / (0.5 * (bbox[0][1] - bbox[0][0]))
        v = (pts[:, 1:2] - 0.5 * (bbox[1][0] + bbox[1][1])) / (0.5 * (bbox[1][1] - bbox[1][0]))
        w = (pts[:, 2:3] - 0.5 * (bbox[2][0] + bbox[2][1])) / (0.5 * (bbox[2][1] - bbox[2][0]))
        uvw = rearrange(torch.cat([u, v, w], dim=1), "b c (n t q) -> b n t q c", t=1, q=1)
        feature = torch.nn.functional.grid_sample(feature_volume, uvw)
        feature = rearrange(feature, "b c n t q -> b c (n t q)")
        return feature

    def mult_dist_interp(self, pts, volume, bbox):
        u = (pts[:, 0:1] - 0.5 * (bbox[0][0] + bbox[0][1])) / (0.5 * (bbox[0][1] - bbox[0][0]))
        v = (pts[:, 1:2] - 0.5 * (bbox[1][0] + bbox[1][1])) / (0.5 * (bbox[1][1] - bbox[1][0]))
        w = (pts[:, 2:3] - 0.5 * (bbox[2][0] + bbox[2][1])) / (0.5 * (bbox[2][1] - bbox[2][0]))
        # print(pts[0, 2])
        uvw = rearrange(torch.cat([u, v, w], dim=1), "b c (n t q) -> b n t q c", t=1, q=1)

        feature_list = []
        for i in range(self.interp_level):
            feature_volume = volume[:, :, :: 2**i, :: 2**i, :: 2**i]
            feature = torch.nn.functional.grid_sample(feature_volume, uvw)
            feature = rearrange(feature, "b c n t q -> b c (n t q)")
            feature_list.append(feature)
        feature = torch.cat(feature_list, dim=1)
        return feature


class AvatarMAVField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        geo_feat_dim: output geo feat dimensions
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        geo_feat_dim: int = 108,
        spatial_distortion: Optional[SpatialDistortion] = None,
        num_cameras_per_batch: int = 4,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim
        self._num_cameras_per_batch = num_cameras_per_batch

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.step = 0

        self.headmodule = HeadModule()

        # self.flame_exp_codes_per_cam = torch.zeros((self.num_images, self.headmodule.exp_dim), device="cuda")
        # self.flame_pose_codes_per_cam = torch.zeros((self.num_images, 6), device="cuda")

    @property
    def num_cameras_per_batch(self):
        if self.training:
            return self._num_cameras_per_batch
        else:
            return 1

    def get_exp_pose(self, ray_samples: RaySamples, n_rays_per_camera: int) -> Tuple[Tensor, Tensor]:
        if "flame_exps" in ray_samples.metadata:
            exp = ray_samples.metadata["flame_exps"][::n_rays_per_camera, 0]
        else:
            exp = torch.zeros(
                (self.num_cameras_per_batch, self.headmodule.exp_dim), device=ray_samples.frustums.directions.device
            )
        if not self.training and os.path.exists("/tmp/exp.npy"):
            exp = (
                torch.from_numpy(np.load("/tmp/exp.npy"))
                .view(1, -1)
                .repeat(self.num_cameras_per_batch, 1)
                .to(ray_samples.frustums.directions.device)
            )
        if "flame_poses" in ray_samples.metadata:
            pose = ray_samples.metadata["flame_poses"]  # has shape [n_rays, n_samples, 6]
            pose = pose[::n_rays_per_camera, 0]
        else:
            pose = torch.zeros((self.num_cameras_per_batch, 6), device=ray_samples.frustums.directions.device)
        if not self.training and os.path.exists("/tmp/pose.npy"):
            pose = (
                torch.from_numpy(np.load("/tmp/pose.npy"))
                .view(1, -1)[:, : self.headmodule.exp_dim]
                .repeat(self.num_cameras_per_batch, 1)
                .to(ray_samples.frustums.directions.device)
            )
        return exp, pose

    # def get_exp_pose(self, ray_samples: RaySamples, n_rays_per_camera: int) -> Tuple[Tensor, Tensor]:
    #     cam_indices_one_per_cam = ray_samples.camera_indices[::n_rays_per_camera, 0, 0]

    #     self.flame_exp_codes_per_cam = self.flame_exp_codes_per_cam.to(ray_samples.frustums.directions.device)
    #     self.flame_pose_codes_per_cam = self.flame_pose_codes_per_cam.to(ray_samples.frustums.directions.device)

    #     exp = self.flame_exp_codes_per_cam[cam_indices_one_per_cam]
    #     pose = self.flame_pose_codes_per_cam[cam_indices_one_per_cam]

    #     if not self.training and os.path.exists("/tmp/pose.npy"):
    #         pose = (
    #             torch.from_numpy(np.load("/tmp/pose.npy"))
    #             .view(1, 6)
    #             .repeat(cam_indices_one_per_cam.shape[0], 1)
    #             .to(ray_samples.frustums.directions.device)
    #         )
    #     return exp, pose

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        n_rays = positions.shape[0]
        n_rays_per_camera = n_rays // self.num_cameras_per_batch
        for i in range(self.num_cameras_per_batch):
            # TODO(LS): can delete this sanity check later
            subset = ray_samples.camera_indices[n_rays_per_camera * i : n_rays_per_camera * (i + 1), :, 0].reshape(-1)
            assert (subset[0] == subset).all()

        positions = rearrange(positions, "(n rays) samples dim -> n dim (rays samples)", n=self.num_cameras_per_batch)

        exp, pose = self.get_exp_pose(ray_samples, n_rays_per_camera)

        # Positions going in should have shape [nCams, 3, (nRays * nSamples)]
        # Exp going in should have shape [nCams, expDim]
        h = self.headmodule.density(positions, exp, pose)
        h = rearrange(h, "(rays samples) dim -> rays samples dim", rays=n_rays)

        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        # positions.shape = (num_rays, num_samples, 3)

        n_rays = ray_samples.frustums.directions.shape[0]
        n_rays_per_camera = n_rays // self.num_cameras_per_batch

        # density_embedding should have shape [nCams, featureDim, nRays * nSamples]
        density_embedding = rearrange(
            density_embedding, "(n rays) samples dim -> n dim (rays samples)", n=self.num_cameras_per_batch
        )
        # query_viewdirs should have shape [nCams, 3, nRays * nSamples]
        viewdirs = rearrange(directions, "(n rays) samples dim -> n dim (rays samples)", n=self.num_cameras_per_batch)
        # exp should have shape [nCams, expDim]

        exp, pose = self.get_exp_pose(ray_samples, n_rays_per_camera)
        rgb = self.headmodule.color(density_embedding, viewdirs, exp, pose)
        rgb = rearrange(rgb, "n c (rays samples) -> (n rays) samples c", samples=directions.shape[1])
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
