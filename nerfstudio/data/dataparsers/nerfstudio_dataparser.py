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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.fields.avatar_mav_field import _so3_exp_map
from nerfstudio.flame.flame import FlameHead
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval. 
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    apply_flame_poses_to_cams: bool = False
    """If True, FLAME poses will be added to camera poses, and zero pose will be passed to ray deformation."""
    apply_neck_rot_to_flame_pose: bool = False
    """If True, neck rotation will be applied to the flame pose, instead of to expression code."""
    neck_pose_to_expr: bool = False
    """If True, neck rotation will be passed to the conditioning expression no matter what!"""
    eyes_pose_to_expr: bool = False
    """If True, eyes rotation will be passed to the conditioning expression code."""


@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: NerfstudioDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        flame_model = FlameHead(300, 100)

        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.name.endswith("_train.json"):
            transforms_json_path = Path(str(self.config.data).replace("_train.json", f"_{split}.json"))
            print(split, "transforms_json_path", transforms_json_path)
            meta = load_from_json(transforms_json_path)
            data_dir = self.config.data.parent
        elif self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        fg_mask_fnames = []
        flame_param_fnames = []

        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

            if "fg_mask_path" in frame:
                fg_mask_filepath = Path(frame["fg_mask_path"])
                fg_mask_fname = self._get_fname(fg_mask_filepath, data_dir)
                fg_mask_fnames.append(fg_mask_fname)

            if "flame_param_path" in frame:
                flame_param_filepath = Path(frame["flame_param_path"])
                flame_param_fname = self._get_fname(flame_param_filepath, data_dir)
                flame_param_fnames.append(flame_param_fname)

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(fg_mask_fnames) == 0 or (
            len(fg_mask_fnames) == len(image_filenames)
        ), """
        Different number of image and fg mask filenames.
        You should check that fg_mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(flame_param_fnames) == 0 or (
            len(flame_param_fnames) == len(image_filenames)
        ), """
        Different number of image and flame param filenames.
        You should check that flame_param_path is specified for every frame (or zero frames) in transforms.json.
        """

        indices = np.arange(len(image_filenames))

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        fg_mask_fnames = [fg_mask_fnames[i] for i in indices] if len(fg_mask_fnames) > 0 else []
        flame_param_fnames = [flame_param_fnames[i] for i in indices] if len(flame_param_fnames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        flame_poses, flame_exps = [], []
        assert len(poses) == len(flame_param_fnames)

        for i, flame_param_fname in enumerate(flame_param_fnames):
            fp = np.load(flame_param_fname)
            R = _so3_exp_map(torch.from_numpy(fp["rotation"]).float())[0]  # [3, 3]
            T = torch.from_numpy(fp["translation"]).float()[0]  # [3,]

            if self.config.apply_neck_rot_to_flame_pose:
                neck_rot = _so3_exp_map(torch.from_numpy(fp["neck_pose"]).float())[0]
                R = R @ neck_rot

                verts, posed_joints, landmarks = flame_model(
                    torch.tensor(fp["shape"][None, ...]).float(),
                    torch.tensor(fp["expr"]).float(),
                    torch.tensor(fp["rotation"]).float(),
                    torch.tensor(fp["neck_pose"]).float(),
                    torch.tensor(fp["jaw_pose"]).float(),
                    torch.tensor(fp["eyes_pose"]).float(),
                    torch.tensor(fp["translation"]).float(),
                    return_landmarks=True,
                )

                neck_translation = posed_joints[:, 1]
                T = neck_translation[...][0].float()
                assert list(T.shape) == [3], T.shape

            if self.config.apply_flame_poses_to_cams:
                # Compute inverse flame pose
                Rflameinv = torch.eye(4, dtype=torch.float32)
                Rflameinv[:3, :3] = R.T
                Rflameinv[:3, 3] = -Rflameinv[:3, :3] @ T
                # Shift the camera by the inverse flame pose
                Rcam = torch.eye(4, dtype=torch.float32)
                Rcam[:3] = poses[i]
                poses[i] = (Rflameinv @ Rcam)[:3]
                # Set the flame pose to identity since it has been applied to the camera
                R = torch.eye(3, dtype=torch.float32)
                T = torch.zeros(3, 1, dtype=torch.float32)

            flame_pose = torch.cat([R.view(-1), T.view(-1)])
            flame_poses.append(flame_pose)

            maybe_neck_pose = (
                []
                if (not self.config.neck_pose_to_expr and self.config.apply_neck_rot_to_flame_pose)
                else [fp["neck_pose"]]
            )
            maybe_eyes_pose = [fp["eyes_pose"]] if self.config.eyes_pose_to_expr else []
            expr_to_cat = maybe_neck_pose + maybe_eyes_pose + [fp["jaw_pose"], fp["expr"]]
            flame_exps.append(torch.from_numpy(np.concatenate(expr_to_cat, 1)[0]).float())

        flame_poses = torch.stack(flame_poses).float()
        flame_exps = torch.stack(flame_exps).float()
        print(split, flame_poses.shape)
        print(split, flame_exps.shape)

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata={
                "flame_poses": flame_poses,
                "flame_exps": flame_exps,
            },
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath
