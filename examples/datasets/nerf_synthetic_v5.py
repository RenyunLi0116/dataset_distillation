"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays


def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    # v4 update:
    # def rotate_camtoworlds(self, angle):
    #     print(f'nerf_synthetic.py: rotate_camtoworlds: angle={angle}')
    #     cos_angle = np.cos(angle)
    #     sin_angle = np.sin(angle)
    #     rotation_matrix = np.array([
    #         [cos_angle, 0, sin_angle, 0],
    #         [0, 1, 0, 0],
    #         [-sin_angle, 0, cos_angle, 0],
    #         [0, 0, 0, 1]
    #     ], dtype=np.float32)
    #     self.camtoworlds = np.einsum("nij, kj -> nki", self.camtoworlds, rotation_matrix)
    #     # self.camtoworlds = np.einsum("nij, kj -> nki", self.camtoworlds.cpu(), rotation_matrix)
    #     # self.camtoworlds = torch.from_numpy(self.camtoworlds).to(self.camtoworlds.device)
    def rotate_camtoworlds(self, angle):
        angle_rad = np.radians(angle)
        print(f'nerf_synthetic.py: rotate_camtoworlds: angle={angle_rad}')
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        rotation_matrix = np.array([
            [cos_angle, 0, sin_angle, 0],
            [0, 1, 0, 0],
            [-sin_angle, 0, cos_angle, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.camtoworlds = torch.tensor(np.einsum("nij, kj -> nki", self.camtoworlds.cpu().numpy(), rotation_matrix), device=self.camtoworlds.device)


    # def rotate_camtoworlds_v2(self, angle_x, angle_y, angle_z):
    #     # Convert angles from degrees to radians
    #     angle_x_rad = np.radians(angle_x)
    #     angle_y_rad = np.radians(angle_y)
    #     angle_z_rad = np.radians(angle_z)

    #     print(f'nerf_synthetic.py: rotate_camtoworlds: angle_x={angle_x_rad}, angle_y={angle_y_rad}, angle_z={angle_z_rad}')

    #     # Precompute cosines and sines for efficiency
    #     cx, cy, cz = np.cos([angle_x_rad, angle_y_rad, angle_z_rad])
    #     sx, sy, sz = np.sin([angle_x_rad, angle_y_rad, angle_z_rad])

    #     # Directly define the combined rotation matrix R
    #     rotation_matrix = np.array([
    #         [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz, 0],
    #         [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz, 0],
    #         [-sy,     cy * sx,                cx * cy,                0],
    #         [0,       0,                      0,                      1]
    #     ], dtype=np.float32)

    #     # Apply the rotation to the camtoworlds matrices
    #     # Assuming self.camtoworlds is a PyTorch tensor on a CUDA device
    #     self.camtoworlds = torch.tensor(np.einsum("nij, kj -> nki", self.camtoworlds.cpu().numpy(), rotation_matrix), device=self.camtoworlds.device)



    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]

    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
        mode: str = "train",
        # angle_x: float = 0, angle_y: float = 0, angle_z: float = 0
        # receive N*3 cam pose
        cam = None
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        # v4 update:
        # if mode == "train":
        #     angle = 30
        #     self.rotate_camtoworlds(angle)
        # if mode == "train":
        #     self.rotate_camtoworlds_v2(angle_x, angle_y, angle_z)
        if cam is not None:
            assert len(cam) == len(self.camtoworlds), "Number of provided camera poses must match the number of camtoworld matrices."
            for i, pose in enumerate(cam):
                # Ensure the pose is a PyTorch tensor and on the correct device
                pose_tensor = torch.tensor(pose, dtype=torch.float32, device=self.device)
                # Update the camtoworld matrix for each corresponding pose
                self.camtoworlds[i] = pose_tensor
        
        self.K = self.K.to(device)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(42)

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays, c2w = data["rgba"], data["rays"], data["c2w"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(
                    3, device=self.images.device, generator=self.g
                )
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            "c2w": c2w,  # [n_rays, 3, 4]
            # **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
            **{k: v for k, v in data.items() if k not in ["rgba", "rays", "c2w"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                    generator=self.g,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0,
                self.WIDTH,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
            y = torch.randint(
                0,
                self.HEIGHT,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "c2w": c2w,  # [num_rays, 3, 4]
        }