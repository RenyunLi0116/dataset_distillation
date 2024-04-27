"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import pathlib
import time
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField
from torch.utils.data import Subset

from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator


def sample_indices(file_path, num_samples):
    values = np.loadtxt(file_path)
    normalized_values = values / np.sum(values) # normalize weights to sum to 1

    indices = np.random.choice(len(values), size=num_samples, p=normalized_values)
    indices = indices.tolist()

    return indices



def run(args):
    device = "cuda:0"
    set_random_seed(42)
    

    if args.scene in MIPNERF360_UNBOUNDED_SCENES:
        from datasets.nerf_360_v2 import SubjectLoader
        # v4 update: rotate the view

        # training parameters
        max_steps = 20000
        init_batch_size = 1024
        target_sample_batch_size = 1 << 18
        weight_decay = 0.0
        # scene parameters
        aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
        near_plane = 0.2
        far_plane = 1.0e10
        # dataset parameters
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        # model parameters
        grid_resolution = 128
        grid_nlvl = 4
        # render parameters
        render_step_size = 1e-3
        alpha_thre = 1e-2
        cone_angle = 0.004

    else:
        from datasets.nerf_synthetic_v6 import SubjectLoader

        # training parameters
        max_steps = 20000
        init_batch_size = 1024
        target_sample_batch_size = 1 << 18
        weight_decay = (
            1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
        )
        # scene parameters
        aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
        near_plane = 0.0
        far_plane = 1.0e10
        # dataset parameters
        train_dataset_kwargs = {}
        test_dataset_kwargs = {}
        # model parameters
        grid_resolution = 128
        grid_nlvl = 1
        # render parameters
        render_step_size = 5e-3
        alpha_thre = 0.0
        cone_angle = 0.0

    # angle_x, angle_y, angle_z = args.angle

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=init_batch_size,
        device=device,
        mode='train',
        # cam=cam,
        **train_dataset_kwargs,
    )

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="test",
        num_rays=None,
        device=device,
        mode='test',
        **test_dataset_kwargs,
    )

    # TODO
    val_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="val",
        num_rays=None,
        device=device,
        # mode='val',
        **test_dataset_kwargs,
    )

    if args.vdb:
        from fvdb import sparse_grid_from_dense

        from nerfacc.estimators.vdb import VDBEstimator

        assert grid_nlvl == 1, "VDBEstimator only supports grid_nlvl=1"
        voxel_sizes = (aabb[3:] - aabb[:3]) / grid_resolution
        origins = aabb[:3] + voxel_sizes / 2
        grid = sparse_grid_from_dense(
            1,
            (grid_resolution, grid_resolution, grid_resolution),
            voxel_sizes=voxel_sizes,
            origins=origins,
        )
        estimator = VDBEstimator(grid).to(device)
        estimator.aabbs = [aabb]
    else:
        estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
        ).to(device)

    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)
    optimizer = torch.optim.Adam(
        radiance_field.parameters(),
        lr=1e-2,
        eps=1e-15,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=100
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
            ),
        ]
    )
    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    # training
    tic = time.time()
    # v2 update: store psnr, lpips, loss
    # psnr_store = []
    # lpips_store = []
    # loss_store = []
    # v3 update: find dramatic changes of loss and psnr
    # significant_changes = {'psnr': [], 'loss': [], 'data': []}
    # prev_rate = None
    # prev_psnr = None
    # prev_loss = None


    # # v6 update:
    # loss_store = []
    # psnr_store = []
    # index_store = []
    # c2w_store = []

    stage = args.stage
    path = args.output
    if not os.path.exists(path):
        os.makedirs(path)
    stage_loss = []
    stage_psnr = []
    stage_idx = []
    stage_c2w = []
    stage_l = []

    stage_loss_val = []
    stage_psnr_val = []
    stage_idx_val = []
    stage_c2w_val = []
    stage_l_val = []

    '''
    # sampling distribution
    weight_path = args.weight_path
    sample_num  = args.sample_num 
    weight = sample_indices(weight_path, sample_num)
    print(f'weight is: {weight}, type is: {type(weight)}')
    '''
    
    # update train_dataset with the index from weight
    # train_dataset = train_dataset[weight]
    # train_dataset = [train_dataset[i] for i in weight]
    # train_dataset = Subset(train_dataset, weight)

    print(f'train dataset len is {len(train_dataset)}, val dataset len is {len(val_dataset)}')

    for step in range(max_steps + 1):
        radiance_field.train()
        estimator.train()

        if step % stage == 0 and step != 0:
            print(f'stage {step//stage} starts')
            # i = torch.randint(0, len(train_dataset), (1,)).item()
            # data = train_dataset[i]
            # for i in range(len(data)):
            #     data_part = data[i]
            #     # ...

            # train_dataset_temp = [train_dataset[i] for i in weight]

            for i in range(len(train_dataset)):
                data = train_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                c2w = data["c2w"] # batch x 4 x 4, 7120 x 4 x 4
                l = c2w.shape[0]
                # print(f'c2w shape: {c2w.shape}')
                rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )
                loss = F.mse_loss(rgb, pixels)
                stage_loss.append(loss.item())
                psnr = -10.0 * torch.log(loss) / np.log(10.0)
                stage_psnr.append(psnr.item())
                stage_idx.append(i)
                stage_c2w.append(c2w.cpu().numpy())
                stage_l.append(l)
            # find top 10% loss and corresponding c2w
            threshold = np.percentile(stage_loss, 90)
            top_10_percent_index = [index for index, loss in zip(stage_idx, stage_loss) if loss > threshold]
            corresponding_c2w = [stage_c2w[index] for index in top_10_percent_index] # shape (num, 4, 4)
            top_10_percent_loss = [stage_loss[index] for index in top_10_percent_index]
            top_10_percent_psnr = [stage_psnr[index] for index in top_10_percent_index]
            top_10_percnet_len = [stage_l[index] for index in top_10_percent_index]
            print(f'index len: {len(top_10_percent_index)}, c2w len: {len(corresponding_c2w)}, loss len: {len(top_10_percent_loss)}, psnr len: {len(top_10_percent_psnr)}')
           
            with open(f'{path}/high_loss_c2w_stage_{step}.txt', 'w') as file:
                for matrix in corresponding_c2w:
                    for row in matrix:
                        # Ensure row is flattened to a 1D list of floats
                        flattened_row = row.flatten()  # Flatten the row if necessary
                        # Format each value in scientific notation
                        formatted_row = ', '.join(f"{value:.6e}" for value in flattened_row)
                        file.write(formatted_row + '\n')  # Write each row on a new line
            # with open(f'high_loss_c2w_stage_{step}.txt', 'w') as file:
            #     for matrix in corresponding_c2w:
            #         for row in matrix:
            #             file.write(' '.join(f"{value:.6f}" for value in row) + '\n')
            #         file.write('\n')  # Separate matrices by a newline
            # Save corresponding losses
            with open(f'{path}/high_loss_values_stage_{step}.txt', 'w') as file:
                for loss_value in top_10_percent_loss:
                    file.write(f"{loss_value:.8e}\n")
            # Save corresponding psnr
            with open(f'{path}/high_psnr_values_stage_{step}.txt', 'w') as file:
                for psnr_value in top_10_percent_psnr:
                    file.write(f"{psnr_value:.8e}\n")            
            # save l_store
            with open(f'{path}/high_len_values_stage_{step}.txt', 'w') as file:
                for l_value in top_10_percnet_len:
                    file.write(f"{l_value}\n")
            
            stage_loss = []
            stage_psnr = []
            stage_idx = []
            stage_c2w = []
            stage_l = []

        # validation per stage
        if step % stage == 0 and step != 0:
            print(f'stage {step//stage} validation starts')
            for i in range(len(val_dataset)):
                data = val_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                c2w = data["c2w"]
                l = c2w.shape[0]
                rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )
                loss = F.mse_loss(rgb, pixels)
                stage_loss_val.append(loss.item())
                psnr = -10.0 * torch.log(loss) / np.log(10.0)
                stage_psnr_val.append(psnr.item())
                stage_idx_val.append(i)
                stage_c2w_val.append(c2w.cpu().numpy())
                stage_l_val.append(l)
            # find top 10% loss and corresponding c2w
            threshold = np.percentile(stage_loss_val, 0)
            top_10_percent_index = [index for index, loss in zip(stage_idx_val, stage_loss_val) if loss > threshold]
            corresponding_c2w = [stage_c2w_val[index] for index in top_10_percent_index] # shape (num, 4, 4)
            top_10_percent_loss = [stage_loss_val[index] for index in top_10_percent_index]
            top_10_percent_psnr = [stage_psnr_val[index] for index in top_10_percent_index]
            top_10_percnet_len = [stage_l_val[index] for index in top_10_percent_index]
            print(f'index len: {len(top_10_percent_index)}, c2w len: {len(corresponding_c2w)}, loss len: {len(top_10_percent_loss)}, psnr len: {len(top_10_percent_psnr)}')

            with open(f'{path}/high_loss_c2w_val_stage_{step}.txt', 'w') as file:
                for matrix in corresponding_c2w:
                    for row in matrix:
                        # Ensure row is flattened to a 1D list of floats
                        flattened_row = row.flatten()
                        # Format each value in scientific notation
                        formatted_row = ', '.join(f"{value:.6e}" for value in flattened_row)
                        file.write(formatted_row + '\n')  # Write each row on a new line
            # Save corresponding losses
            with open(f'{path}/high_loss_values_val_stage_{step}.txt', 'w') as file:
                for loss_value in top_10_percent_loss:
                    file.write(f"{loss_value:.8e}\n")
            # Save corresponding psnr
            with open(f'{path}/high_psnr_values_val_stage_{step}.txt', 'w') as file:
                for psnr_value in top_10_percent_psnr:
                    file.write(f"{psnr_value:.8e}\n")
            # save l_store
            with open(f'{path}/high_len_values_val_stage_{step}.txt', 'w') as file:
                for l_value in top_10_percnet_len:
                    file.write(f"{l_value}\n")
            
            stage_loss_val = []
            stage_psnr_val = []
            stage_idx_val = []
            stage_c2w_val = []
            stage_l_val = []
        

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]
        # print(f'train_dataset len: {len(train_dataset)}, data len: {len(data)}')
        # print(data["c2w"].shape)

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        c2w = data["c2w"]

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)

        # compute loss
        loss = F.smooth_l1_loss(rgb, pixels)

        '''
        # v6 update:
        loss_store.append(loss.item()) 
        index_store.append(i)
        # c2w_store.append(c2w)
        c2w_store.append(c2w.cpu().numpy())
        '''

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()


        '''
        if step > 2 and step % 100 == 0:
            # print(f"loss len: {len(loss_store)}, psnr len: {len(psnr_store)}")
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | psnr={psnr:.2f} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                f"max_depth={depth.max():.3f} | "
            )

            # v6 update: top 10% loss corresponding index and c2w
            threshold = np.percentile(loss_store, 90)
            top_10_percent_index = [index for index, loss in zip(index_store, loss_store) if loss > threshold]
            corresponding_c2w = [c2w_store[index] for index in top_10_percent_index] # shape (num, 4, 4)
            top_10_percent_loss = [loss_store[index] for index in top_10_percent_index]
            top_10_percent_psnr = [psnr_store[index] for index in top_10_percent_index]
            print(f'index len: {len(top_10_percent_index)}, c2w len: {len(corresponding_c2w)}, loss len: {len(top_10_percent_loss)}, psnr len: {len(top_10_percent_psnr)}')
            for i in range(len(corresponding_c2w)):
                print(corresponding_c2w[i].shape)

            # with open(f'high_loss_c2w_step_{step}.txt', 'w') as file:
            #     for matrix in corresponding_c2w:
            #         # Write each matrix in a formatted way
            #         for row in matrix:
            #             file.write(' '.join(map(str, row)) + '\n')
            #         file.write('\n')  # Separate matrices by a newline

            # save corresponding_c2w into txt, each 4 row stands for a 4x4 matrix, delimiter is ','
            # print(corresponding_c2w)
            # corresponding_c2w_np = np.array(corresponding_c2w).reshape(-1, 4)
            # np.savetxt(f'high_loss_c2w_step_{step}.txt', corresponding_c2w_np, delimiter=',')
            with open(f'high_loss_c2w_step_{step}.txt', 'w') as file:
                for matrix in corresponding_c2w:
                    for row in matrix:
                        # Ensure row is flattened to a 1D list of floats
                        flattened_row = row.flatten()  # Flatten the row if necessary
                        # Format each value in scientific notation
                        formatted_row = ', '.join(f"{value:.8e}" for value in flattened_row)
                        file.write(formatted_row + '\n')  # Write each row on a new line
            # Save corresponding losses
            with open(f'high_loss_values_step_{step}.txt', 'w') as file:
                for loss_value in top_10_percent_loss:
                    file.write(f"{loss_value:.8e}\n")
            # Save corresponding psnr
            with open(f'high_psnr_values_step_{step}.txt', 'w') as file:
                for psnr_value in top_10_percent_psnr:
                    file.write(f"{psnr_value:.8e}\n")

            # ini loss_store, index_store, c2w_store
            loss_store = []
            psnr_store = []
            index_store = []
            c2w_store = []
        '''

        if step > 0 and step % max_steps == 0:
            # evaluation
            radiance_field.eval()
            estimator.eval()

            psnrs = []
            lpips = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(test_dataset))):
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]
                    c2w = data["c2w"]

                    # rendering
                    # rgb, acc, depth, _ = render_image_with_occgrid_test(
                    #     1024,
                    #     # scene
                    #     radiance_field,
                    #     estimator,
                    #     rays,
                    #     # rendering options
                    #     near_plane=near_plane,
                    #     render_step_size=render_step_size,
                    #     render_bkgd=render_bkgd,
                    #     cone_angle=cone_angle,
                    #     alpha_thre=alpha_thre,
                    # )
                    rgb, acc, depth, _ = render_image_with_occgrid(
                        radiance_field,
                        estimator,
                        rays,
                        # rendering options
                        near_plane=near_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                    )
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                    lpips.append(lpips_fn(rgb, pixels).item())

                    # psnr_store.append(psnr.item())
                    # lpips.append(lpips_fn(rgb, pixels).item())
                    # loss_store.append(mse.item())
                    # if i == 0:
                    #     imageio.imwrite(
                    #         "rgb_test.png",
                    #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                    #     )
                    #     imageio.imwrite(
                    #         "rgb_error.png",
                    #         (
                    #             (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                    #         ).astype(np.uint8),
                    #     )
            psnr_avg = sum(psnrs) / len(psnrs)
            lpips_avg = sum(lpips) / len(lpips)
            print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
            # psnr_store.append(psnr_avg)
            # lpips_store.append(lpips_avg)
            # loss_store.append(loss.item())
    
    # save psnr_store, lpips_store, loss_store into txt
    # np.savetxt(f"psnr_store_dramatic_v4_{args.name}.txt", np.array(psnr_store))
    # np.savetxt(f"loss_store_dramatic_v4_{args.name}.txt", np.array(loss_store))
    # np.save(f'loss_v4_{args.name}.npy', significant_changes['loss'])
    # np.save(f'data_v4_{args.name}.npy', significant_changes['data'])



    # # plot psnr_store, lpips_store, loss_store and save into png
    # import matplotlib.pyplot as plt
    # plt.plot(psnr_store)
    # plt.xlabel('step')
    # plt.ylabel('psnr')
    # plt.title('psnr vs step')
    # plt.savefig(f'psnr_store_v4_{args.name}.png')
    # plt.close()

    # # plt.plot(lpips_store)
    # # plt.xlabel('step')
    # # plt.ylabel('lpips')
    # # plt.title('lpips vs step')
    # # plt.savefig('lpips_store_v4_y90.png')
    # # plt.close()

    # plt.plot(loss_store)
    # plt.xlabel('step')
    # plt.ylabel('loss')
    # plt.title('loss vs step')
    # plt.savefig(f'loss_store_v4_{args.name}.png')
    # plt.close()

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        # default=str(pathlib.Path.cwd() / "data/360_v2"),
        default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
        help="the root dir of the dataset",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
        help="which scene to use",
    )
    parser.add_argument(
        "--vdb",
        action="store_true",
        help="use VDBEstimator instead of OccGridEstimator",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="the name of the experiment",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1000,
        help="the stage of the experiment",
    )
    parser.add_argument(
        "--output",
        type=None,
        default="./",
        help="output path"
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="./0424_val_all/high_loss_values_val_stage_20000.txt",
        # default='high_loss_values_val_stage_20000.txt'
        help="the path of the weight file"
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=30,
        help="the number of samples"
    )
    # parser.add_argument(
    #     "--angle", 
    #     nargs=3, 
    #     type=float, 
    #     default=[0, 0, 0], 
    #     help="Rotation angles around x, y, and z axes in degrees"
    # )
    args = parser.parse_args()

    run(args)
