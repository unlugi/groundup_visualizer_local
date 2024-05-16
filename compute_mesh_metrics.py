import glob
import json
import os
import time
from pathlib import Path

import click
import numpy as np
import open3d as o3d
import torch
from pytorch3d.io import IO
from tqdm import tqdm

from utils.mesh_metrics_utils import (SimpleVolume, VisibilityAggregator,
                                      compute_point_cloud_metrics, BackprojectDepth)
from visualizers.visualizer_p3d import \
    GroundUpVisualizerP3D as GroundUpVisualizer

"""
Run with something like:
CUDA_VISIBLE_DEVICES=3 python compute_mesh_metrics.py \
--save_path debug_dump/ \
--predictions_path /mnt/data_f/gunlu/Experiments/GroundUp/Diffusion/diff_results_v14/test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3/test/epoch0035/ \
--save_path debug_dump/test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3

If you need to output debug renders and meshes, use the --debug_dump flag. Use --verbose to print scenes metrics and logs to console.
"""

def get_visualizer_and_mesh(sample_path, dataset_root, mode="gt", scene_name=None, verbose=False):
    add_mesh_color = True

    gup_visualizer = GroundUpVisualizer(
        sample_path=sample_path,
        dataset_root=dataset_root,
        scene_name=scene_name,
        add_color_to_mesh=add_mesh_color,
        device='cuda',
        verbose=verbose,
    )
    # Run meshing
    gup_visualizer.get_mesh_in_world_coordinates(model_name=mode)
    
    return gup_visualizer, gup_visualizer.get_trimesh(update_face_colors=False)


def dump_debug_viz(gup_visualizer, save_path, mode="gt", offset=(-3.0, 0.0, 4.0)):
    # Save mesh
    filename = os.path.join(save_path, "mesh_{}_elevation_{}.ply".format(mode, gup_visualizer.sample_idx))
    IO().save_mesh(gup_visualizer.mesh, filename)
    
    # Render image
    cam_p_p3d, rendered_image = gup_visualizer.render_scene_pyrender(image_size=(1024, 1024), offset=offset)
    # cam_p_p3d, rendered_image = gup_visualizer.render_scene(image_size=(1024, 1024), offset=offset)
    filename = os.path.join(save_path, "image_{}_{}.png".format(mode, gup_visualizer.sample_idx))
    rendered_image.save(filename, 'PNG')
    return cam_p_p3d, rendered_image


def compute_metrics_for_sample(
    sample_path, 
    dataset_root, 
    sample_save_path,
    debug_dump=False, 
    scene_name=None, 
    verbose=False, 
    clip_with_visibility=False
):

    try:
    # get separate visualizers for gt and pred, since the visualizer code uses the same internal variable for meshes.
        gt_visualizer, gt_trimesh_mesh = get_visualizer_and_mesh(sample_path, dataset_root, mode="gt", scene_name=scene_name, verbose=verbose)
        pred_visualizer, pred_trimesh_mesh = get_visualizer_and_mesh(sample_path, dataset_root, mode="pred", scene_name=scene_name, verbose=verbose)
        
        # convert to open3d and sample to get a point cloud.
        gt_o3d_mesh = gt_trimesh_mesh.as_open3d
        gt_o3d_pcd = gt_o3d_mesh.sample_points_uniformly(number_of_points=100000)
        
        pred_o3d_mesh = pred_trimesh_mesh.as_open3d
        pred_o3d_pcd = pred_o3d_mesh.sample_points_uniformly(number_of_points=100000)
        
        if debug_dump:
            # cleanliness check
            _test_metrics, _, _ = compute_point_cloud_metrics(gt_o3d_pcd, gt_o3d_pcd)
            assert _test_metrics['chamfer↓'] == 0.0
            assert _test_metrics['f1_score↑'] == 1.0

        visible_pred_indices = None
        visible_gt_indices = None
        if clip_with_visibility:
            # get gt mesh bounds
            gt_bounds_min = np.array(gt_o3d_mesh.vertices).min(axis=0)
            gt_bounds_max = np.array(gt_o3d_mesh.vertices).max(axis=0)
            bounds = {}
            bounds['xmin'] = gt_bounds_min[0]
            bounds['ymin'] = gt_bounds_min[1]
            bounds['zmin'] = gt_bounds_min[2]
            bounds['xmax'] = gt_bounds_max[0]
            bounds['ymax'] = gt_bounds_max[1]
            bounds['zmax'] = gt_bounds_max[2]
            
            
            # create a volume
            visibility_volume = SimpleVolume.from_bounds(bounds=bounds, voxel_size=0.05)
            visibility_volume.cuda()
            
            # create a visibility aggregator
            visibility_aggregator = VisibilityAggregator(volume=visibility_volume, additional_extent=0.5)
            
            # get projection matrices
            K_b44 = torch.from_numpy(gt_visualizer.fix_camera_intrinsics(gt_visualizer.cameras['K_p'].copy(), [1,1])).cuda().clone().unsqueeze(0)
            cam_T_world_b44 = torch.from_numpy(gt_visualizer.cameras['cam_perspective']['camera_pc']).clone().cuda().unsqueeze(0)
            world_T_cam_b44 = torch.inverse(cam_T_world_b44) 
            
            
            depth_b1hw = torch.tensor(gt_visualizer.data['gt_perspective'])[None, None].cuda()
            depth_b1hw = depth_b1hw.transpose(2,3)
            depth_b1hw[torch.isnan(depth_b1hw)] = 0
            visibility_aggregator.integrate_into_volume(depth_b1hw, cam_T_world_b44, K_b44)
                   
            if debug_dump:
                # dump the volume
                pcd_visibility_volume_save_path = Path(sample_save_path) / "pcd_visibility_volume.ply"
                o3d.io.write_point_cloud(str(pcd_visibility_volume_save_path), visibility_volume.to_point_cloud(threshold=0.5, num_points=100000))
                
                resolution = 1024
                invK_b44 = torch.linalg.inv(torch.from_numpy(gt_visualizer.fix_camera_intrinsics(gt_visualizer.cameras['K_p'].copy(), [resolution,resolution])).cuda().clone().unsqueeze(0))

                upsamplined_depth_map = torch.nn.functional.interpolate(depth_b1hw, size=(resolution,resolution), mode='nearest')
                backprojector = BackprojectDepth(width=resolution, height=resolution).cuda()
                points = backprojector(upsamplined_depth_map, invK_b44)
                points = world_T_cam_b44 @ points
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.squeeze().permute(1,0)[:, :3].cpu().numpy())
                # print("Radius oulier removal")
                # cl, ind = pcd.remove_radius_outlier(nb_points=2, radius=0.05)
                # pcd = pcd.select_by_index(ind)
                o3d.io.write_point_cloud(str(Path(sample_save_path) / "backproj_pers_mine.ply"), pcd)
                gt_visualizer.get_pointcloud_depth_perspective_in_world_coordinates(path_to_save=Path(sample_save_path) / "backproj_pers.ply", is_save=True)
                
            # get the raw points from the pred pcd
            pcd_points_N3 = torch.tensor(np.array(pred_o3d_pcd.points)).float().cuda()
            # sample the volume at those pred points to figure out if they're visible
            vis_samples_N = visibility_volume.sample_volume(world_points_N3=pcd_points_N3)
            valid_mask_N = vis_samples_N > 0.5
            # get visible indices
            visible_pred_indices = valid_mask_N.nonzero().squeeze().cpu().numpy().tolist()
            
            # get the raw points from the pred pcd
            pcd_points_N3 = torch.tensor(np.array(gt_o3d_pcd.points)).float().cuda()
            # sample the volume at those pred points to figure out if they're visible
            vis_samples_N = visibility_volume.sample_volume(world_points_N3=pcd_points_N3)
            valid_mask_N = vis_samples_N > 0.5
            # get visible indices
            visible_gt_indices = valid_mask_N.nonzero().squeeze().cpu().numpy().tolist()

        # compute metrics
        metrics, distances_pred2gt, distances_gt2pred = compute_point_cloud_metrics(gt_o3d_pcd, pred_o3d_pcd, visible_pred_indices=visible_pred_indices, visible_gt_indices=visible_gt_indices)

        if verbose:
            print(sample_path, "\n", metrics)
        
        if debug_dump:
            dump_debug_viz(gt_visualizer, sample_save_path, mode="gt")
            dump_debug_viz(pred_visualizer, sample_save_path, mode="pred")
        
        # save metrics in a json file
        metrics_save_path = Path(sample_save_path) / "metrics.json"
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    except Exception as e:
        print("Error in metrics:", e)
        print(sample_path)
        metrics = {}
        metrics["acc↓"] = 1.0
        metrics["compl↓"] = 1.0
        metrics["chamfer↓"] = 1.0
        metrics["precision↑"] = 0.0
        metrics["recall↑"] = 0.0
        metrics["f1_score↑"] = 0.0
        
    return metrics

def main(sample_paths, dataset_root, save_path, debug_dump=False, scene_name=None, verbose=False, clip_with_visibility=False):
    
    all_metrics = []
    # sample_paths = sample_paths[316:]
    for sample_path in tqdm(sample_paths):
        # create a folder for each sample
        sample_save_path = Path(save_path) / sample_path.split('/')[-1].split('.')[0]
        sample_save_path.mkdir(exist_ok=True, parents=True)
        
        sample_metrics = compute_metrics_for_sample(
            sample_path=sample_path, 
            dataset_root=dataset_root, 
            sample_save_path=sample_save_path, 
            debug_dump=debug_dump, 
            scene_name=scene_name, 
            verbose=verbose, 
            clip_with_visibility=clip_with_visibility,
        )
        
        all_metrics.append(sample_metrics)
    
    # aggregate metrics
    metrics = {}
    for metric_name in all_metrics[0].keys():
        metrics[metric_name] = torch.tensor([sample_metrics[metric_name] for sample_metrics in all_metrics]).mean().item()
    
    # add some extra info
    metrics['num_samples'] = len(all_metrics)
    metrics["sample_path"] = str(Path(sample_paths[0]).parent.resolve())
    
    # save metrics in a json file
    metrics_save_path = Path(save_path) / "metrics.json"
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        


@click.command()
@click.option(
    "--dataset_path",
    type=str,
    default="/mnt/data_s/gunlu/data_gen/UrbanScene3D",
    help="Root directory of datasets",
)
@click.option(
    "--dataset_name",
    type=str,
    default="sf_test_1k_2",
    help="dataset name",
)
@click.option(
    "--predictions_path",
    type=str,
    default="/mnt/data_s/gunlu/data_gen/UrbanScene3D/sf_test_1k_2",
    help="Root directory of predictions",
)
@click.option(
    "--model_name",
    type=str,
    default="test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3",
    help="Model name",
)
@click.option(
    "--save_path",
    type=str,
    help="Save path.",
)
@click.option(
    "--debug_dump",
    is_flag=True,
)
@click.option(
    "--verbose",
    is_flag=True,
)
@click.option(
    "--clip_with_visibility",
    is_flag=True,
)
def cli(
    dataset_path: str, 
    dataset_name: str, 
    predictions_path: str, 
    model_name: str, 
    save_path: str, 
    debug_dump: bool=False, 
    verbose: bool=False, 
    clip_with_visibility: bool=False,
):

    # get paths of samples to evaluate. They should have "_gt.npy" in the name,
    # and there should be a corresponding prediction file with "_pred.npy" in the name.
    sample_paths = sorted(glob.glob(os.path.join(predictions_path, '*.npy')))
    sample_paths = [sample_path for sample_path in sample_paths if "_gt.npy" in Path(sample_path).name]
    
    main(
        sample_paths,
        dataset_root=os.path.join(dataset_path, dataset_name),
        save_path=save_path,
        scene_name=dataset_name,
        debug_dump=debug_dump,
        verbose=verbose,
        clip_with_visibility=clip_with_visibility,
    )

if __name__ == '__main__':
    cli()