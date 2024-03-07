import os
import glob
import time
import argparse
import yaml
import bpy

from visualizers.visualizer_blender import GroundUpVisualizerBlender as GroundUpVisualizer

debug = False # True if you want to run in Blender GUI

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(run_cfg, add_mesh_color=True):

    # Get paths
    dataset_root = run_cfg['dataset_root']
    samples_diffusion = run_cfg['samples_diffusion']
    samples_hf = run_cfg['samples_hf']
    samples_sr = run_cfg['samples_sr']
    save_path = run_cfg['save_path']
    scene_name = run_cfg['dataset_name']

    # of samples to visualize
    num_samples = run_cfg['num_samples']

    # image size for rendering
    image_size = (run_cfg['image_size'], run_cfg['image_size'])

    for i in range(num_samples):
        print(i)
        # Run visualization
        gup_visualizer = GroundUpVisualizer(sample_path=samples_diffusion[i],
                                            dataset_root=dataset_root,
                                            save_path=save_path,
                                            scene_name=scene_name,
                                            add_color_to_mesh=add_mesh_color,
                                            device='cuda',
                                            samples_baseline=samples_hf[i],
                                            samples_sr=samples_sr[i],
                                            cfg_dict=run_cfg,
                                            image_size=(256, 256),
                                            light_offset=(0, 0, 5),
                                            )

        # Generate the mesh and render all modes
        # gup_visualizer.mesh_and_render_all_modes(image_size=image_size,
        #                                          render_scene=True,
        #                                          export_mesh=True,
        #                                          fix_colors=run_cfg["fix_mesh_colors"]
        #                                          )

        # gup_visualizer.prepare_mesh_for_bpy(mode='gt')

        print('done')



def define_options():
    parser = argparse.ArgumentParser(description="Mesh Results and Render Script")

    parser.add_argument("--data_root", type=str, default='/mnt/data_s/gunlu/data_gen/UrbanScene3D/',
                        help="Path to the dataset")
    parser.add_argument("--dataset_name", type=str, default='sf_test_1k_2',
                        help="Path to the test dataset name")
    parser.add_argument("--path_root_diffusion", type=str,
                        default='/mnt/data_f/gunlu/Experiments/GroundUp/Diffusion/diff_results_v14/',
                        help="Root path to pre-trained diffusion models")
    parser.add_argument("--model_name_diffusion", type=str,
                        default='test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3',
                        help="Model name for diffusion")
    parser.add_argument("--path_root_hf", type=str,
                        default='/mnt/data_s/gunlu/experiments/GroundUp/result_paper/',
                        help="Path to the HF baseline models")
    parser.add_argument("--model_name_hf", type=str, default='ms_perturb_grad_normal_2',
                        help="Model name for HF")
    parser.add_argument("--path_root_sr", type=str,
                        default='/mnt/data_s/gunlu/Experiments/GroundUp/SimpleRecon/RESULTS/v9',
                        help="Path to the SimpleRecon models")
    parser.add_argument("--model_name_sr", type=str, default='epi_v9_occ_p_empty_linscale_big_50_0',
                        help="Model name for SR")
    parser.add_argument("--save_path", type=str,
                        default='/mnt/data_f/gunlu/Experiments/GroundUp/results_papers/qualitative',
                        help="Save path for renders and meshes")
    parser.add_argument("--num_samples", type=int,
                        default=1,
                        help="Number of samples to visualize")
    parser.add_argument("--image_size", type=int,
                        default=1024,
                        help="Image size for rendering")
    parser.add_argument("--fix_mesh_colors", type=str2bool,
                        default=False,
                        help="Post-process vertex colors for improved color between ground-building boundary - SLOW!")
    parser.add_argument('--path_repo_root', type=str,
                        default='/home/gunlu/Projects/groundup_visualizer_local',
                        help='Path to repo root')
    parser.add_argument('--path_blender_configs', type=str,
                        default='configs/config_UrbanScene3D_visualization.yaml',
                        help='config file location under repo root')
    args = parser.parse_args()
    return args

def get_configs(run_in_gui):

    cfg = define_options()

    # PATHS

    # GT dataset
    dataset_root_path = os.path.join(cfg.data_root, cfg.dataset_name)

    # Get samples for diffusion
    # path_diffusion_samples = os.path.join(cfg.path_root_diffusion, cfg.model_name_diffusion, 'test/epoch0001')
    path_diffusion_samples = os.path.join(cfg.path_root_diffusion, cfg.model_name_diffusion, 'test/epoch0035')
    samples_diffusion = sorted(glob.glob(os.path.join(path_diffusion_samples, '*_gt.npy')))

    # Get samples for HF
    path_hf_samples = os.path.join(cfg.path_root_hf, cfg.model_name_hf, cfg.dataset_name, 'diffusion/pred_depth')
    samples_hf = sorted(glob.glob(os.path.join(path_hf_samples, '*_pred.npy')))

    # Get samples for SR
    path_sr_samples = os.path.join(cfg.path_root_sr, cfg.model_name_sr, '')  # 'diffusion/pred_depth')
    samples_sr = sorted(glob.glob(os.path.join(path_sr_samples, '*.pickle')))


    # Save path
    # save_path = os.path.join(cfg.save_path, 'testing_renders_blender')
    save_path = os.path.join(cfg.save_path, 'HERE')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Blender configs
    config_path = os.path.join(cfg.path_repo_root, cfg.path_blender_configs)
    if run_in_gui:
        mount_path = r'//wsl.localhost/Ubuntu/'
        config_path = bpy.path.abspath(os.path.normpath(config_path))
        config_path = os.path.join(mount_path, config_path)
    configs_blender = yaml.safe_load(open(config_path))


    return {"dataset_root": dataset_root_path,
            "samples_diffusion": samples_diffusion,
            "samples_hf": samples_hf,
            "samples_sr": samples_sr,
            "save_path": save_path,
            "dataset_name": cfg.dataset_name,
            "image_size": cfg.image_size,
            "num_samples": cfg.num_samples,
            "fix_mesh_colors": cfg.fix_mesh_colors,
            "cfg_blender": configs_blender,
            }




if __name__ == '__main__':

    # If Blender GUI mode, True
    run_in_gui = debug

    run_config = get_configs(run_in_gui)

    start_time = time.time()
    # Run visualization
    main(run_cfg=run_config)
    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
