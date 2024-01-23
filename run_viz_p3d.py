import os
import glob
import time
import argparse

from visualizers.visualizer_p3d import GroundUpVisualizerP3D as GroundUpVisualizer

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
    save_path = run_cfg['save_path']
    scene_name = run_cfg['dataset_name']

    # of samples to visualize
    num_samples = run_cfg['num_samples']

    # image size for rendering
    image_size = (run_cfg['image_size'], run_cfg['image_size'])

    for i in range(num_samples):
        # Run visualization
        gup_visualizer = GroundUpVisualizer(sample_path=samples_diffusion[i],
                                            dataset_root=dataset_root,
                                            save_path=save_path,
                                            scene_name=scene_name,
                                            add_color_to_mesh=add_mesh_color,
                                            device='cuda',
                                            samples_baseline=samples_hf[i],
                                            )

        # Generate the mesh and render all modes
        gup_visualizer.mesh_and_render_all_modes(image_size=image_size,
                                                 render_scene=True,
                                                 export_mesh=True,
                                                 fix_colors=run_cfg["fix_mesh_colors"],
                                                 light_offset=(0.0, 0.0, 3.5)
                                                 )

        print('done')


    # save_mesh = False
    # # mode = 'gt'
    # mode = 'pred'
    #
    # # Generate the mesh
    # gup_visualizer.get_mesh_in_world_coordinates(model_name=mode)
    #
    # # Run meshing
    # offset = (-3.0, 0.0, 4.0) # for the position of the light in the scene - offset from the camera
    # # cam_p_p3d, rendered_image = gup_visualizer.render_scene(image_size=(256, 256), offset=offset)
    # cam_p_p3d = 0
    # rendered_image = 0
    #
    # gup_visualizer.export_mesh_p3d(mesh_name=mode+'_updated', save_path=save_path, update_face_colors=False)
    #
    # # Save mesh
    # if save_mesh: # p3d mesh won't save like this
    #     filename = os.path.join(save_path, "mesh_{}_elevation_{}.ply".format(mode ,gup_visualizer.sample_idx))
    #     gup_visualizer.mesh.export(filename, file_type='ply')
    #
    # # filename = os.path.join(save_path, "image_{}_{}.png".format(mode, gup_visualizer.sample_idx))
    # # ::rendered_image.save(filename, 'PNG')
    # return gup_visualizer, cam_p_p3d, rendered_image


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
                        help="Post-process vertex colors for improved color between ground-building boundary - SLOW!"
                        )

    args = parser.parse_args()
    return args

def get_configs():

    cfg = define_options()

    # PATHS

    # GT dataset
    dataset_root_path = os.path.join(cfg.data_root, cfg.dataset_name)

    # Get samples for diffusion
    path_diffusion_samples = os.path.join(cfg.path_root_diffusion, cfg.model_name_diffusion, 'test/epoch0035')
    samples_diffusion = sorted(glob.glob(os.path.join(path_diffusion_samples, '*_gt.npy')))

    # Get samples for HF
    path_hf_samples = os.path.join(cfg.path_root_hf, cfg.model_name_hf, cfg.dataset_name, 'diffusion/pred_depth')
    samples_hf = sorted(glob.glob(os.path.join(path_hf_samples, '*_pred.npy')))

    # Save path
    save_path = os.path.join(cfg.save_path, 'testing_renders')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)


    return {"dataset_root": dataset_root_path,
            "samples_diffusion": samples_diffusion,
            "samples_hf": samples_hf,
            "save_path": save_path,
            "dataset_name": cfg.dataset_name,
            "image_size": cfg.image_size,
            "num_samples": cfg.num_samples,
            "fix_mesh_colors": cfg.fix_mesh_colors
            }




if __name__ == '__main__':

    # # data path
    # data_path = '/home/Projects/groundup_visualizer_local/data'
    #
    # # Get paths
    # # model_name = 'test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3' #'test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3' #'test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3' #'v15_no_guide_pt_l1l2norm_max5.2_lr1'
    # model_name = 'test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3_hf' #'test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3' #'test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3' #'v15_no_guide_pt_l1l2norm_max5.2_lr1'
    #
    # path_to_samples = os.path.join(data_path, 'models', model_name)
    # samples = sorted(glob.glob(os.path.join(path_to_samples, '*.npy')))
    #
    # dataset_name =  'sf_test_1k_2' #'sf_pilot_demo_delete_3' # 'sf_test_1k_2'
    # dataset_root_path = os.path.join(data_path, dataset_name)
    #
    # # Save path
    # model_name = model_name + '_DEBUG'
    #
    # save_path = os.path.join(data_path, 'models', model_name, "viz")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path, exist_ok=True)

    # # # of samples to visualize
    # num_samples = 1
    # for sample in samples[:num_samples]:
    #     # Start timer
    #     start_time = time.time()
    #     # Run visualization
    #     gup_viz, cam_p, rendered_image = main(sample,
    #                                           dataset_root=dataset_root_path,
    #                                           save_path=save_path,
    #                                           scene_name=dataset_name)
    #     # End timer
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print(f"Execution time: {execution_time} seconds")


    run_config = get_configs()

    start_time = time.time()
    # Run visualization
    main(run_cfg=run_config)
    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")



# --data_root
# /home/Projects/groundup_visualizer_local/data
# --dataset_name
# sf_test_1k_2
# --path_root_diffusion
# /home/Projects/groundup_visualizer_local/data/models
# --model_name_diffusion
# test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3
# --path_root_hf
# /home/Projects/groundup_visualizer_local/data/models
# --model_name_hf
# HF_baseline
# --save_path
# /home/Projects/groundup_visualizer_local/data/models
