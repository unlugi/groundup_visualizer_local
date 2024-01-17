import os
import glob
import time

from visualizers.visualizer_p3d import GroundUpVisualizerP3D as GroundUpVisualizer


def main(sample_path, dataset_root, save_path, scene_name=None):

    add_mesh_color = True

    print(sample_path)
    gup_visualizer = GroundUpVisualizer(sample_path=sample_path,
                                        dataset_root=dataset_root,
                                        scene_name=scene_name,
                                        add_color_to_mesh=add_mesh_color,
                                        device='cuda')
    save_mesh = False
    # mode = 'gt'
    mode = 'pred'

    # Generate the mesh
    gup_visualizer.get_mesh_in_world_coordinates(model_name=mode)

    # Run meshing
    offset = (-3.0, 0.0, 4.0)
    cam_p_p3d, rendered_image = gup_visualizer.render_scene(image_size=(1024, 1024), offset=offset)

    # gup_visualizer.export_mesh_p3d(mesh_name=mode+'_updated', save_path=save_path, update_face_colors=True)

    # Save mesh
    if save_mesh: # p3d mesh won't save like this
        filename = os.path.join(save_path, "mesh_{}_elevation_{}.ply".format(mode ,gup_visualizer.sample_idx))
        gup_visualizer.mesh.export(filename, file_type='ply')

    filename = os.path.join(save_path, "image_{}_{}.png".format(mode, gup_visualizer.sample_idx))
    rendered_image.save(filename, 'PNG')

    return gup_visualizer, cam_p_p3d, rendered_image

"""
1) read gt, proj, pred, sketch_td, sketch_p, cam_td, cam_p
2) mesh gt
3) mesh pred
4) pc proj
5) Render from cam_p
"""
if __name__ == '__main__':

    # data path
    data_path = '/home/Projects/groundup_visualizer_local/data'

    # Get paths
    model_name = 'test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3' #'test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3' #'test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3' #'v15_no_guide_pt_l1l2norm_max5.2_lr1'
    path_to_samples = os.path.join(data_path, 'models', model_name)
    samples = sorted(glob.glob(os.path.join(path_to_samples, '*.npy')))

    dataset_name =  'sf_test_1k_2' #'sf_pilot_demo_delete_3' # 'sf_test_1k_2'
    dataset_root_path = os.path.join(data_path, dataset_name)

    # Save path
    model_name = model_name + '_testing'

    save_path = os.path.join(data_path, 'models', model_name, "viz")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # # of samples to visualize
    num_samples = 1
    for sample in samples[:num_samples]:
        # Start timer
        start_time = time.time()
        # Run visualization
        gup_viz, cam_p, rendered_image = main(sample,
                                              dataset_root=dataset_root_path,
                                              save_path=save_path,
                                              scene_name=dataset_name)
        # End timer
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        # Plot mesh
        # plotly_vis.plot_batch_individually([gup_viz.mesh, cam_p])

    # cam_p_p3d, rendered_image = gup_viz.render_scene(image_size=(512, 512), )
