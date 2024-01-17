import os
import glob
import time

from visualizers.visualizer_blender import GroundUpVisualizerBlender as GroundUpVisualizer


def main(sample_path, dataset_root, save_path, scene_name=None):

    # add color to mesh
    add_mesh_color = True

    # Initialize visualizer
    gup_visualizer = GroundUpVisualizer(sample_path=sample_path,
                                        dataset_root=dataset_root,
                                        scene_name=scene_name,
                                        add_color_to_mesh=add_mesh_color,
                                        device='cuda'
                                        )

    # Set mode
    mode = 'pred'

    # Generate the mesh
    gup_visualizer.get_mesh_in_world_coordinates(model_name=mode)

    # Get the blender mesh
    gup_visualizer.convert_to_blender_mesh()

    # # Run meshing
    # offset = (-3.0, 0.0, 4.0)
    # cam_p_p3d, rendered_image = gup_visualizer.render_scene(image_size=(1024, 1024), offset=offset)
    #
    # Export mesh
    gup_visualizer.export_mesh_p3d(mesh_name="mesh_{}_{}.obj".format(mode, gup_visualizer.sample_idx), save_path=save_path, update_face_colors=False)

    #
    # # Save render image
    # filename = os.path.join(save_path, "image_{}_{}.png".format(mode, gup_visualizer.sample_idx))
    # rendered_image.save(filename, 'PNG')

    return gup_visualizer, #cam_p_p3d, rendered_image


if __name__ == '__main__':

    # Get paths
    model_name = 'test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3'
    path_to_samples = os.path.join('data', 'models', model_name)
    samples = sorted(glob.glob(os.path.join(path_to_samples, '*.npy')))

    dataset_name = 'sf_test_1k_2' #'sf_pilot_demo_delete_3' # 'sf_test_1k_2'
    dataset_root_path = os.path.join("data", dataset_name)

    # Save path
    model_name = model_name + '_testing_BLENDER'
    save_path = os.path.join("data", 'models', model_name, "viz")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # # of samples to visualize
    num_samples = 1
    for sample in samples[:num_samples]:
        # Start timer
        start_time = time.time()
        # Run visualization
        # gup_viz, cam_p, rendered_image = main(sample,
        #                                       dataset_root=dataset_root_path,
        #                                       save_path=save_path,
        #                                       scene_name=dataset_name)
        gup_viz = main(sample,
                      dataset_root=dataset_root_path,
                      save_path=save_path,
                      scene_name=dataset_name)


        # End timer
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

