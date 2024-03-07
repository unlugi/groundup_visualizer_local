import os
import time
import glob
import numpy as np
import yaml

import bpy
import math
from mathutils import Matrix

debug = False # True if you want to run in Blender GUI

class RendererBlender:
    def __init__(self, cfg, cfg_vis, cameras, mode='gt'):
        self.cfg = cfg
        self.cfg_vis = cfg_vis
        self.cameras = cameras
        self.mode = mode # 'gt' or 'pred' mesh
        # self.data_paths = self.prepare_data_paths()
        self.set_rendering_settings()
        self.set_light_settings()
        self.initialize_cameras()
        # self.import_input_model()

    def render(self, save_path):

        # Set rendering save path
        # bpy.context.scene.render.filepath = r'//{}'.format(self.data_paths['path_save_render'])
        bpy.context.scene.render.filepath = save_path

        # Render still
        bpy.ops.render.render(write_still=True)

    def prepare_data_paths(self):

        # Root path to repo where code and data lives
        path_root_repo = self.cfg['PATHS']['PATH_TO_REPO']

        # Convert path to WSL path if needed (for debugging in GUI)
        if self.cfg['DEBUG']['USE_WSL_PATHS'] and self.cfg['DEBUG']['USE_DEBUG']:
            path_root_repo = bpy.path.abspath(os.path.normpath(path_root_repo))
            path_root_repo = os.path.join(r'{}'.format(self.cfg['DEBUG']['WSL_MOUNT']), path_root_repo)

        # Path to topdown depth samples - for predictions of the diffusion model we are visualizing
        path_samples = os.path.join(path_root_repo, 'data', 'models', self.cfg['PATHS']['PATH_TO_MODEL'])

        # Path to mesh
        path_mesh = glob.glob(os.path.join(path_samples, 'mesh_{}_*'.format(self.mode)))[0]

        # Path to dataset - the test dataset for predictions
        path_root_dataset = os.path.join(path_root_repo, "data", self.cfg['PATHS']['DATASET_NAME'])

        # Sample id
        # sample_idx = path_mesh.split('/')[-1].split('.')[0].split('_')[-1]
        sample_idx = path_mesh.split('mesh_')[-1].split('.')[0].split('_')[-1]

        # Path to camera_perspective_raw - camera pose for the sample
        path_cam_perspective = os.path.join(path_root_dataset, 'Camera', 'camera', 'campose_raw_{}.npz'.format(sample_idx))

        # Save path for renders
        path_save_render = os.path.join(path_samples, 'blender_renders', 'rgb_{}_{}.png'.format(self.mode, sample_idx))

        return {'path_root_repo': path_root_repo,
                'path_samples': path_samples,
                'path_mesh': path_mesh,
                'path_root_dataset': path_root_dataset,
                'sample_idx': sample_idx,
                'path_cam_perspective': path_cam_perspective,
                'path_save_render': path_save_render}

    def set_rendering_settings(self, view_layer_name="ViewLayer"):
        # Set up rendering
        scene = bpy.context.scene
        render = bpy.context.scene.render

        render.engine = self.cfg['SETTINGS']['ENGINE']
        render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
        render.image_settings.color_depth = self.cfg['RENDER']['IMAGE']['COLOR_DEPTH']  # ('8', '16')
        render.image_settings.file_format = self.cfg['RENDER']['IMAGE']['FORMAT']  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
        render.resolution_x = self.cfg['RENDER']['RES_X']
        render.resolution_y = self.cfg['RENDER']['RES_Y']
        render.resolution_percentage = 100
        render.film_transparent = True

        scene.use_nodes = True
        scene.view_layers[view_layer_name].use_pass_normal = True
        scene.view_layers[view_layer_name].use_pass_diffuse_color = True
        scene.view_layers[view_layer_name].use_pass_object_index = True
        scene.view_layers[view_layer_name].use_pass_material_index = True


    def set_light_settings(self ):

        # Delete existing lights
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()

        # Get the current camera and its location
        camera = bpy.context.scene.camera
        camera_location = camera.location

        # Calculate the new location for the new light
        angle_offset = math.radians(self.cfg['LIGHTING']['LOCATION_ANGLE_OFFSET'])  # 30 degrees to radians
        light_location = (camera_location.x + math.cos(angle_offset) * self.cfg['LIGHTING']['LOCATION_DISTANCE_OFFSET'],
                          camera_location.y + math.sin(angle_offset) * self.cfg['LIGHTING']['LOCATION_DISTANCE_OFFSET'],
                          camera_location.z + self.cfg['LIGHTING']['LOCATION_HEIGHT_OFFSET'])

        # Create a new lamp
        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=light_location)

        # Get the light object in the scene
        light = [objs for objs in bpy.context.scene.objects if objs.type == 'LIGHT'][0]

        # Set the light settings (sun, energy, color, angle, max_bounces)
        light.data.type = 'SUN' if self.cfg['LIGHTING']['USE_SUN']  else 'POINT'
        light.data.use_shadow = True
        light.data.energy = self.cfg['LIGHTING']['STRENGTH']
        light.data.angle = math.radians(self.cfg['LIGHTING']['DIRECTION_ANGLE']) # sun
        # light.data.shadow_soft_size = 150 # point

        #light.data.specular_factor = 1.015
        #light.data.color = (1, 0.74561, 0.391147)

        # CYCLES-only setting
        if self.cfg['SETTINGS']['ENGINE'] == 'CYCLES':
            light.data.cycles.max_bounces = 1014


    def import_input_model(self, path_mesh, collection_name='Collection'):

        # # Parse filepath to extract the file extension
        # file_path = self.data_paths['path_mesh']
        # file_name = file_path.split('/')[-1]
        # file_extension = file_name.split('.')[-1]

        # Parse filepath to extract the file extension
        file_path = path_mesh
        file_name = file_path.split('/')[-1]
        file_extension = file_name.split('.')[-1]


        if file_extension == 'obj':
            # bpy.ops.import_scene.obj(filepath=file_path)
            bpy.ops.wm.obj_import(filepath=file_path)
        elif file_extension == 'ply':
            bpy.ops.import_mesh.ply(filepath=file_path)
        elif file_extension == 'blend':
            # prepare files list for bpy.ops.wm.append(..)
            files = []
            with bpy.data.libraries.load(file_path) as (data_from, data_to):
                for name in data_from.objects:
                    files.append({'name': name})

            bpy.ops.wm.append(directory=file_path + "/Object/", files=files)
        else:
            print('Error! Input 3D file format not recognized/supported.')

        bpy.context.evaluated_depsgraph_get().update()

    def enable_material(self, mode):
        # Enable color

        # create new material
        mat = bpy.data.materials.new('Material.001')
        mat.use_nodes = True

        # Assign material to mesh object
        current_mesh = bpy.data.objects['mesh_{}'.format(mode)]
        current_mesh.data.materials.append(mat)

        # Get material nodes
        nodes = mat.node_tree.nodes

        # Create input and output nodes
        node_color_input = nodes.new(type='ShaderNodeVertexColor')
        node_color_input.layer_name = "Color"
        node_output = nodes.new(type='ShaderNodeOutputMaterial')

        # linking
        links = mat.node_tree.links
        link_color = links.new(node_color_input.outputs['Color'], nodes.get("Principled BSDF").inputs['Base Color'])
        link_output = links.new(nodes.get("Principled BSDF").outputs['BSDF'],  node_output.inputs['Surface'])


    def initialize_cameras(self, collection_name="Collection"):

        # Read camera pose
        # camera_matrix_raw = np.load(self.data_paths['path_cam_perspective'])['data']
        camera_matrix_raw = self.cameras['cam_perspective_raw'].copy()

        # User Camera
        cam = bpy.context.scene.camera # 'Camera'

        cam.data.lens = 20
        cam.data.sensor_width = 36
        cam.data.clip_start = self.cfg['CAMERA']['CLIP_START']
        cam.data.clip_end = self.cfg['CAMERA']['CLIP_END'] # depending on dataset 3 or 4
        cam.data.display_size = 0.1 # for viewport visualization in Blender GUI
        cam.data.type = 'PERSP'

        bpy.context.scene.camera = cam  # Set the active camera in the scene for now.

        # reset location and rotation
        cam.rotation_euler = (0, 0, 0)
        cam.location = (0, 0, 0)

        # Get camera object and set its position and rotation

        # Decompose the matrix_world of cam
        cam_matrix_world = Matrix(camera_matrix_raw.tolist())
        location, rotation, scale = cam_matrix_world.decompose()

        # Rotation
        # 180 degree rot around z axis - coordinate change
        rot_z_180 = Matrix(((-1, 0, 0, 0),
                             (0, -1, 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1)))
        cam.matrix_world = rot_z_180 @ cam_matrix_world

        # Location
        change_of_coordinates = Matrix([[0, 1, 0],[1, 0, 0],[0, 0, 1]])
        cam.location = change_of_coordinates @ location

        # Scale
        cam.scale = (1,1,1) #scale

        # Update the scene to see the changes in camera pose
        bpy.context.view_layer.update()


    def assign_vertex_colors_to_mesh(self, vertex_colors_n4):
        current_mesh_data = bpy.context.scene.objects['mesh_gt'].data

        # This is to reference the vertex color layer later
        vertex_colors_name = "vert_colors"

        # Here the color layer is made on the mesh
        current_mesh_data.vertex_colors.new(name=vertex_colors_name)

        # We define a variable that is used to easily reference
        # the color layer in code later
        color_layer = current_mesh_data.vertex_colors[vertex_colors_name]

        # We loop over all the polygons
        for polygon in current_mesh_data.polygons:
            for i, index in enumerate(polygon.vertices):
                loop_index = polygon.loop_indices[i]
                current_mesh_data.vertex_colors.active.data[loop_index].color = vertex_colors_n4[i]


def test_mesh_blender(run_in_gui):

    path_to_repo = '/home/Projects/groundup_visualizer_local/'

    # 0) Load render settings configuration file
    config_path = os.path.join(path_to_repo, 'configs', 'config_UrbanScene3D_visualization.yaml')
    if run_in_gui:
        mount_path = r'//wsl.localhost/Ubuntu/'
        config_path = bpy.path.abspath(os.path.normpath(config_path))
        config_path = os.path.join(mount_path, config_path)
    configs = yaml.safe_load(open(config_path))


    # 1) Initialize the renderer class
    # renderer = RendererBlender(configs, mode='gt')
    renderer = RendererBlender(configs, mode='pred')

    # 2) Render mesh
    renderer.render()


if __name__ == '__main__':
    # Start timer
    start_time = time.time()

    # Run test code
    run_in_gui = debug
    test_mesh_blender(run_in_gui)

    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

