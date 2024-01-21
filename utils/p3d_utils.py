import torch
import numpy as np

from pytorch3d.renderer import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.renderer import ( RasterizationSettings,
                                 MeshRasterizer, MeshRenderer,
                                 SoftPhongShader, HardPhongShader, HardFlatShader,
                                 AlphaCompositor )
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer, PointsRenderer
from pytorch3d.renderer import PointLights, DirectionalLights, AmbientLights


def define_camera(K, imsize, R_for_camera_topdown, t_for_camera_topdown, device):
    fx, fy, px, py = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
    height, width = imsize

    cameras = PerspectiveCameras(R=R_for_camera_topdown, T=t_for_camera_topdown,
                                            focal_length=((fx, fy),),
                                            principal_point=((px, py),),
                                            in_ndc=False,
                                            image_size=((height, width),),
                                            device=device,
                                        )
    return cameras


# mesh renderer
def mesh_renderer(cameras, imsize, is_depth=False, bg_color_white=True, device='cpu', offset=(0, 0, 0)):

    height, width = imsize

    # if bg_color_white:
    #     bg_color = torch.tensor([1.0, 1.0, 1.0], device="cpu")
    # else:
    #     bg_color = torch.tensor([0.7, 0.7, 1.0], device="cpu") # light blue

    # Configure rasterization settings
    mesh_raster_settings = RasterizationSettings( image_size=(height, width),
                                                  blur_radius=0.0,
                                                  faces_per_pixel=1,

    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=mesh_raster_settings)

    # if is_depth:
    #     return rasterizer
    # else:
    #     if bg_color_white:
    #         bg_color = torch.tensor([1.0, 1.0, 1.0], device="cpu")
    #     else:
    #         bg_color = torch.tensor([0.7, 0.7, 1.0], device="cpu") # light blue
    #
    #     renderer = MeshRenderer(rasterizer=rasterizer,
    #                                 shader=SoftPhongShader(device=cameras.device)
    #                                 )
    #     return renderer

    # Get the camera's position + offset location if needed
    offset = torch.tensor((offset,), device=cameras.device)
    camera_position = cameras.get_camera_center() + offset

    # Define a light source at the camera's position
    point_lights = PointLights(device=device,
                               location=((camera_position[0,0], camera_position[0,1], camera_position[0,2]) , ),
                               ambient_color=((0.6, 0.6, 0.6),),
                               diffuse_color=((0.4, 0.4, 0.4),),
                               specular_color=((0.1, 0.1, 0.1),),
                               # ambient_color=((0.95, 0.95, 1.0), ),
                               # diffuse_color=((0.4, 0.4, 0.4), ),
                               # specular_color=((0.2, 0.2, 0.2),)
                               )

    renderer = MeshRenderer(rasterizer=rasterizer,
                            # shader=SoftPhongShader(device=cameras.device,
                            #                        cameras=cameras,
                            #                        # lights=point_lights,
                            #                        blend_params=BlendParams(sigma=1e-4, gamma=1e-4)
                            # )
                            shader=HardPhongShader(device=cameras.device,
                                                   cameras=cameras,
                                                   lights=point_lights,
                                    )
                            )

    return renderer