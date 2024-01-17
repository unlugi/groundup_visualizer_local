# Results visualizer for GroundUp

###### TODO: this will be in similar structure to Pytorch3D renderer
###### Priority scale less to more <0 - 9>
###### Need the gup environment - test in Blender in UI
###### Can't elevate in blender?

## Render Engine Blender

### 1) run_viz_blender.py: the script for running the visualizer
### 2) visualizer/visualizer_blender.py: Visualizer Class with Blender as the rendering engine
### 3) result path loading - same as p3d
### 4) Meshing: HF_mesher or HF_mesher+Blender(Geometry Modifiers) or Blender Meshing only (TODO:IMPORTANT! - 5)
### 5) GT-mesh (save for pilot/test datasets?) - or used Elevation meshes
### 6) Lighting and Materials are important! (TODO: IMPORTANT! 9)
### 7) Make buildings snap to ground no matter what! (TODO: IMPORTANT! Priority: 8)
### 8) Cameras: Use Blender Cameras (camera_pose_raw.npy? ) - shouldn't be too hard (TODO: IMPORTANT! Priority: 9)


