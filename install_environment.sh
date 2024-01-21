#!/bin/bash

# Step 1: Create and activate a conda environment
conda create -n blender_vis python=3.10

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blender_vis

# Step 2: Install bpy
pip install bpy==3.4.0

# Step 3: Install PyTorch and torchvision
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia

# Step 4: Install fvcore and iopath
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

# Step 5: Install Jupyter
conda install jupyter

# Step 6: Install additional packages
pip install scikit-image matplotlib imageio plotly opencv-python

# Step 7: Install pytorch3d
conda install pytorch3d -c pytorch3d

# Step 8: Install kornia
pip install kornia

# Step 9: Install numba
pip install numba

# Step 10: Install trimesh
pip install trimesh