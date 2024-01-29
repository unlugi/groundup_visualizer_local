#!/bin/bash

# python rendering_my_own.py \
# --save_path debug_dump/visualization_final \
# --predictions_path /mnt/data_f/gunlu/Experiments/GroundUp/Diffusion/diff_results_v14/test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3/test/epoch0035/ \
# --debug_dump \
# --clip_with_visibility \
# --prefix ours; 

# python rendering_my_own.py \
# --save_path debug_dump/visualization_final \
# --predictions_path /mnt/data_f/gunlu/Experiments/GroundUp/Diffusion/diff_results_v14/test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3/test/epoch0035/ \
# --debug_dump \
# --clip_with_visibility \
# --prefix ablate_normals; 

python rendering_my_own.py \
--save_path debug_dump/visualization_mono \
--predictions_path /mnt/data_f/gunlu/Experiments/GroundUp/Diffusion/diff_results_v14/test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3/test/epoch0035/ \
--debug_dump \
--clip_with_visibility \
--prefix ours; 