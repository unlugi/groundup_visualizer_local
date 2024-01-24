#!/bin/bash

python print_depth_scores.py \
-s debug_dump/test231202_233841_v14_no_guide_pt_l1l2_max5.2_lr3/depth_metrics.json \
-s debug_dump/test231202_234647_v14_guide_gt_fs_rn_fs_vae_l1l2_max5.2_lr3/depth_metrics.json \
-s debug_dump/test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3/depth_metrics.json \
-s debug_dump/test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3/depth_metrics.json \
-s debug_dump/heightfields/scores/depth_metrics.json  \
-p "\$S_t\$ (pt)" \
-p "\$S_t\$+\$Dt\$ (fs)" \
-p "\$S_t\$+\$Dt\$ (pt)" \
-p "\$S_t\$+\$Dt\$ (pt) + $\mathcal{L}_\textrm{norm}$" \
-p "\textit{HeightFields}\shortcite{watson2023wacv}";

# python print_depth_scores.py \
# -s debug_dump/test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3/depth_metrics.json \
# -s debug_dump/heightfields/scores/depth_metrics.json  \
# -p "\$S_t\$+$Dt\$ (pt) + $\mathcal{L}_\textrm{norm}$" \
# -p "\textit{HeightFields}\shortcite{watson2023wacv}";