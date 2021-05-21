# DeepTag 
This is the project page of the CVPR 2021 oral paper DeepTag ([[arXiv](https://arxiv.org/abs/2103.02772)][[Video](https://www.youtube.com/watch?v=pCG4pbkllrs)]).
## DeepTag: An Unsupervised Deep Learning Method for Motion Tracking on Cardiac Tagging Magnetic Resonance Images.

We propose a fully unsupervised deep learning-based method for regional myocardium motion estimation on cardiac tagging magnetic resonance images (t-MRI). We incorporate the concept of motion decomposition and recomposition in our framework and achieve significant superior performance over traditional methods.

<div align=center><img width="650" height="300" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/MT_tmri.png"/></div>

## Supplementary results

1. Tagging image sequence registration results: (upper-left) tagging image sequence; (upper-right) forward registration results; (bottom-left) backward registration results; (bottom-right) Lagrangian registration results. The blue grid lines are to aid visual inspection. 
<div align=center><img width="300" height="300" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/SAX_STACK_45_23_reg_img.gif"/></div>

2. Landmarks tracking results: red is ground truth, green is prediction. (left) basal slice (On the septum wall, which is between RV and LV, tags may apparently disappear in some frames, due to through-plane motion, as do the ground truth landmarks, but we still show the predicted landmarks on the closest position); (middle) middle slice; (right) apex slice. Note that our method can even track the motion on the last several frames very accurately in spite of the significant image quality degradation.
<div align=center><img width="200" height="200" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/SAX_STACK_43_21_lm_img.gif"/><img width="200" height="200" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/SAX_STACK_45_23_lm_img.gif"/><img width="200" height="200" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/SAX_STACK_48_26_lm_img.gif"/></div>

3. Interframe (INF) motion fields and Lagrangian motion fields represented as a "quiver" form. (left) INF motions; (right) Lagrangian motions. Note that our method accurately captures the back-and-forth motion (left) in the left ventricle myocardium wall during systole. Also note that our method can even track the right ventricle's motion accurately.
<div align=center><img width="200" height="200" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/SAX_STACK_45_23_eu_motion.gif"/><img width="200" height="200" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/SAX_STACK_45_23_lag_motion.gif"/></div>

4. Lagrangian motion fields: (left) x component; (right) y component.
<div align=center><img width="400" height="180" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/SAX_STACK_45_23_lag_motion_map.gif"/></div>

5. Tag grid tracking results on the short axis view: (left) tagging image sequence; (middle) warped virtual tag grid by the Lagrangian motion field; (right) virtual tag grid superimposed on tagging images. Note that the virtual tag grid has been aligned with the tag pattern at time t=0. As time goes on, the virtual tag grid is deformed by the predicted Lagrangian motion field and follows the underlying tag pattern in the images very well.
<div align=center><img width="600" height="200" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/SAX_STACK_45_23_tag_grid_img.gif"/></div>

6. Tag grid tracking results on the long axis view: (upper) tagging image sequence; (bottom) virtual tag grid superimposed on tagging images. (left) 2 chamber view; (middle) 3 chamber view; (right) 4 chamber view. Our method can track local myocardium motion on both short axis and long axis views, by which we could recover the 3D motion field of the heart wall.
<div align=center><img width="200" height="400" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/2_CH_11_15_tag_grid_img.gif"/><img width="200" height="400" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/3_CH_12_16_tag_grid_img.gif"/><img width="200" height="400" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/4_CH_10_14_tag_grid_img.gif"/></div>

## Acknowledgments
Our code implementation borrows heavily from [VoxelMorph](https://github.com/voxelmorph/voxelmorph).
