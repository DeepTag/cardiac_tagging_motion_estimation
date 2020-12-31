import matplotlib.pyplot as plt

# """""
import os
import SimpleITK as sitk
import numpy as np
import json

def rearrange_rms_slice_vec(rms_slice_vec):
    p = len(rms_slice_vec)
    N =10
    step = p/N
    new_vec = []
    for k in range(1, N+1):
        v = 0
        n = 0
        for q in range(1, p+1):
            if q>(k-1)*step and q<=k* step:
                v += rms_slice_vec[q-1]
                n +=1
        new_vec.append(v/n)
    return new_vec


lm_path ='/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/models/lm/'

dst_path = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/models/results/tracking_error/'

if not os.path.exists(dst_path): os.makedirs(dst_path)

mean = []
std = []
models = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
for model in models:
    print(model)
    rms_vec = []
    rms_mm_vec = []
    rms_phase_vec = []
    rms_phase_mm_vec = []
    for subroot, dirs, files in os.walk(lm_path):
        if len(files) < 13: continue
        root_vec = subroot.split(os.path.sep)

        patient_num = root_vec[-3]+'_'+root_vec[-2]+'_'+root_vec[-1]

        # exist_patient[model][str(exist_num)] = {}
        # exist_patient[model][str(exist_num)]['patient_num']=patient_num

        lm4_file = False
        for file in files:
            if file.endswith('nii.gz') and 'lm4' in file:
                gt_tag_file = os.path.join(subroot, file)
                gt_sitk_tag_image = sitk.ReadImage(gt_tag_file)
                # lm4_file = True
            if file.endswith('nii.gz') and 'lm_'+model in file:
                tag_file = os.path.join(subroot, file)
                sitk_tag_image = sitk.ReadImage(tag_file)
                # lm4_file = True
                # break

        gt_tag_image = sitk.GetArrayFromImage(gt_sitk_tag_image)
        spacing = gt_sitk_tag_image.GetSpacing()
        shape = gt_tag_image.shape
        tag_image = sitk.GetArrayFromImage(sitk_tag_image)
        lm0 = gt_tag_image[0,::]
        n_gt_lm = 40
        for k in range(n_gt_lm, 0, -1):
            p = np.where(lm0==k)
            if len(p[0]) > 0:
                n_gt_lm = k
                break
        rms_slice = 0
        rms_mm_slice = 0
        rms_slice_vec = []
        rms_mm_slice_vec= []
        for s in range(1, shape[0]):
            rms_phase = 0
            rms_mm_phase = 0
            real_n = 0
            for k in range (1, n_gt_lm+1):
                gt_lm = np.where(gt_tag_image[s,::]==k)
                pre_lm = np.where(tag_image[s,::]==k)
                if len(gt_lm[0]) == 0 or len(pre_lm[0])==0: continue
                gt_lm_mean = np.mean(gt_lm, axis=1)
                pre_lm_mean = np.mean(pre_lm, axis=1)
                rms = np.sqrt(np.sum(np.power((gt_lm_mean - pre_lm_mean), 2)))
                rms_mm = rms*spacing[1]
                rms_phase += rms
                rms_mm_phase += rms_mm
                real_n += 1
            rms_phase = rms_phase/real_n
            rms_mm_phase = rms_mm_phase/real_n
            rms_slice_vec.append(rms_phase)
            rms_mm_slice_vec.append(rms_mm_phase)
            rms_slice += rms_phase
            rms_mm_slice += rms_mm_phase
        rms_phase_vec.append(rearrange_rms_slice_vec(rms_slice_vec))
        rms_phase_mm_vec.append(rearrange_rms_slice_vec(rms_mm_slice_vec))
        rms_slice = rms_slice/(shape[0]-1)
        rms_mm_slice = rms_mm_slice/(shape[0]-1)
        rms_vec.append(rms_slice)
        rms_mm_vec.append(rms_mm_slice)

    mean.append(np.mean(rms_phase_mm_vec, axis=0))
    std.append(np.std(rms_phase_mm_vec, axis=0, ddof=1))

fig = plt.figure()
x = np.linspace(5, 95, 10)
y_sin = np.sin(x)
y_cos = np.cos(x)

'''
plt.errorbar(x, mean[0], std[0], capsize=10,linewidth=1.5, label='HARP')
plt.errorbar(x, mean[1], std[1], capsize=10,linewidth=1.5, label='OF-TV')
plt.errorbar(x, mean[2], std[2], capsize=10,linewidth=1.5, label='VM (SSD)')
plt.errorbar(x, mean[3], std[3], capsize=10,linewidth=1.5, label='VM (NCC)')
plt.errorbar(x, mean[4], std[4], capsize=10,linewidth=1.5, label='VM-DIF')
plt.errorbar(x, mean[5], std[5], capsize=10,linewidth=1.5, label='Ours', color='#bcbd22')
'''
plt.errorbar(x, mean[0], std[0], capsize=10,linewidth=1.5, label='A1')
plt.errorbar(x, mean[1], std[1], capsize=10,linewidth=1.5, label='A2')
plt.errorbar(x, mean[2], std[2], capsize=10,linewidth=1.5, label='A3')
plt.errorbar(x, mean[3], std[3], capsize=10,linewidth=1.5, label='A4')
plt.errorbar(x, mean[4], std[4], capsize=10,linewidth=1.5, label='A5')
plt.errorbar(x, mean[5], std[5], capsize=10,linewidth=1.5, label='A6')
plt.errorbar(x, mean[6], std[6], capsize=10,linewidth=1.5, label='Ours', color='#bcbd22')


plt.grid()
plt.legend(loc='upper left')
plt.xlabel('% cardiac cycle')
plt.ylabel('RMS error (mm)')
plt.axis([0, 100, 0, 8.25])
plt.xticks(range(5, 100, 10), ['5', '15', '25', '35', '45', '55', '65', '75', '85', '95'])
plt.show()
plt.margins(1.2, 1.2)
fig.savefig(os.path.join(dst_path, 'tracking_error2_2.pdf'), bbox_inches='tight', pad_inches=0.0, dpi=300, quality=95)
plt.close()

