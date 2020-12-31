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

exist_patient={}

exist_patient['RMS'] = {}

for kk in range(1, 7):
    exist_num = 1
    rms_vec = []
    rms_mm_vec = []
    rms_phase_vec = []
    rms_phase_mm_vec = []
    model = 'm'+str(kk)
    exist_patient['RMS'][model] = {}
    print(model)
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

        # exist_patient[model][str(exist_num)]['rms_slice_vec'] = str(rearrange_rms_slice_vec(rms_slice_vec))
        # exist_patient[model][str(exist_num)]['rms_mm_slice_vec'] = str(rearrange_rms_slice_vec(rms_mm_slice_vec))

        rms_phase_vec.append(rearrange_rms_slice_vec(rms_slice_vec))
        rms_phase_mm_vec.append(rearrange_rms_slice_vec(rms_mm_slice_vec))

        rms_slice = rms_slice/(shape[0]-1)
        rms_mm_slice = rms_mm_slice/(shape[0]-1)
        rms_vec.append(rms_slice)
        rms_mm_vec.append(rms_mm_slice)

        # exist_patient[model][str(exist_num)]['rms_slice'] = str(rms_slice)
        # exist_patient[model][str(exist_num)]['rms_mm_slice'] = str(rms_mm_slice)

        exist_num += 1
        # print(exist_num-1)

    # exist_patient[model]['rms_vec'] = str(rms_vec)
    # exist_patient[model]['rms_mm_vec'] = str(rms_mm_vec)
    exist_patient['RMS'][model]['rms']= str([np.mean(rms_vec), np.std(rms_vec, ddof=1)])
    exist_patient['RMS'][model]['rms_mm'] = str([np.mean(rms_mm_vec), np.std(rms_mm_vec, ddof=1)])
    exist_patient['RMS'][model]['rms_phase']= str([np.mean(rms_phase_vec, axis=0), np.std(rms_phase_vec, axis=0, ddof=1)])
    exist_patient['RMS'][model]['rms_phase_mm'] = str([np.mean(rms_phase_mm_vec, axis=0), np.std(rms_phase_mm_vec, axis=0, ddof=1)])
    # exist_patient[model]['rms_phase_vec']= str(rms_phase_vec)
    # exist_patient[model]['rms_phase_mm_vec'] = str(rms_phase_mm_vec)

with open(os.path.join(dst_path, 'Tagging_motion_tracking'+ '_rms_error.json'), 'w', encoding='utf-8') as f_json:
    json.dump(exist_patient, f_json, indent=4)

