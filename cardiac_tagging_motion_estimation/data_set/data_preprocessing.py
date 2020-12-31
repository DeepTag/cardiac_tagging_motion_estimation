import SimpleITK as sitk
import numpy as np
import os

def normalize_data_2d(img_np):
    # preprocessing
    t,y,x = img_np.shape
    for i in range(t):
        img = img_np[i,::]
        cmin = np.min(img)
        cmax = np.max(img)
        img_array = (img - cmin) / (cmax- cmin + 0.0001)
        img_np[i, ::] = img_array

    return img_np

def normalize_data_3d(img_np):
    # preprocessing
    cmin = np.min(img_np)
    cmax = np.max(img_np)
    img_np = (img_np - cmin) / (cmax- cmin + 0.0001)
    return img_np

def normalize_data(img_np):
    # preprocessing
    cm = np.median(img_np)
    img_np = img_np / (2*cm + 0.0001)
    img_np[img_np < 0] = 0.0
    img_np[img_np >1.0] = 1.0
    return img_np

src_root =  '/home/DeepTag/data/cropped/'
dst_root = '/home/DeepTag/data/normed/'

if not os.path.exists(dst_root): os.mkdir(dst_root)

for subroot, dirs, files in os.walk(src_root):
    if len(files)<3: continue
    patient_num_vec = subroot.split('/')
    patient_num = patient_num_vec[-3]
    print('working on %s', patient_num)
    tgt_root1 = os.path.join(dst_root, patient_num_vec[-3])
    if not os.path.exists(tgt_root1): os.mkdir(tgt_root1)
    tgt_root2 = os.path.join(tgt_root1, patient_num_vec[-2])
    if not os.path.exists(tgt_root2): os.mkdir(tgt_root2)
    tgt_root3 = os.path.join(tgt_root2, patient_num_vec[-1])
    if not os.path.exists(tgt_root3): os.mkdir(tgt_root3)

    for file in files:
        if file.endswith('nii.gz') and 'TAG' in file:
            tag_file0 = file
            tag_file = os.path.join(subroot, file)
            tag_image = sitk.ReadImage(tag_file)
            tag_image = sitk.Cast(tag_image, sitk.sitkFloat64)
            spacing1 = tag_image.GetSpacing()
            origin1 = tag_image.GetOrigin()
            direction1 = tag_image.GetDirection()
            np_img = sitk.GetArrayFromImage(tag_image)
            np_img_normed = normalize_data(np_img)
            mask1 = sitk.GetImageFromArray(np_img_normed)
            mask1.SetSpacing(spacing1)
            mask1.SetOrigin(origin1)
            mask1.SetDirection(direction1)
            sitk.WriteImage(mask1, os.path.join(tgt_root3, tag_file0))

        if file.endswith('nii.gz') and 'TAG' not in file:
            cine_file0 = file
            cine_file = os.path.join(subroot, file)
            cine_image = sitk.ReadImage(cine_file)
            cine_image = sitk.Cast(cine_image, sitk.sitkFloat64)
            spacing2 = cine_image.GetSpacing()
            origin2 = cine_image.GetOrigin()
            direction2 = cine_image.GetDirection()
            np_img = sitk.GetArrayFromImage(cine_image)
            np_img_normed = normalize_data(np_img)
            mask2 = sitk.GetImageFromArray(np_img_normed)
            mask2.SetSpacing(spacing2)
            mask2.SetOrigin(origin2)
            mask2.SetDirection(direction2)
            sitk.WriteImage(mask2, os.path.join(tgt_root3, cine_file0))