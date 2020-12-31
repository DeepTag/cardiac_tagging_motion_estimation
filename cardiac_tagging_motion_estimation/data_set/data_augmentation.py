import SimpleITK as sitk
from scipy import ndimage
import cv2
import numpy as np
import os

def generate_tag_grid_sin(N, sx, sy, k):
    M = 3*N
    grid = np.zeros((M, M))

    for a in range(0, k):
        for i in range(sx+a, M, k):
            for j in range(i, -1, -1):
                grid[j, i-j] += 0.5*np.sin(np.pi*a/(k))

    for a in range(0, k):
        for i in range(sy+a, M, k):
            for j in range(i, M, 1):
                grid[j-i, j] += 0.5*np.sin(np.pi*a/(k))

    return grid[0:N, N: 2*N]

def generate_tag_grid(N, sx, sy, k):
    M = 3*N
    grid = np.ones((M, M))
    b = 3

    for a in range(0, b):
        for i in range(sx+a, M, k):
            for j in range(i, -1, -1):
                grid[j, i-j] = 0.4*1*abs(a-1)

    for a in range(0, b):
        for i in range(sy+a, M, k):
            for j in range(i, M, 1):
                grid[j-i, j] = 0.4*1*abs(a-1)

    return grid[0:N, N: 2*N]

def data_augment(image, label, shift=10.0, rotate=10.0, scale=0.1, intensity=0.1, flip=False):
    # Perform affine transformation on image and label, which are 4D tensors of dimension (N, C, X, Y).
    image2 = np.zeros(image.shape, dtype='float32')
    label2 = np.zeros(label.shape, dtype='float32')
    for i in range(image.shape[0]):
        # Random affine transformation using normal distributions
        shift_var = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
        rotate_var = np.clip(np.random.normal(), -3, 3) * rotate
        scale_var = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_var = 1 + np.clip(np.random.normal(), -0.5, 0) * intensity

        # Apply affine transformation (rotation + scale + shift) to training images
        row, col = image.shape[2:]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_var, 1.0 / scale_var)
        M[:, 2] += shift_var
        for c in range(image.shape[1]):
            image2[i, c] = ndimage.interpolation.affine_transform(image[i, c], M[:, :2], M[:, 2], order=1)
            label2[i, c] = ndimage.interpolation.affine_transform(label[i, c], M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        if np.random.uniform() >= 0.67:
            image2[i, :] *= intensity_var
            label2[i, :] *= intensity_var

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.67:
                image2[i, :] = image2[i, :, ::-1, :]
                label2[i, :] = label2[i, :, ::-1, :]
            elif np.random.uniform() <= 0.33:
                image2[i, :] = image2[i, :, :, ::-1]
                label2[i, :] = label2[i, :, :, ::-1]
    return image2, label2

src_root =  '/home/DeepTag/data/normed/'
dst_root = '/home/DeepTag/data/auged/'

if not os.path.exists(dst_root): os.mkdir(dst_root)
aug_num = 20

for subroot, dirs, files in os.walk(src_root):
    if len(files)<2: continue
    patient_num_vec = subroot.split('/')
    patient_num = patient_num_vec[-3]
    print('working on %s', patient_num)
    tgt_root1 = os.path.join(dst_root, patient_num_vec[-3])
    if not os.path.exists(tgt_root1): os.mkdir(tgt_root1)

    for file in files:
        if file.endswith('nii.gz') and 'TAG' in file:
            tag_file0 = file
            tag_file = os.path.join(subroot, file)
            tag_image = sitk.ReadImage(tag_file)
            tag_image = sitk.Cast(tag_image, sitk.sitkFloat64)
            spacing1 = tag_image.GetSpacing()
            origin1 = tag_image.GetOrigin()
            direction1 = tag_image.GetDirection()
            np_tag_img = sitk.GetArrayFromImage(tag_image)

        if file.endswith('nii.gz') and 'TAG' not in file:
            cine_file0 = file
            cine_file = os.path.join(subroot, file)
            cine_image = sitk.ReadImage(cine_file)
            cine_image = sitk.Cast(cine_image, sitk.sitkFloat64)
            spacing2 = cine_image.GetSpacing()
            origin2 = cine_image.GetOrigin()
            direction2 = cine_image.GetDirection()
            np_cine_img = sitk.GetArrayFromImage(cine_image)

    tag1 = generate_tag_grid(N=192, sx=1, sy=10, k=12)
    tag2 = generate_tag_grid_sin(N=192, sx=1, sy=10, k=12)
    tag_grid = np.concatenate((tag1[np.newaxis], tag2[np.newaxis]))
    np_tag_img = np.concatenate((tag_grid, np_tag_img))
    np_cine_img = np.concatenate((tag_grid, np_cine_img))

    for i in range(aug_num):
        tgt_root1_aug = os.path.join(tgt_root1, 'aug_'+ str(i))
        if not os.path.exists(tgt_root1_aug): os.mkdir(tgt_root1_aug)

        tgt_root2 = os.path.join(tgt_root1_aug, patient_num_vec[-2])
        if not os.path.exists(tgt_root2): os.mkdir(tgt_root2)
        tgt_root3 = os.path.join(tgt_root2, patient_num_vec[-1])
        if not os.path.exists(tgt_root3): os.mkdir(tgt_root3)

        if i==0:
            np_tag_img2 = np_tag_img
            np_cine_img2 = np_cine_img
        else:
            np_tag_img2, np_cine_img2 = data_augment(np_tag_img[np.newaxis], np_cine_img[np.newaxis], shift=10.0, rotate=10.0, scale=0.1, \
                                                     intensity=0.1, flip=True)
            np_tag_img2 = np_tag_img2.squeeze(0)
            np_cine_img2 = np_cine_img2.squeeze(0)

        mask1 = sitk.GetImageFromArray(np_tag_img2)
        mask1.SetSpacing(spacing1)
        mask1.SetOrigin(origin1)
        mask1.SetDirection(direction1)
        sitk.WriteImage(mask1, os.path.join(tgt_root3, tag_file0))
        tag_npz_file = os.path.join(tgt_root3, 'tag.npz')
        np.savez_compressed(tag_npz_file, np_tag_img2)

        mask2 = sitk.GetImageFromArray(np_cine_img2)
        mask2.SetSpacing(spacing2)
        mask2.SetOrigin(origin2)
        mask2.SetDirection(direction2)
        sitk.WriteImage(mask2, os.path.join(tgt_root3, cine_file0))
        cine_npz_file = os.path.join(tgt_root3, 'cine.npz')
        np.savez_compressed(cine_npz_file, np_cine_img2)