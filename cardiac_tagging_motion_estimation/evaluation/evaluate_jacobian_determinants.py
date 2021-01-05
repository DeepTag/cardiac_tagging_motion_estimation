import os
import SimpleITK as sitk
import numpy as np
import json
import matplotlib.pyplot as plt

def generate_grid2(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid0 = np.array(np.meshgrid(z, y, x))
    grid1 = np.rollaxis(grid0, 0, 4)
    grid2 = np.swapaxes(grid1,0,2)
    grid = np.swapaxes(grid2,1,2)
    return grid

def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    grid0 = np.array(np.meshgrid(y, x))
    grid = np.rollaxis(grid0, 0, 3)
    return grid

def JacobianDet0(x, y):
    imgshape = x.shape
    grid = generate_grid((192, 192))
    x = x + grid[:,:,0]
    y = y + grid[:,:,1]
    dxy = x[:, 1:, :-1] - x[:, :-1, :-1]
    dxx = x[:, :-1, 1:] - x[:, :-1, :-1]
    dyy = y[:, 1:, :-1] - y[:, :-1, :-1]
    dyx = y[:, :-1, 1:] - y[:, :-1, :-1]

    Jdet = dxx * dyy - dxy * dyx
    return Jdet

def JacobianDet(x, y):
    imgshape = x.shape
    grid = generate_grid((192, 192))
    x = x[1:,::] - x[:-1, ::] # eu
    y = y[1:,::] - y[:-1, ::] # eu
    x = x + grid[:,:,0]
    y = y + grid[:,:,1]
    dxy = x[:, 1:, :-1] - x[:, :-1, :-1]
    dxx = x[:, :-1, 1:] - x[:, :-1, :-1]
    dyy = y[:, 1:, :-1] - y[:, :-1, :-1]
    dyx = y[:, :-1, 1:] - y[:, :-1, :-1]

    Jdet = dxx * dyy - dxy * dyx
    return Jdet
#
# imgshape = imgshape = (160, 192)
# grid = generate_grid(imgshape)
# x = grid[:, :,1]
# plt.imshow(x)

flow_root = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/models_testing/'

dst_path = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/models_testing/results/JD/'

if not os.path.exists(dst_path): os.makedirs(dst_path)

exist_patient={}

exist_patient['Jacobian'] = {}

# np =3
for kk in range(12, 13):
    exist_num = 1
    JD_vec = []
    model = 'M'+str(kk)
    exist_patient['Jacobian'][model] = {}
    print(model)
    for subroot, dirs, files in os.walk(os.path.join(flow_root, model)):
        if len(files) < 3: continue
        root_vec = subroot.split(os.path.sep)

        patient_num = root_vec[-3]+'_'+root_vec[-2]+'_'+root_vec[-1]

        for file in files:
            if file.endswith('.nii.gz') and 'deformation_matrix_x_img' in file and 'neg' not in file and 'eu' not in file:
                deformation_matrix_x_img_file = file
            if file.endswith('.nii.gz') and 'deformation_matrix_y_img' in file and 'neg' not in file and 'eu' not in file:
                deformation_matrix_y_img_file = file
                    # break
        deformation_matrix_x_img = sitk.ReadImage(os.path.join(subroot, deformation_matrix_x_img_file))
        deformation_matrix_y_img = sitk.ReadImage(os.path.join(subroot, deformation_matrix_y_img_file))
        deformation_matrix_x = sitk.GetArrayFromImage(deformation_matrix_x_img)
        deformation_matrix_y = sitk.GetArrayFromImage(deformation_matrix_y_img)

        s = deformation_matrix_x.shape
        JD = JacobianDet0(deformation_matrix_y, deformation_matrix_x)
        # JD = JacobianDet(-deformation_matrix_x, -deformation_matrix_y) # OF-TV
        # JD = JacobianDet(-deformation_matrix_y, -deformation_matrix_x) # HARP
        JD[-1,::] = deformation_matrix_x[-1,:-1, :-1]
        # JD = np.concatenate((JD, np.expand_dims(deformation_matrix_x[0,:-1, :-1],axis=0)), axis=0)

        test = JD[:-1, ::]
        p = np.where(JD[:-1,::] <= 0)

        n= len(p[0])/(s[0]-1)
        JD_vec.append(n)

        spacing1 = deformation_matrix_x_img.GetSpacing()
        origin1 = deformation_matrix_x_img.GetOrigin()
        direction1 = deformation_matrix_x_img.GetDirection()

        JD_img = sitk.GetImageFromArray(JD)
        JD_img.SetSpacing(spacing1)
        JD_img.SetOrigin(origin1)
        JD_img.SetDirection(direction1)
        # sitk.WriteImage(JD_img, os.path.join(subroot, 'JD_eu.nii.gz'))

        exist_num += 1
        print(exist_num-1)

    exist_patient['Jacobian'][model]['Jacobian']= str([np.mean(JD_vec), np.std(JD_vec, ddof=1)])


with open(os.path.join(dst_path, 'Tagging_motion_tracking'+ '_Jacobian_12_Lag.json'), 'w', encoding='utf-8') as f_json:
    json.dump(exist_patient, f_json, indent=4)

