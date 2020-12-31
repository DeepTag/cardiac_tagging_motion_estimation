import sys
sys.path.append("..")

import torch
import os, json
from ME_nets.LagrangianMotionEstimationNet import Lagrangian_motion_estimate_net
from data_set.load_data_for_cine_ME import add_np_data, get_np_data_as_groupids,load_np_datagroups, DataType, \
    load_Dataset
import SimpleITK as sitk


# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dec_weights(model, weights):
    print('Resuming net weights from {} ...'.format(weights))
    w_dict = torch.load(weights)
    model.load_state_dict(w_dict, strict=True)
    return model

def test_Cardiac_Tagging_ME_net(net, \
                                np_data_root, \
                                val_dataset_files, \
                                model_path, \
                                dst_root, \
                                case = 'proposed'):
    with open(val_dataset_files, 'r', encoding='utf-8') as f_json:
        data_config = json.load(f_json)
    if data_config is not None and data_config.__class__ is dict:
        grouped_data_sets = data_config.get('test')
        if grouped_data_sets.__class__ is not dict: print('invalid train_config.')

    # check grouped_data_sets
    if grouped_data_sets.__class__ is not dict: print('invalid data config file.')

    group_names = grouped_data_sets.keys()
    val_data_list = []
    for group_name in group_names:
        print('working on %s', group_name)
        filesListDict = grouped_data_sets.get(group_name)
        if filesListDict.__class__ is not dict: continue

        for sample in filesListDict.keys():
            each_trainingSets = filesListDict.get(sample)
            # list images_data_niix in each dataset
            cine_npz = each_trainingSets.get('cine')
            val_data_list.append(cine_npz)

    validation_data_group_ids = get_np_data_as_groupids(model_root=np_data_root, data_type=DataType.VALIDATION)
    validation_cines, validation_tags  = load_np_datagroups(np_data_root, validation_data_group_ids,
                                                                           data_type=DataType.VALIDATION)
    val_dataset = load_Dataset(validation_cines, validation_tags)
    val_batch_size = 1
    test_set_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
    if not os.path.exists(dst_root): os.makedirs(dst_root)
    if case == 'proposed':
        model = '/pro_new_model.pth'
    else:
        model = '/baseline_model.pth'
    ME_model = load_dec_weights(net, model_path + model)
    ME_model = ME_model.to(device)
    ME_model.eval()

    for i, data in enumerate(test_set_loader):
        # cine0, tag = data
        untagged_cine, tagged_cine = data

        # wrap input data in a Variable object
        cine0 = tagged_cine.to(device)

        cine1 = tagged_cine[:, 2:, ::]  # no grid frame

        # wrap input data in a Variable object
        img = cine1.cuda()
        img = img.float()

        x = img[:, 1:, ::]  # other frames except the 1st frame
        y = img[:, 0:24, ::]  # 1st frame also is the reference frame
        shape = x.shape  # batch_size, seq_length, height, width
        seq_length = shape[1]
        height = shape[2]
        width = shape[3]
        x = x.contiguous()
        x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        # y = y.repeat(1, seq_length, 1, 1)  # repeat the ES frame to match other frames contained in a Cine
        y = y.contiguous()
        y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        z = cine0[:, 0:1, ::]  # Tag grid frame also is the reference frame
        z = z.repeat(1, seq_length, 1, 1)  # repeat the ES frame to match other frames contained in a Cine
        z = z.contiguous()
        z = z.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        # forward pass
        with torch.no_grad():
            val_resgistered_cine1, val_resgistered_cine2, val_resgistered_cine_lag, val_flow_param, \
            val_deformation_matrix, val_deformation_matrix_neg, val_deformation_matrix_lag = net(y, x)

        y = img[:, -1, ::]  # the last frame

        val_deformation_matrix_lag0 = torch.cat((val_deformation_matrix_lag[:,0,::], y), dim=0)
        val_deformation_matrix_lag0 = val_deformation_matrix_lag0.cuda()
        val_deformation_matrix_lag0 = val_deformation_matrix_lag0.cpu().detach().numpy()
        # val_deformation_matrix_lag0 = val_deformation_matrix_lag0.squeeze(0)

        val_deformation_matrix_lag1 = torch.cat((val_deformation_matrix_lag[:, 1, ::], y), dim=0)
        val_deformation_matrix_lag1 = val_deformation_matrix_lag1.cuda()
        val_deformation_matrix_lag1 = val_deformation_matrix_lag1.cpu().detach().numpy()
        # val_deformation_matrix_lag1 = val_deformation_matrix_lag1.squeeze(0)


        file_path = val_data_list[i][0]
        root_vec = file_path.split(os.path.sep)
        tgt_root1 = os.path.join(dst_root, root_vec[-5])
        if not os.path.exists(tgt_root1): os.mkdir(tgt_root1)
        tgt_root2 = os.path.join(tgt_root1, root_vec[-3])
        if not os.path.exists(tgt_root2): os.mkdir(tgt_root2)
        tgt_root3 = os.path.join(tgt_root2, root_vec[-2])
        if not os.path.exists(tgt_root3): os.mkdir(tgt_root3)


        val_img_file_root = file_path.replace('cine.npz', '')

        val_cine_file = ''
        for subroot, dirs, files in os.walk(val_img_file_root):
            if len(files) < 4: continue
            for file in files:
                if file.endswith('.nii.gz') and 'CINE' in file and 'TAG' in file:
                    val_cine_file = file
                    break

        cine_image = sitk.ReadImage(os.path.join(val_img_file_root, val_cine_file))
        spacing1 = cine_image.GetSpacing()
        origin1 = cine_image.GetOrigin()
        direction1 = cine_image.GetDirection()

        img_matrix = sitk.GetArrayFromImage(cine_image)
        cine_img = sitk.GetImageFromArray(img_matrix[2:,::])
        cine_img.SetSpacing(spacing1)
        cine_img.SetDirection(direction1)
        cine_img.SetOrigin(origin1)
        sitk.WriteImage(cine_img, os.path.join(tgt_root3, val_cine_file))


        val_deformation_matrix_lag_img0 = sitk.GetImageFromArray(val_deformation_matrix_lag0)
        val_deformation_matrix_lag_img0.SetSpacing(spacing1)
        val_deformation_matrix_lag_img0.SetOrigin(origin1)
        val_deformation_matrix_lag_img0.SetDirection(direction1)
        sitk.WriteImage(val_deformation_matrix_lag_img0, os.path.join(tgt_root3, 'deformation_matrix_x.nii.gz'))

        val_deformation_matrix_lag_img1 = sitk.GetImageFromArray(val_deformation_matrix_lag1)
        val_deformation_matrix_lag_img1.SetSpacing(spacing1)
        val_deformation_matrix_lag_img1.SetOrigin(origin1)
        val_deformation_matrix_lag_img1.SetDirection(direction1)
        sitk.WriteImage(val_deformation_matrix_lag_img1, os.path.join(tgt_root3, 'deformation_matrix_y.nii.gz'))

        print('finish: ' + str(i))




if __name__ == '__main__':
    # data loader
    test_dataset = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/Cardiac_ME_test_config.json'
    np_data_root = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/np_data'

    if not os.path.exists(np_data_root):
        os.mkdir(np_data_root)
        add_np_data(project_data_config_files=test_dataset, data_type='validation', model_root=np_data_root)

    # proposed model
    vol_size = (192, 192)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]
    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    test_model_path = '/home/DeepTag/models/cardiac_ME/DeepTag_Lag'
    dst_root = '/home/DeepTag/data/Motion_tracking_20200610/tagging_analysis/test_results/'
    if not os.path.exists(dst_root): os.mkdir(dst_root)
    test_Cardiac_Tagging_ME_net(net=net,
                             np_data_root=np_data_root,
                             val_dataset_files=test_dataset,
                             model_path= test_model_path,
                             dst_root=dst_root,
                             case = 'proposed')










