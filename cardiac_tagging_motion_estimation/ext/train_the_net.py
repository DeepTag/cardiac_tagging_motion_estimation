import sys
sys.path.append("..")

import torch
import torch.optim as optim
import os, time
from ME_nets.LagrangianMotionEstimationNet import Lagrangian_motion_estimate_net
from losses.train_loss import VM_diffeo_loss, NCC
import numpy as np
from data_set.load_data_for_cine_ME import add_np_data, get_np_data_as_groupids,load_np_datagroups, DataType, \
    load_Dataset

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_Cardiac_Tagging_ME_net(net, \
                                 np_data_root, \
                                 batch_size, \
                                 n_epochs, \
                                 learning_rate, \
                                 model_path, \
                                 kl_loss, \
                                 recon_loss, \
                                 smoothing_loss):
    net.train()
    net.cuda()
    net = net.float()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # training start time
    training_start_time = time.time()

    validation_data_group_ids = get_np_data_as_groupids(model_root=np_data_root, data_type=DataType.VALIDATION)
    validation_cines, validation_tags = load_np_datagroups(np_data_root, validation_data_group_ids,
                                                                           data_type=DataType.VALIDATION)
    val_dataset = load_Dataset(validation_cines, validation_tags)
    val_batch_size = 1
    val_set_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)

    train_loss_dict = []
    val_loss_dict = []

    for outer_epoch in range(n_epochs):
        # print training log
        print("epochs = ", outer_epoch)
        print("." * 50)

        nb_combined_groups = 1
        training_data_group_ids = get_np_data_as_groupids(model_root=np_data_root, data_type=DataType.TRAINING)
        if training_data_group_ids == []:
            print('there is no available training nets data.')

        training_data_combined_groupids_list = []
        combined_groupids = []
        for i, group_id in enumerate(training_data_group_ids):
            if i % nb_combined_groups == 0:
                combined_groupids = [group_id]
            else:
                combined_groupids.append(group_id)
            if i % nb_combined_groups == nb_combined_groups - 1 or i == len(training_data_group_ids) - 1:
                training_data_combined_groupids_list.append(combined_groupids)
        epoch_loss = 0
        group_num = 0

        for combined_groupids in training_data_combined_groupids_list:
            group_num += 1
            train_cines, train_tags = load_np_datagroups(np_data_root, combined_groupids,
                                                                    data_type=DataType.TRAINING)
            train_dataset = load_Dataset(train_cines, train_tags)

            training_set_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                              shuffle=True)
            train_n_batches = len(training_set_loader)
            # in each epoch do ...
            epoch_loss_0 = 0
            for i, data in enumerate(training_set_loader):
                cine, tag0 = data
                tag = tag0[:, 2:, ::]

                # wrap input data in a Variable object
                tag = tag.to(device)
                img = tag.cuda()
                img = img.float()

                x = img[:, 1:, ::]  # other frames except the 1st frame
                y = img[:, 0:24, ::]  # 1st frame also is the reference frame
                shape = x.shape  # batch_size, seq_length, height, width
                batch_size = shape[0]
                seq_length = shape[1]
                height = shape[2]
                width = shape[3]
                x = x.contiguous()
                x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
                y = y.contiguous()
                y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                # set the param gradients as zero
                optimizer.zero_grad()
                # forward pass, backward pass and optimization
                registered_cine1, registered_cine2, registered_cine1_lag, flow_param,  \
                deformation_matrix, deformation_matrix_neg, deformation_matrix_lag = net(y, x)

                train_smoothing_loss = smoothing_loss(deformation_matrix)
                train_smoothing_loss_neg = smoothing_loss(deformation_matrix_neg)
                train_smoothing_loss_lag = smoothing_loss(deformation_matrix_lag)

                a = 5
                b = 1
                training_loss = kl_loss(x, flow_param) + 0.5 * recon_loss(x, registered_cine1) + \
                                0.5 * recon_loss(y, registered_cine2) + 0.5 * recon_loss(x, registered_cine1_lag) + \
                                a * train_smoothing_loss + a * train_smoothing_loss_neg + b * train_smoothing_loss_lag

                training_loss.backward()
                optimizer.step()
                # statistic
                epoch_loss_0 += training_loss.item()


            epoch_loss_0 = epoch_loss_0 / train_n_batches
            print("training ME_epoch_loss_0 : {:.6f} ".format(epoch_loss_0))
            epoch_loss += epoch_loss_0

        epoch_loss = epoch_loss / group_num
        train_loss_dict.append(epoch_loss)
        np.savetxt(os.path.join(model_path, 'train_loss.txt'), train_loss_dict, fmt='%.6f')

        print("training loss: {:.6f} ".format(epoch_loss))

        if (outer_epoch) % 1 == 0:
            torch.save(net.state_dict(),
                       os.path.join(model_path, '{:d}_{:.4f}_model.pth'.format(outer_epoch, epoch_loss)))

        # when the epoch is over do a pass on the validation set
        total_val_loss = 0
        val_n_batches = len(val_set_loader)

        for i, data in enumerate(val_set_loader):
            cine, tag0 = data
            tag = tag0[:, 2:, ::]  # no grid frame
            val_batch_num_0 = cine.shape
            val_batch_num = val_batch_num_0[0]
            tag = tag.to(device)
            img = tag.cuda()
            img = img.float()

            x = img[:, 1:, ::]  # other frames except the 1st frame
            y = img[:, 0:24, ::]  # 1st frame also is the reference frame
            shape = x.shape  # batch_size, seq_length, height, width
            batch_size = shape[0]
            seq_length = shape[1]
            height = shape[2]
            width = shape[3]
            x = x.contiguous()
            x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

            y = y.contiguous()
            y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

            # forward pass
            val_registered_cine1, val_registered_cine2, val_registered_cine1_lag, \
            val_flow_param, val_deformation_matrix, val_deformation_matrix_neg, val_deformation_matrix_lag = net(y, x)

            val_smoothing_loss = smoothing_loss(val_deformation_matrix)
            val_smoothing_loss_neg = smoothing_loss(val_deformation_matrix_neg)
            val_smoothing_loss_lag = smoothing_loss(val_deformation_matrix_lag)


            a = 5
            b = 1
            val_loss = kl_loss(x, val_flow_param) + 0.5*recon_loss(x, val_registered_cine1) + \
                       0.5*recon_loss(y, val_registered_cine2) + 0.5*recon_loss(x, val_registered_cine1_lag) + \
                       a * val_smoothing_loss + a * val_smoothing_loss_neg + b * val_smoothing_loss_lag

            val_loss = val_loss / val_batch_num
            total_val_loss += val_loss.item()

        val_epoch_loss = total_val_loss / val_n_batches
        val_loss_dict.append(val_epoch_loss)
        np.savetxt(os.path.join(model_path, 'val_loss.txt'), val_loss_dict, fmt='%.6f')

        print("validation loss: {:.6f} ".format(val_epoch_loss))

    torch.save(net.state_dict(), os.path.join(model_path, 'end_model.pth'))
    print("Training finished! It took {:.2f}s".format(time.time() - training_start_time))


if __name__ == '__main__':
    # data loader
    train_dataset = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/Cardiac_ME_train_config.json'
    val_dataset = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/Cardiac_ME_val_config.json'
    np_data_root = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse/np_data'

    if not os.path.exists(np_data_root):
        os.mkdir(np_data_root)
        add_np_data(project_data_config_files=train_dataset, data_type='train', model_root=np_data_root)
        add_np_data(project_data_config_files=val_dataset, data_type='validation', model_root=np_data_root)

    training_model_path = '/home/DeepTag/models/cardiac_ME/DeepTag_Lag'

    if not os.path.exists(training_model_path):
        os.mkdir(training_model_path)
    n_epochs = 1500
    learning_rate = 5e-4
    batch_size = 1
    print("......HYPER-PARAMETERS 4 TRAINING......")
    print("batch size = ", batch_size)
    print("learning rate = ", learning_rate)
    print("." * 30)

    # proposed model
    vol_size = (192, 192)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]
    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    loss_class = VM_diffeo_loss(image_sigma=0.02, prior_lambda=10, flow_vol_shape=vol_size).cuda()
    my_ncc_loss = NCC()

    train_Cardiac_Tagging_ME_net(net=net,
                         np_data_root=np_data_root,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         learning_rate=learning_rate,
                         model_path=training_model_path,
                         kl_loss=loss_class.kl_loss,
                         recon_loss=my_ncc_loss,
                         smoothing_loss = loss_class.gradient_loss
                         )








