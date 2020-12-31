import numpy as np
import json, os
from torch.utils.data import Dataset
from enum import Enum
from glob import glob

from data_set.logger import  get_logger
logger = get_logger()

class DataType(Enum):
    TRAINING = 0
    VALIDATION = 1



# # --------------------------------------------------------------------------------------------
#
def load_np_array_from_npz(npz_file):
    npz = np.load(npz_file)
    np_arrs = []
    for np_file in npz.keys(): np_arrs.append(npz[np_file])
    A = np_arrs[0]
    B = A.T
    return [B]

# # --------------------------------------------------------------------------------------------
#
def load_np_mask_array_from_npz(npz_file):
    npz = np.load(npz_file)
    np_arrs = []
    for np_file in npz.keys(): np_arrs.append(npz[np_file])
    A = np_arrs[0]
    t,y,x = A.shape
    extra_t = t - 27
    B = np.zeros((27, y, x))
    if extra_t > 0:
        B[0:22,::] = A[0:22,::]
        B[-5:, ::] = A[-5:,::]
    else:
        B = A
    return [B]

# # --------------------------------------------------------------------------------------------
#
def load_np_mask_array_from_npz1(npz_file):
    npz = np.load(npz_file)
    np_arrs = []
    for np_file in npz.keys(): np_arrs.append(npz[np_file])
    A = np_arrs[0]
    B = A
    return [B]
# ----------------------------------------------------------------------------------
#
def get_np_data_prefix(data_type=DataType.TRAINING):
    """
    :return:
    """
    if data_type == DataType.TRAINING:
        images_np_prefix = 'training_cine_'
        masks_np_prefix = 'training_tag_'
    elif data_type == DataType.VALIDATION:
        images_np_prefix = 'validation_cine_'
        masks_np_prefix = 'validation_tag_'
    return images_np_prefix, masks_np_prefix

# ----------------------------------------------------------------------------------
#
def get_np_data_as_groupids(model_root, data_type=DataType.TRAINING):
    """
    :param model_root:
    :param data_type:
    :return:
    """
    groupids = []
    # read info from stored files
    images_prefix, masks_np_prefix = get_np_data_prefix(data_type=data_type)
    images_npzs = glob(os.path.join(model_root, images_prefix + '*.npz'))
    masks_npzs = glob(os.path.join(model_root, masks_np_prefix + '*.npz'))
    if len(images_npzs) != len(masks_npzs): return groupids
    if len(images_npzs) < 1 or len(masks_npzs) < 1: return groupids

    # get the group ids
    images_groupids = []
    masks_groupids = []

    for images_npz in images_npzs: images_groupids.append(
        int(os.path.basename(images_npz[:-4]).replace(images_prefix, '')))
    for masks_npz in masks_npzs: masks_groupids.append(
        int(os.path.basename(masks_npz[:-4]).replace(masks_np_prefix, '')))
    if images_groupids != masks_groupids: return images_groupids

    return images_groupids

# ----------------------------------------------------------------------------------
#
def get_np_data_filename( i_subgroup, data_type=DataType.TRAINING):
    """
    :param i_subgroup:
    :return: images_00000, targets_00000
    """
    images_prefix,  masks_np_prefix = get_np_data_prefix(data_type=data_type)
    images_np_name = images_prefix + '{:05d}.npz'.format(i_subgroup)
    masks_np_name = masks_np_prefix + '{:05d}.npz'.format(i_subgroup)

    return images_np_name, masks_np_name

# ----------------------------------------------------------------------------------
#
def save_np_data(model_root, np_images, np_masks, data_type=DataType.TRAINING):
    """
    :param np_images:
    :param group_name:
    :param files_list_dict:
    :return:
    """
    # groups starting from 0 then appending to previous groupids
    # counting groups
    group_ids = get_np_data_as_groupids(model_root, data_type=data_type)
    if group_ids == []: group_ids = [0]
    group_ids = sorted(group_ids)
    group_ids_expand = group_ids + [group_ids[-1] + 1]
    group_ids_left = [x for x in group_ids_expand if x not in group_ids]
    images_np, masks_np = get_np_data_filename(group_ids_left[0], data_type=data_type)
    images_np_file = os.path.join(model_root, images_np)
    masks_np_file = os.path.join(model_root, masks_np)
    np.savez_compressed(images_np_file, np_images)
    np.savez_compressed(masks_np_file, np_masks)
    return
# # --------------------------------------------------------------------------------------------
#

def add_np_data(project_data_config_files, data_type, model_root):
    with open(project_data_config_files, 'r', encoding='utf-8') as f_json:
        data_config = json.load(f_json)
    if data_config is not None and data_config.__class__ is dict:
        grouped_data_sets = data_config.get(data_type)
        if grouped_data_sets.__class__ is not dict: logger.info('invalid train_config.')

    # check grouped_data_sets
    if grouped_data_sets.__class__ is not dict: logger.info('invalid data config file.')

    group_names = grouped_data_sets.keys()
    for group_name in group_names:
        print('working on %s', group_name)
        filesListDict = grouped_data_sets.get(group_name)
        if filesListDict.__class__ is not dict: continue
        logger.info('adding %s into nets data from npz data input...', group_name)
        grouped_np_cine_npz = None # shape = (samples, seq_len, y, x)
        grouped_np_tag_npz = None  # shape = (samples, seq_len, y, x)

        # for sample in tqdm(filesListDict.keys()):
        for sample in filesListDict.keys():
            each_trainingSets = filesListDict.get(sample)

            # list images_data_niix in each dataset
            cine_npz = each_trainingSets.get('cine')
            if cine_npz is None: continue
            cine_npz_List = []
            cine_npz_List.append((load_np_mask_array_from_npz(cine_npz[0]))[0])

            tag_npz = each_trainingSets.get('tag')
            if tag_npz is None: continue
            tag_npz_List = []
            tag_npz_List.append((load_np_mask_array_from_npz(tag_npz[0]))[0])


            # concatenate the np_nets as samples
            if grouped_np_cine_npz is None:
                grouped_np_cine_npz = cine_npz_List
            # sample axis is default to 0
            else:
                grouped_np_cine_npz = np.concatenate([grouped_np_cine_npz, cine_npz_List], axis=0)

            if grouped_np_tag_npz is None:
                grouped_np_tag_npz = tag_npz_List
            # sample axis is default to 0
            else:
                grouped_np_tag_npz = np.concatenate([grouped_np_tag_npz, tag_npz_List], axis=0)

        if data_type == 'train':
            data_type_temp = DataType.TRAINING
        else:
            data_type_temp = DataType.VALIDATION
        save_np_data(model_root, np_images=grouped_np_cine_npz , np_masks=grouped_np_tag_npz, data_type=data_type_temp)

    # return  grouped_np_pc_npz, grouped_np_lv_lx_npz

# # --------------------------------------------------------------------------------------------
#
def load_np_datagroups(model_root, groupids, data_type=DataType.VALIDATION):
    """
    :param groupids:
    :param data_type:
    :return:
    """
    np_images_list = []
    np_masks_list = []
    for groupid in groupids:
        images_npz_filename, masks_npz_filename = get_np_data_filename(groupid, data_type=data_type)
        images_npz_file = os.path.join(model_root, images_npz_filename)
        masks_npz_file = os.path.join(model_root, masks_npz_filename)

        if not os.path.exists(images_npz_file) or not os.path.exists(masks_npz_file): continue
        for np_images, np_masks in zip(load_np_mask_array_from_npz1(images_npz_file),
                                         load_np_mask_array_from_npz1(masks_npz_file)):
            np_images_list.append(np_images.astype(np.float32))
            np_masks_list.append(np_masks.astype(np.float32))

    if np_images_list == []  or np_masks_list == []: return None, None
    np_images = np.concatenate(np_images_list, axis=0)
    np_masks = np.concatenate(np_masks_list, axis=0)

    return np_images, np_masks

# # --------------------------------------------------------------------------------------------
#
def load_training_validation_npz_data(model_root, nb_combined_groups=10, data_type=DataType.TRAINING):
    """
    :param model_root:
    :param data_type:
    :return:
    """
    validation_data_group_ids = get_np_data_as_groupids(data_type=DataType.VALIDATION)
    validation_vols, validation_masks, validation_pcs = load_np_datagroups(validation_data_group_ids, data_type=DataType.VALIDATION)

    training_data_group_ids = get_np_data_as_groupids(model_root=model_root, data_type=data_type)
    if training_data_group_ids == []:
        logger.warning('there is no available training nets data.')
        return False

    # regroup the data
    training_data_combined_groupids_list = []
    combined_groupids = []
    for i, group_id in enumerate(training_data_group_ids):
        if i % nb_combined_groups == 0:
            combined_groupids = [group_id]
        else:
            combined_groupids.append(group_id)
        if i % nb_combined_groups == nb_combined_groups - 1 or i == len(training_data_group_ids) - 1:
            training_data_combined_groupids_list.append(combined_groupids)

    for combined_groupids in training_data_combined_groupids_list:
        train_vols, train_targets = load_np_datagroups(combined_groupids,data_type=DataType.TRAINING)
# ------------------------------------------------------------------------------
#
class load_Dataset(Dataset):
    """
    load training/validation/test data set to the torch Dataset
    """
    def __init__(self, cines, tags):
        self.cine = cines
        self.tag = tags

    def __getitem__(self, item):
        return self.cine[item], self.tag[item]

    def __len__(self):
        return len(self.cine)


if __name__ == '__main__':

    project_data_config_files ='/home/DeepTag/data/Motion_tracking_20200610/val1/Cardiac_ME_val_config.json'
    data_type = 'validation'
    model_root = '/home/DeepTag/data/Motion_tracking_20200610/val1/'
    grouped_cine_tag_npz = add_np_data(project_data_config_files, data_type, model_root)

    validation_data_group_ids = get_np_data_as_groupids(model_root=model_root, data_type=DataType.VALIDATION)
    validation_cine, validation_tag = load_np_datagroups(model_root, validation_data_group_ids,
                                                                           data_type=DataType.VALIDATION)
    tag = validation_tag
    a01 = tag[0, ::]
    a02 = tag[1, ::]
    #
    cine = validation_cine
    a21 = cine[0, ::]
    a22 = cine[20, ::]
