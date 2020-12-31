import numpy as np
import os, json


src_lx_img = '/home/DeepTag/data/auged/'
grouped_subfolder = '/home/DeepTag/data/Motion_tracking_20200610/val1_reverse'

train_groups = ['005',
                '007', '008', '009', '010',
                '012', '013',  '014', '015',
                '017', '018', '019', '020', '022', '023']

val_groups = [ '006', '011', '016']

test_groups = [ '001', '002', '003',  '004', '021']


test_config = {}
test_config['test'] = {}

lx_img_size=0
i_group = 0
group_size = 500
for test_group in test_groups:
    dcm_root = os.path.join(src_lx_img, 'NYU_CON'+test_group)
    for patient_root, subdirs, files in os.walk(dcm_root):
        if len(files) <= 0: continue
        patient_root_vec = patient_root.split(os.path.sep)
        aug_num = patient_root_vec[-3]
        if aug_num != 'aug_0': continue
        lx_img_size += 1
        if i_group % group_size == 0:
            group_id = 1 + int(i_group / group_size)
            group_name = 'Cardiac_ME_' + str(group_id)
            test_config['test'][group_name] = {}
        sample_name = test_group+ '_' + patient_root_vec[-4] + '_' + patient_root_vec[-3] \
                      + '_' +patient_root_vec[-2]+ '_Aug_' + patient_root_vec[-1]
        test_config['test'][group_name][sample_name] ={}

        tag_file =  [os.path.join(patient_root, 'tag.npz')]
        cine_file = [os.path.join(patient_root, 'cine.npz')]

        test_config['test'][group_name][sample_name]['tag'] = tag_file
        test_config['test'][group_name][sample_name]['cine'] = cine_file
        i_group += 1

print('test sample size:')
print(lx_img_size)
target_root = grouped_subfolder
with open(os.path.join(target_root, 'Cardiac_ME_' + 'test_config.json'), 'w', encoding='utf-8') as f_json:
        json.dump(test_config, f_json, indent=4)

validation_config = {}
validation_config['validation'] = {}

lx_img_size=0
i_group = 0
group_size = 500
for val_group in val_groups:
    dcm_root = os.path.join(src_lx_img, 'NYU_CON'+val_group)
    for patient_root, subdirs, files in os.walk(dcm_root):
        if len(files) <= 0: continue
        patient_root_vec = patient_root.split(os.path.sep)
        aug_num = patient_root_vec[-3]
        if aug_num != 'aug_0': continue
        lx_img_size += 1
        if i_group % group_size == 0:
            group_id = 1 + int(i_group / group_size)
            group_name = 'Cardiac_ME_' + str(group_id)
            validation_config['validation'][group_name] = {}
        sample_name = val_group+ '_' + patient_root_vec[-4] + '_' + patient_root_vec[-3] \
                      + '_' +patient_root_vec[-2]+ '_Aug_' + patient_root_vec[-1]
        validation_config['validation'][group_name][sample_name] ={}

        tag_file =  [os.path.join(patient_root, 'tag.npz')]
        cine_file = [os.path.join(patient_root, 'cine.npz')]

        validation_config['validation'][group_name][sample_name]['tag'] = tag_file
        validation_config['validation'][group_name][sample_name]['cine'] = cine_file
        i_group += 1

print('val sample size:')
print(lx_img_size)
target_root = grouped_subfolder
with open(os.path.join(target_root, 'Cardiac_ME_' + 'val_config.json'), 'w', encoding='utf-8') as f_json:
        json.dump(validation_config, f_json, indent=4)


train_config = {}
train_config['train'] = {}
i_group = 0
group_size = 500
lx_img_size = -1
for group in train_groups:
    dcm_root = os.path.join(src_lx_img, 'NYU_CON'+group)
    for patient_root, subdirs, files in os.walk(dcm_root):
        if len(files) <= 0: continue

        patient_root_vec = patient_root.split(os.path.sep)
        lx_img_size += 1
        train_config['train']['tag_cine_' + str(lx_img_size)] = {}
        sample_name = group + '_' + patient_root_vec[-4] + '_' + patient_root_vec[-3] \
                      + '_' + patient_root_vec[-2] + '_Aug_' + patient_root_vec[-1]
        train_config['train']['tag_cine_' + str(lx_img_size)]['sample_name'] = sample_name
        train_config['train']['tag_cine_' + str(lx_img_size)][sample_name] = {}
        tag_file = [os.path.join(patient_root, 'tag.npz')]
        cine_file = [os.path.join(patient_root, 'cine.npz')]
        train_config['train']['tag_cine_' + str(lx_img_size)][sample_name]['tag'] = tag_file
        train_config['train']['tag_cine_' + str(lx_img_size)][sample_name]['cine'] = cine_file

# randomize the training samples
print('training sample size:')
print(lx_img_size+1)
arr = np.arange(lx_img_size+1)
np.random.shuffle(arr)

train_group_config = {}
train_group_config['train'] = {}
for i in arr:
    # group name
    if i_group % group_size == 0:
        group_id = 1 + int(i_group / group_size)
        group_name = 'Cardiac_ME_' + str(group_id)
        train_group_config['train'][group_name] = {}
    index = int(i_group % group_size + 1)
    sample_name = train_config['train']['tag_cine_' + str(i)]['sample_name']

    my_sample_lesion_name = str(index) + '_tag_cine_' + sample_name
    train_group_config['train'][group_name][my_sample_lesion_name] = {}
    train_group_config['train'][group_name][my_sample_lesion_name]['tag'] = \
        train_config['train']['tag_cine_' + str(i)][sample_name]['tag']
    train_group_config['train'][group_name][my_sample_lesion_name]['cine'] = \
        train_config['train']['tag_cine_' + str(i)][sample_name]['cine']
    i_group += 1

target_root = grouped_subfolder
with open(os.path.join(target_root, 'Cardiac_ME_' + 'train_config.json'), 'w', encoding='utf-8') as f_json:
        json.dump(train_group_config, f_json, indent=4)

