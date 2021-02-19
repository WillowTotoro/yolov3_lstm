import numpy as np
look_back = 3
batch_size = 4
num_sample = 96
npy_fold_path = 'lstm_npy/'


def load_image_feature(count):
    if count > num_sample:
        count = count % num_sample
    batch_input = np.array([])
    for batch in range(batch_size):
        x = np.array([])
        for i in range(look_back):
            file_num = count+look_back+batch-i
            file_name = 'bottle_'+str(file_num)+'.npy'
            npy_path = npy_fold_path+file_name
            x = np.load(npy_path)
            np.reshape(x, (1, 1797))
        sample = np.concatenate(x, axis=0)
    batch_input = np.concatenate(sample, axis=2)
    return batch_input


def load_gt_batch(count):
