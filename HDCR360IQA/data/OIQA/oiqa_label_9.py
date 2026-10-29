import os
import torch
import numpy as np
import cv2
import math

from utils_tools.erp2rec import ERP2REC


np.random.seed(2)


def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def generate_lat_prob():
    x = np.linspace(-90, 90, 181) / 90
    y = {}
    val = []

    for i in range(x.shape[0]):
        val.append(normal_distribution(x[i], 0, 0.2))

    val = np.array(val)
    val = softmax(val)

    idx = -90
    for i in range(val.shape[0]):
        y[idx] = val[i]
        idx += 1

    return y


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class OIQA(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, viewport_size,
                 viewport_nums, fov, start_points):
        super(OIQA, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.viewport_size = viewport_size
        self.viewport_nums = 9
        self.fov = [110, 110]
        self.domain_transform = ERP2REC()
        self.start_points = start_points

        # define the latitude weights
        self.lat_weights = generate_lat_prob()


        self.equator_coords = [
            (0, -180), (0, -135), (0, -90), (0, -45), (0, 0),
            (0, 45), (0, 90), (0, 135), (0, 180)
        ]

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    if os.path.exists(os.path.join(self.dis_path, dis) + ".jpg"):
                        dis_files_data.append(dis + ".jpg")
                    elif os.path.exists(os.path.join(self.dis_path, dis) + ".png"):
                        dis_files_data.append(dis + ".png")
                    else:
                        print("There is no data {}".format(dis))
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data, range_val = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.range_val = range_val

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)


        d1 = self.select_equator_viewports(d_img)
        for i in range(d1.shape[0]):
            for j in range(d1.shape[1]):
                d1[i][j] = cv2.cvtColor(d1[i][j], cv2.COLOR_BGR2RGB)
                d1[i][j] = np.array(d1[i][j]).astype('float32') / 255
        d1 = np.transpose(d1, (0, 1, 4, 2, 3))

        score = self.data_dict['score_list'][idx]
        d1 = np.array(d1)
        sample = {
            'd_img_org': d1,
            'score': score,
            'name': d_img_name,
            'range_val': self.range_val,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range, range
    
    def cal_entropy(self, sig):
        """ Calculate the entropy
        Args:
            sig (ndarray): the one-dim array flatten from a window
        """
        len = sig.size
        sig_set = list(set(sig))
        p_list = [np.size(sig[sig == i]) / len for i in sig_set]
        entropy = np.sum([p * np.log2(1.0 / p) for p in p_list])
        return entropy
    
    def get_img_entropy(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        sum_entropy = self.cal_entropy(gray_img.flatten())
        return sum_entropy
    
    def cal_next_patch_coordinate(self, viewport, cur_coordinate=(0, 0)):
        pass
        
    def dfs_get_viewport(self, img, cur_coordinate=(0, 0)):
        pass


    def select_equator_viewports(self, img):
        viewports_list = []
        for (r, c) in self.equator_coords:
            viewport = self.domain_transform.toREC(
                frame=img,
                center_point=np.array([c, r]),
                FOV=self.fov,
                width=self.viewport_size[0],
                height=self.viewport_size[1]
            )
            viewports_list.append(viewport)
        seq_list = [np.array(viewports_list, dtype=np.float32)]
        return np.array(seq_list)
        
    def select_viewports(self, img):

        return self.select_equator_viewports(img)
    
    def modify_c(self, _c):
        if _c > 180:
            _c = -180 + (_c - 180)
        elif _c < -180:
            _c = 180 - (-180 - _c)
        else:
            pass
        return _c