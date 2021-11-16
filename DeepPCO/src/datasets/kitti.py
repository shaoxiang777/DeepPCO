import os
import csv
import cv2
import glob
import torch
import random
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.utils import exp_SO3, log_SO3
from configs.cfg import args


class KITTI(data.Dataset):
    def __init__(self, split):
        # split: train or test split
        self.split = split
        self._lst = KITTI.load_gt(args["kitti_path"], split)
        self.transforms = T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        first_img = cv2.imread(self._lst[index]["img_files"][0], 0)
        second_img = cv2.imread(self._lst[index]["img_files"][1], 0)

        first_img = np.stack((first_img, first_img, first_img),
                             axis=-1)
        second_img = np.stack((second_img, second_img, second_img),
                              axis=-1)

        first_img = self.transforms(first_img)
        second_img = self.transforms(second_img)

        gt_t = torch.FloatTensor(self._lst[index]["pose"][:3])
        gt_r = torch.FloatTensor(self._lst[index]["pose"][3:])

        # if args["hflip"] and self.split == "train":
        #     hflip = random.random() < 0.5
        #     if hflip:
        #         H = torch.diag(torch.FloatTensor([1, -1, 1]))
        #         first_img = T.functional.hflip(first_img)
        #         second_img = T.functional.hflip(second_img)
        #         C = torch.matmul(torch.matmul(H, exp_SO3(gt_r)), H.transpose(0, 1))
        #         gt_t = torch.matmul(H, gt_t)
        #         gt_r = log_SO3(C)

        image = torch.cat((first_img, second_img), 0)

        return image, gt_t, gt_r

    def __len__(self):
        return len(self._lst)

    @staticmethod
    def get_files(path, split):
        """
        :param path: "your/path/kitti_dataset"
        :return: list of image file paths and label file paths
        """
        if split == "train":
            sequences = args['train_split']
        else:
            sequences = args['test_split']
        files = []
        for sequence_name in sequences:
            tmp = []

            sequence_path = os.path.join(path, sequence_name)
            img_files = sorted(glob.glob(os.path.join(sequence_path,
                                                      "panoramic/*.png")))
            for idx, img_file in enumerate(img_files):
                data_dict = {"img_file": img_file}

                with open(os.path.join(sequence_path, sequence_name + ".txt"),
                          mode='r') as f:
                    lines = f.readlines()
                    pose = lines[idx].rstrip().split(' ')
                    data_dict["pose"] = pose
                    data_dict["idx"] = [idx, int(sequence_name)]

                tmp.append(data_dict)

            if split == "train":
                paired_files = KITTI.pairing(tmp)
                if args["reverse"]:
                    paired_files_reversed = KITTI.pairing(list(reversed(tmp)))
                    paired_files = paired_files + paired_files_reversed
            else:
                paired_files = KITTI.pairing(tmp)

            files += paired_files

        return files

    @staticmethod
    def load_gt(path, split):
        ret = []
        for paired_files in KITTI.get_files(path, split):
            ret.append(
                {
                    "img_files": paired_files['img_files'],
                    "pose": paired_files['pose'],
                }
            )

        return ret

    @staticmethod
    def pairing(files):
        paired_files = []
        # discard last several frames
        for idx in range(len(files) - 1):
            # random.seed(13)
            next_idx1 = 1 + idx

            dict1 = {'img_files': (files[idx]["img_file"],
                                   files[next_idx1]["img_file"])}

            p1 = list(map(float, files[idx]["pose"]))
            p2 = list(map(float, files[next_idx1]["pose"]))

            SE3_1 = np.array([
                [p1[0], p1[1], p1[2], p1[3]],
                [p1[4], p1[5], p1[6], p1[7]],
                [p1[8], p1[9], p1[10], p1[11]],
                [0, 0, 0, 1]])

            SE3_2 = np.array([
                [p2[0], p2[1], p2[2], p2[3]],
                [p2[4], p2[5], p2[6], p2[7]],
                [p2[8], p2[9], p2[10], p2[11]],
                [0, 0, 0, 1]])

            # diff in previous frame coordinate instead of in fixed coordinate
            SE3_diff = np.linalg.inv(SE3_1).dot(SE3_2)

            # the translation from T
            t_diff = SE3_diff[0:3, 3]

            r_diff = SE3_diff[0:3, 0:3]

            r_diff = R.from_matrix(r_diff)

            # r_diff = r_diff.as_euler('zyx')
            r_diff = r_diff.as_rotvec()

            dict1['pose'] = [*t_diff, *r_diff]

            paired_files.append(dict1)

        return paired_files
