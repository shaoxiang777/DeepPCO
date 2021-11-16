import os
import csv
import cv2
import glob
import torch
import random
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np
import quaternion
from configs.cfg import args
from utils.utils import exp_SO3, log_SO3
from scipy.spatial.transform import Rotation as R


class EuRoC(data.Dataset):
    def __init__(self, split):
        # split: train or test split
        self.split = split
        self._lst = EuRoC.load_gt(args["euroc_path"], self.split)
        self.transforms = T.Compose(
            T.ToTensor())

    def __getitem__(self, index):
        first_img = cv2.imread(self._lst[index]["img_files"][0], -1)
        second_img = cv2.imread(self._lst[index]["img_files"][1], -1)

        first_img = cv2.resize(first_img, args["input_size"],
                           cv2.INTER_LINEAR)

        second_img = cv2.resize(second_img, args["input_size"],
                            cv2.INTER_LINEAR)

        first_img = np.stack((first_img, first_img, first_img),
                             axis=-1)
        second_img = np.stack((second_img, second_img, second_img),
                              axis=-1)

        first_img = first_img / 255.0
        second_img = second_img / 255.0

        first_img = torch.from_numpy(
            first_img.astype(np.float32)).permute(2, 0, 1)
        second_img = torch.from_numpy(
            second_img.astype(np.float32)).permute(2, 0, 1)

        gt_t = torch.FloatTensor(self._lst[index]["pose"][:3])
        gt_r = torch.FloatTensor(self._lst[index]["pose"][3:])

        if args["hflip"] and self.split == "train":
            hflip = random.random() < 0.5
            if hflip:
                H = torch.diag(torch.FloatTensor([1, -1, 1]))
                first_img = T.functional.hflip(first_img)
                second_img = T.functional.hflip(second_img)
                C = torch.matmul(torch.matmul(H, exp_SO3(gt_r)), H.transpose(0, 1))
                gt_t = torch.matmul(H, gt_t)
                gt_r = log_SO3(C)
        
        image = torch.cat((first_img, second_img), 0)

        # 0 for machine hall (no rotation gt), 1 for vicon room
        dataset_idx = self._lst[index]["dataset_idx"]

        return image, gt_t, gt_r, dataset_idx

    def __len__(self):
        return len(self._lst)

    @staticmethod #hat nicht viel geholfen -> 
    def get_files(path, split):
        """
        :param path: "your/path/euroc_dataset"
        :return: list of image file paths and label file paths
        """
        # bags = os.listdir(path)
        if split == "train":
            bags = args['train_split']
        else:
            bags = args['test_split']
        files = []
        for bag_name in bags:
            tmp = []

            bag_path = os.path.join(path, bag_name)
            img_files = sorted(glob.glob(os.path.join(bag_path,
                                                      "images/*.png")),
                               key=lambda k: int((k.split("/")[-1][4:]).split(".")[0]))
            for idx, img_file in enumerate(img_files):
                data_dict = {"img_file": img_file}

                with open(os.path.join(bag_path + "/poses.csv"),
                          mode='r') as f:
                    reader = csv.reader(f)
                    pose = list(reader)[idx][1:]
                    data_dict["pose"] = pose

                tmp.append(data_dict)

            if split == "train":
                paired_files = EuRoC.pairing(tmp)
                if args["reverse"]:
                    paired_files_reversed = EuRoC.pairing(list(reversed(tmp)))
                    paired_files = paired_files + paired_files_reversed
            else:
                paired_files = EuRoC.pairing_test(tmp)
            files += paired_files

        return files

    @staticmethod
    def load_gt(path, split):
        ret = []
        for paired_files in EuRoC.get_files(path, split):
            ret.append(
                {
                    "img_files": paired_files['img_files'],
                    "pose": paired_files['pose'],
                    "dataset_idx": paired_files['dataset_idx']
                }
            )

        return ret

    @staticmethod
    def pairing(files):
        paired_files = []
        # discard last several frames
        for idx in range(len(files) - 5):
            random.seed(13)
            next_idx1 = 1 + idx
            next_idx2 = 2 + idx
            next_idx3 = random.randint(3, 4) + idx

            dict1 = {'img_files': (files[idx]["img_file"],
                                  files[next_idx1]["img_file"])}
            dict2 = {'img_files': (files[idx]["img_file"],
                                   files[next_idx2]["img_file"])}
            dict3 = {'img_files': (files[idx]["img_file"],
                                   files[next_idx3]["img_file"])}

            if len(files[idx]["pose"]) == 3:
                t1 = list(map(float, files[idx]["pose"][:3]))
                t2 = list(map(float, files[next_idx1]["pose"][:3]))
                t3 = list(map(float, files[next_idx2]["pose"][:3]))
                t4 = list(map(float, files[next_idx3]["pose"][:3]))

                t_diff12 = [x - y for x, y in zip(t2, t1)]
                t_diff13 = [x - y for x, y in zip(t3, t1)]
                t_diff14 = [x - y for x, y in zip(t4, t1)]

                r_diff12 = [0.0, 0.0, 0.0]
                r_diff13 = [0.0, 0.0, 0.0]
                r_diff14 = [0.0, 0.0, 0.0]

                # 0 for machine hall (no rotation gt), 1 for vicon room
                dataset_idx = 0

            else:
                t1 = list(map(float, files[idx]["pose"][:3]))
                t2 = list(map(float, files[next_idx1]["pose"][:3]))
                t3 = list(map(float, files[next_idx2]["pose"][:3]))
                t4 = list(map(float, files[next_idx3]["pose"][:3]))

                q1 = list(map(float, files[idx]["pose"][3:]))
                q2 = list(map(float, files[next_idx1]["pose"][3:]))
                q3 = list(map(float, files[next_idx2]["pose"][3:]))
                q4 = list(map(float, files[next_idx3]["pose"][3:]))

                q1 = np.quaternion(*q1)
                q2 = np.quaternion(*q2)
                q3 = np.quaternion(*q3)
                q4 = np.quaternion(*q4)

                r1 = quaternion.as_rotation_matrix(q1)
                r2 = quaternion.as_rotation_matrix(q2)
                r3 = quaternion.as_rotation_matrix(q3)
                r4 = quaternion.as_rotation_matrix(q4)

                SE3_1 = np.identity(4)
                SE3_1[0:3, 0:3] = r1
                SE3_1[0][3] = t1[0]
                SE3_1[1][3] = t1[1]
                SE3_1[2][3] = t1[2]

                SE3_2 = np.identity(4)
                SE3_2[0:3, 0:3] = r2
                SE3_2[0][3] = t2[0]
                SE3_2[1][3] = t2[1]
                SE3_2[2][3] = t2[2]

                SE3_3 = np.identity(4)
                SE3_3[0:3, 0:3] = r3
                SE3_3[0][3] = t3[0]
                SE3_3[1][3] = t3[1]
                SE3_3[2][3] = t3[2]

                SE3_4 = np.identity(4)
                SE3_4[0:3, 0:3] = r4
                SE3_4[0][3] = t4[0]
                SE3_4[1][3] = t4[1]
                SE3_4[2][3] = t4[2]

                SE3_diff12 = np.linalg.inv(SE3_1).dot(SE3_2)
                SE3_diff13 = np.linalg.inv(SE3_1).dot(SE3_3)
                SE3_diff14 = np.linalg.inv(SE3_1).dot(SE3_4)

                # the translation from T
                t_diff12 = SE3_diff12[0:3, 3]
                t_diff13 = SE3_diff13[0:3, 3]
                t_diff14 = SE3_diff14[0:3, 3]

                r_diff12 = SE3_diff12[0:3, 0:3]
                r_diff13 = SE3_diff13[0:3, 0:3]
                r_diff14 = SE3_diff14[0:3, 0:3]

                r_diff12 = R.from_matrix(r_diff12)
                r_diff13 = R.from_matrix(r_diff13)
                r_diff14 = R.from_matrix(r_diff14)

                r_diff12 = r_diff12.as_rotvec()
                r_diff13 = r_diff13.as_rotvec()
                r_diff14 = r_diff14.as_rotvec()

                # 0 for machine hall (no rotation gt), 1 for vicon room
                dataset_idx = 1

            dict1['pose'] = [*t_diff12, *r_diff12]
            dict1['dataset_idx'] = dataset_idx

            dict2['pose'] = [*t_diff13, *r_diff13]
            dict2['dataset_idx'] = dataset_idx

            dict3['pose'] = [*t_diff14, *r_diff14]
            dict3['dataset_idx'] = dataset_idx

            paired_files.append(dict1)
            paired_files.append(dict2)
            paired_files.append(dict3)

        return paired_files

    @staticmethod
    def pairing_test(files):
        paired_files = []
        # discard last several frames
        for idx in range(len(files) - 3):
            # random.seed(13)
            next_idx1 = 1 + idx

            dict1 = {'img_files': (files[idx]["img_file"],
                                   files[next_idx1]["img_file"])}

            if len(files[idx]["pose"]) == 3:
                t1 = list(map(float, files[idx]["pose"][:3]))
                t2 = list(map(float, files[next_idx1]["pose"][:3]))

                t_diff12 = [x - y for x, y in zip(t2, t1)]

                r_diff12 = [0.0, 0.0, 0.0]

                # 0 for machine hall (no rotation gt), 1 for vicon room
                dataset_idx = 0

            else:
                t1 = list(map(float, files[idx]["pose"][:3]))
                t2 = list(map(float, files[next_idx1]["pose"][:3]))

                q1 = list(map(float, files[idx]["pose"][3:]))
                q2 = list(map(float, files[next_idx1]["pose"][3:]))

                q1 = np.quaternion(*q1)
                q2 = np.quaternion(*q2)

                r1 = quaternion.as_rotation_matrix(q1)
                r2 = quaternion.as_rotation_matrix(q2)

                SE3_1 = np.identity(4)
                SE3_1[0:3, 0:3] = r1
                SE3_1[0][3] = t1[0]
                SE3_1[1][3] = t1[1]
                SE3_1[2][3] = t1[2]

                SE3_2 = np.identity(4)
                SE3_2[0:3, 0:3] = r2
                SE3_2[0][3] = t2[0]
                SE3_2[1][3] = t2[1]
                SE3_2[2][3] = t2[2]

                SE3_diff = np.linalg.inv(SE3_1).dot(SE3_2)

                # the translation from T
                t_diff12 = SE3_diff[0:3, 3]

                r_diff12 = SE3_diff[0:3, 0:3]

                r_diff12 = R.from_matrix(r_diff12)

                r_diff12 = r_diff12.as_rotvec()

                # 0 for machine hall (no rotation gt), 1 for vicon room
                dataset_idx = 1

            dict1['pose'] = [*t_diff12, *r_diff12]
            dict1['dataset_idx'] = dataset_idx

            paired_files.append(dict1)

        return paired_files

    @staticmethod
    def to8bit(path):
        bags = ['MH_01_easy', 'MH_02_easy',
                'MH_04_difficult', 'MH_05_difficult',
                'V1_01_easy', 'V1_03_difficult',
                'V2_01_easy', 'V2_03_difficult',
                'MH_03_medium', 'V1_02_medium',
                'V2_02_medium']
        for bag_name in bags:
            bag_path = os.path.join(path, bag_name)
            img_files = glob.glob(os.path.join(bag_path, "images/*.png"))
            for file in img_files:
                img_name = file.split("/")[-1]
                img = cv2.imread(file, -1)
                img = img[:, 64:-1]
                height, width = img.shape
                img[img==65520] = 0
                img[img>1400]=1399
                img = img / 1400 * 255
                img = img.astype('uint8')
                out_dir = "/home/julius/Downloads/euroc" + "/" + bag_name + "/" + "images"
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(out_dir + "/" + img_name, img)


if __name__ == '__main__':
    EuRoC.to8bit("/home/julius/euroc_dataset")