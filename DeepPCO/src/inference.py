import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from models.model import DeepPCO
from datasets.kitti import KITTI
from datasets.euroc import EuRoC
from configs.cfg import args
from scipy.spatial.transform import Rotation as R


if __name__ == '__main__':
    torch.cuda.set_device(6)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())

    net = DeepPCO().to(device)
    checkpoint = torch.load(args["net_path"], map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    test_set = KITTI("test")
    train_loader = DataLoader(test_set, 1,
                              num_workers=args['num_workers'],
                              shuffle=False)

    init_pose = [1.000000e+00, 1.197625e-11, 1.704638e-10, 1.665335e-16,
                1.197625e-11, 1.000000e+00, 3.562503e-10, -1.110223e-16,
                1.704638e-10, 3.562503e-10, 1.000000e+00, 2.220446e-16]

    SE3_prev = np.array([
        [init_pose[0], init_pose[1], init_pose[2], init_pose[3]],
        [init_pose[4], init_pose[5], init_pose[6], init_pose[7]],
        [init_pose[8], init_pose[9], init_pose[10], init_pose[11]],
        [0, 0, 0, 1]])

    for data in iter(train_loader):
        # with open("/home/ies/zhu/project/result.txt", 'a') as fd:
        #     fd.writelines(' '.join(str(i)) for i in SE3_prev.flatten()[:12])
        #     fd.write('\n')
        lst = SE3_prev.flatten()[:12]
        with open("/home/ies/zhu/project/05_pred.txt", 'a') as fd:
            fd.write(str(lst[0]) + ' ' + str(lst[1]) + ' ' + str(lst[2]) +
                     ' ' + str(lst[3]) + ' ' + str(lst[4]) + ' ' + str(lst[5]) +
                     ' ' + str(lst[6]) + ' ' + str(lst[7]) + ' ' + str(lst[8]) +
                     ' ' + str(lst[9]) + ' ' + str(lst[10]) + ' ' + str(lst[11]) + '\n')

        input = data[0].to(device)

        pred_t_t, pred_t_r, pred_r_t, pred_r_r = net(input)
        pred_t_t = pred_t_t.squeeze().cpu().detach().numpy()
        pred_r_r = pred_r_r.squeeze().cpu().detach().numpy()

        r = R.from_rotvec(pred_r_r)
        r = r.as_matrix().flatten()

        SE3_diff = np.array([
            [r[0], r[1], r[2], pred_t_t[0]],
            [r[3], r[4], r[5], pred_t_t[1]],
            [r[6], r[7], r[8], pred_t_t[2]],
            [0, 0, 0, 1]])
        print(SE3_diff)

        SE3_prev = SE3_prev.dot(SE3_diff)

    with open("/home/ies/zhu/project/05_pred.txt", 'a') as fd:
        fd.write(str(lst[0]) + ' ' + str(lst[1]) + ' ' + str(lst[2]) +
                 ' ' + str(lst[3]) + ' ' + str(lst[4]) + ' ' + str(lst[5]) +
                 ' ' + str(lst[6]) + ' ' + str(lst[7]) + ' ' + str(lst[8]) +
                 ' ' + str(lst[9]) + ' ' + str(lst[10]) + ' ' + str(lst[11]) + '\n')
