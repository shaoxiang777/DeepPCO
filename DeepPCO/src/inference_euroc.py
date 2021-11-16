import torch
import numpy as np
import quaternion
from torch.utils.data import DataLoader
from models.model import DeepPCO
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

    test_set = EuRoC("test")
    train_loader = DataLoader(test_set, 1,
                              num_workers=args['num_workers'],
                              shuffle=False)

    init_pose_t = [0.6711268770624547,2.1595768705867653,1.418139687890817]
    init_pose_r = np.quaternion(-0.011144622412415718,0.01075570700992646,-0.15716881684583092,0.9874502899737703)

    init_pose_rotation = quaternion.as_rotation_matrix(init_pose_r).flatten()
    SE3_prev = np.array([
        [init_pose_rotation[0], init_pose_rotation[1], init_pose_rotation[2], init_pose_t[0]],
        [init_pose_rotation[3], init_pose_rotation[4], init_pose_rotation[5], init_pose_t[1]],
        [init_pose_rotation[6], init_pose_rotation[7], init_pose_rotation[8], init_pose_t[2]],
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

        SE3_prev = SE3_prev.dot(SE3_diff)
        print(SE3_prev)

    with open("/home/ies/zhu/project/05_pred.txt", 'a') as fd:
        fd.write(str(lst[0]) + ' ' + str(lst[1]) + ' ' + str(lst[2]) +
                 ' ' + str(lst[3]) + ' ' + str(lst[4]) + ' ' + str(lst[5]) +
                 ' ' + str(lst[6]) + ' ' + str(lst[7]) + ' ' + str(lst[8]) +
                 ' ' + str(lst[9]) + ' ' + str(lst[10]) + ' ' + str(lst[11]) + '\n')
