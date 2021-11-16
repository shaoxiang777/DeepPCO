import os
import torch
import shutil
import datetime
import torch.nn as nn
import torch.nn.functional as F


def save_checkpoint(state_dict, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state_dict, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load custom correlation module"
                      "which is needed for FlowNetC", ImportWarning)


def mkdir(save_path):
    if not os.path.exists(save_path):
        raise Exception(f"The folder or path: \n{save_path}\n is not exist, please check the path or create the folder!")

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(save_path, time)
    if not os.path.exists(path):
        os.mkdir(path)
        print('create new folder' + str(path))
    else:
        print('folder exist')

    return path


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def exp_SO3(phi):
    phi_norm = torch.norm(phi)

    if phi_norm > 1e-8:
        unit_phi = phi / phi_norm
        unit_phi_skewed = skew3(unit_phi)
        C = torch.eye(3, 3, device=phi.device) + torch.sin(phi_norm) * unit_phi_skewed + \
            (1 - torch.cos(phi_norm)) * torch.mm(unit_phi_skewed, unit_phi_skewed)
    else:
        phi_skewed = skew3(phi)
        C = torch.eye(3, 3, device=phi.device) + phi_skewed + 0.5 * torch.mm(phi_skewed, phi_skewed)

    return C


def skew3(v):
    m = torch.zeros(3, 3, device=v.device)
    m[0, 1] = -v[2]
    m[0, 2] = v[1]
    m[1, 0] = v[2]

    m[1, 2] = -v[0]
    m[2, 0] = -v[1]
    m[2, 1] = v[0]

    return m


# assumes small rotations
def log_SO3(C):
    phi_norm = torch.acos(torch.clamp((torch.trace(C) - 1) / 2, -1.0, 1.0))
    if torch.sin(phi_norm) > 1e-6:
        phi = phi_norm * unskew3(C - C.transpose(0, 1)) / (2 * torch.sin(phi_norm))
    else:
        phi = 0.5 * unskew3(C - C.transpose(0, 1))

    return phi


def unskew3(m):
    return torch.stack([m[2, 1], m[0, 2], m[1, 0]])