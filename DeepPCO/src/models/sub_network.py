import torch
import torch.nn as nn
from configs.cfg import args
from utils.utils import correlate
from models.FlowNetC import flownetc


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels, kernel_size, stride):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride),
            nn.LeakyReLU(0.1, inplace=True))

        self._init_weight()

    def forward(self, x):
        return self.conv_block(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# fcn instead of fc
class FCNBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels, kernel_size=1):
        super().__init__()
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        self.fc_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.fc_block(x)


class FCBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(in_channels,
                      out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.fc_block(x)


# convolutional layers of flownetC
class ConvFlowNet(nn.Module):
    def __init__(self, model_path, pretrained=True):
        super().__init__()
        if pretrained:
            checkpoint = torch.load(model_path, map_location='cpu')
            flownet = flownetc(checkpoint)
        else:
            flownet = flownetc(None)

        conv_list = nn.ModuleList([])
        for name, child in flownet.named_children():
            if name == "deconv5":
                break
            else:
                conv_list.append(child)
        self.conv_block1a = nn.Sequential(*conv_list[:3])
        self.conv_block1b = nn.Sequential(*conv_list[:3])
        self.conv_redir = conv_list[3]
        self.conv_block2 = nn.Sequential(*conv_list[4:])
        del flownet

    def forward(self, x):
        x1 = x[:, :3]
        x2 = x[:, 3:]

        out_conv3a = self.conv_block1a(x1)
        out_conv3b = self.conv_block1b(x2)

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a, out_conv3b)

        x = torch.cat([out_conv_redir, out_correlation], dim=1)

        x = self.conv_block2(x)

        return x


class TranslationSubNet(nn.Module):
    """
    num_pxl: number of pixel in input image
    """
    def __init__(self, fast_inference=False):
        super(TranslationSubNet, self).__init__()
        in_ks = args["fc_ks_euroc"][0]

        self.fast_inference = fast_inference

        self.conv = nn.Sequential(ConvBlock(in_channels=6,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=2),
                                  ConvBlock(in_channels=64,
                                            out_channels=128,
                                            kernel_size=3,
                                            stride=2),
                                  ConvBlock(in_channels=128,
                                            out_channels=256,
                                            kernel_size=3,
                                            stride=2),
                                  ConvBlock(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=2),
                                  ConvBlock(in_channels=512,
                                            out_channels=128,
                                            kernel_size=1,
                                            stride=1))

        # self.fc_t = nn.Sequential(FCBlock(128, 512, in_ks),
        #                           FCBlock(512, 128),
        #                           FCBlock(128, 64),
        #                           nn.Dropout(0.1),
        #                           FCBlock(64, 16),
        #                           nn.Dropout(0.1),
        #                           FCBlock(16, 3))
        #
        # self.fc_r = nn.Sequential(FCBlock(128, 512, in_ks),
        #                           FCBlock(512, 128),
        #                           FCBlock(128, 64),
        #                           nn.Dropout(0.1),
        #                           FCBlock(64, 16),
        #                           nn.Dropout(0.1),
        #                           FCBlock(16, 3))

        self.fc_t = nn.Sequential(nn.Flatten(),
                                  FCBlock(in_ks[0] * in_ks[1] * 128, 512),
                                  FCBlock(512, 128),
                                  FCBlock(128, 64),
                                  nn.Dropout(p=args['dropout_rate']),
                                  FCBlock(64, 16),
                                  nn.Dropout(p=args['dropout_rate']),
                                  FCBlock(16, 3))

        self.fc_r = nn.Sequential(nn.Flatten(),
                                  FCBlock(in_ks[0] * in_ks[1] * 128, 512),
                                  FCBlock(512, 128),
                                  FCBlock(128, 64),
                                  nn.Dropout(p=args['dropout_rate']),
                                  FCBlock(64, 16),
                                  nn.Dropout(p=args['dropout_rate']),
                                  FCBlock(16, 3))

    def forward(self, x):
        x = self.conv(x)
        if self.fast_inference:
            return self.fc_t(x)
        t = self.fc_t(x)
        r = self.fc_r(x)

        return t, r


class RotationSubNet(nn.Module):
    """
    num_pxl: number of pixel in input image
    """
    def __init__(self, fast_inference=False):
        super(RotationSubNet, self).__init__()
        in_ks = args["fc_ks_euroc"][1]
        
        self.fast_inference = fast_inference

        self.flownet_conv = ConvFlowNet(args["flownet_path"],
                                        args["pretrained"])

        # self.fc_t = nn.Sequential(FCBlock(1024, 512, in_ks),
        #                           FCBlock(512, 128),
        #                           FCBlock(128, 64),
        #                           nn.Dropout(0.1),
        #                           FCBlock(64, 16),
        #                           nn.Dropout(0.1),
        #                           FCBlock(16, 3))
        #
        # self.fc_r = nn.Sequential(FCBlock(1024, 512, in_ks),
        #                           FCBlock(512, 128),
        #                           FCBlock(128, 64),
        #                           nn.Dropout(0.1),
        #                           FCBlock(64, 16),
        #                           nn.Dropout(0.1),
        #                           FCBlock(16, 3))

        self.fc_t = nn.Sequential(nn.Flatten(),
                                  FCBlock(in_ks[0] * in_ks[1] * 1024, 512),
                                  FCBlock(512, 128),
                                  FCBlock(128, 64),
                                  nn.Dropout(p=args['dropout_rate']),
                                  FCBlock(64, 16),
                                  nn.Dropout(p=args['dropout_rate']),
                                  FCBlock(16, 3))

        self.fc_r = nn.Sequential(nn.Flatten(),
                                  FCBlock(in_ks[0] * in_ks[1] * 1024, 512),
                                  FCBlock(512, 128),
                                  FCBlock(128, 64),
                                  nn.Dropout(p=args['dropout_rate']),
                                  FCBlock(64, 16),
                                  nn.Dropout(p=args['dropout_rate']),
                                  FCBlock(16, 3))

    def forward(self, x):
        x = self.flownet_conv(x)
        if self.fast_inference:
            return self.fc_r(x)
        t = self.fc_t(x)
        r = self.fc_r(x)

        return t, r
