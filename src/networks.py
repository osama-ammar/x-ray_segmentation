import torch
import torch.nn as nn

from torch.nn import init
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(
            ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, fmaps=16, dropout=0.2):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=fmaps)
        self.Conv2 = conv_block(ch_in=fmaps, ch_out=fmaps*2)
        self.Conv3 = conv_block(ch_in=fmaps*2, ch_out=fmaps*4)
        self.Conv4 = conv_block(ch_in=fmaps*4, ch_out=fmaps*8)
        self.Conv5 = conv_block(ch_in=fmaps*8, ch_out=fmaps*16)

        self.Up5 = up_conv(ch_in=fmaps*16, ch_out=fmaps*8)
        self.Up_conv5 = conv_block(ch_in=fmaps*16, ch_out=fmaps*8)

        self.Up4 = up_conv(ch_in=fmaps*8, ch_out=fmaps*4)
        self.Up_conv4 = conv_block(ch_in=fmaps*8, ch_out=fmaps*4)

        self.Up3 = up_conv(ch_in=fmaps*4, ch_out=fmaps*2)
        self.Up_conv3 = conv_block(ch_in=fmaps*4, ch_out=fmaps*2)

        self.Up2 = up_conv(ch_in=fmaps*2, ch_out=fmaps)
        self.Up_conv2 = conv_block(ch_in=fmaps*2, ch_out=fmaps)

        self.Conv_1x1 = nn.Conv2d(
            fmaps, output_ch, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.dropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.dropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.dropout(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.dropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class SMP(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        super(SMP, self).__init__()
        self.smp_model = smp.Unet(
            # choose encoder, e.g. mobilenet_v2 or imm-mobilenetv3_small_minimal_100
            encoder_name="timm-mobilenetv3_small_minimal_100",
            # use `imagenet` pre-trained weights for encoder initialization
            encoder_weights="imagenet",
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=img_ch,
            # model output channels (number of classes in your dataset)
            classes=output_ch,
        )

    def forward(self, x):
        output = self.smp_model(x)
        return output

