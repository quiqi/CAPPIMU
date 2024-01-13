import torch
import torch.nn as nn
import math
from torch.nn import init
from torch.nn.parameter import Parameter


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape:[]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=(7, 1)):
        super(SpatialAttention, self).__init__()

        assert kernel_size in ((3, 1), (7, 1)), 'kernel size must be 3 or 7'
        padding = (3, 0) if kernel_size == (7, 1) else (1, 0)

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class TAMA(nn.Module):
    def __init__(self, in_channel, ratio=16):
        """
        :param in_channel:
        :param ratio: [2, 4, 8 ,16]
        """
        super(TAMA, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//ratio, 1),
            nn.BatchNorm2d(in_channel//ratio),
            nn.ReLU(True)
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channel//ratio, in_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, s = x.shape
        ta_out = torch.mean(x, dim=-1, keepdim=True)
        ma_out = torch.mean(x, dim=2, keepdim=True)
        ta_out = ta_out.permute(0, 1, 3, 2)
        concat_out = torch.cat([ma_out, ta_out], dim=3)  # S+T
        conv_out = self.conv_1(concat_out)
        split_ta = conv_out[:, :, :, s:].permute(0, 1, 3, 2)  # T
        split_ma = conv_out[:, :, :, :s]  # S
        f_t = self.block(split_ta)
        f_s = self.block(split_ma)

        return x * f_t * f_s


class ChannelEcaAtt(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ChannelEcaAtt, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


class SKNet(nn.Module):
    def __init__(self, features, M=3, r=2, stride=1, L=32):
        super(SKNet, self).__init__()
        d = max(int(features / r), L)  # d=
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i),
                    nn.BatchNorm2d(features),
                    nn.ReLU(False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)  # [bs, 1, c, h, w]
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)  # [bs, 2, c, h, w]
        fea_U = torch.sum(feas, dim=1)  # [bs, c, h, w]
        fea_s = fea_U.mean(-1).mean(-1)  # [bs, c]
        fea_z = self.fc(fea_s)  # [bs,d]
        for i, fc in enumerate(self.fcs):  #
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)  # [bs,2,c]
        attention_vectors = self.softmax(attention_vectors)  # [bs,2,c]
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)  # [bs,2,c,1,1]
        fea_v = (feas * attention_vectors).sum(dim=1)  # [bs, c, h, w]
        return fea_v


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(1, 0), dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, temperature):
        super(AttentionGate, self).__init__()
        kernel_size = (5, 1)
        self.temperature = temperature
        self.compress = ZPool()
        # self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(2, 0), relu=False)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        # print(x.shape, 'ty1')
        x_compress = self.compress(x)
        # print(x_compress.shape, 'Z_pooling')
        x_out = self.conv(x_compress)
        # print(x_out.shape, 'Conv+BN+RelU')
        # scale = torch.softmax(x_out/self.temperature, 1)
        scale = torch.sigmoid(x_out)
        # print((x*scale).shape, 'ty4')
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False, temperature=34):
        super(TripletAttention, self).__init__()

        self.cw = AttentionGate(temperature)
        self.hc = AttentionGate(temperature)
        self.no_spatial = no_spatial

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # initialization
        # self.w1 = torch.nn.init.normal_(self.w1)
        # self.w2 = torch.nn.init.normal_(self.w2)
        # self.w3 = torch.nn.init.normal_(self.w3)
        self.w1.data.fill_(1/3)
        self.w2.data.fill_(1/3)
        self.w3.data.fill_(1/3)

        if not no_spatial:
            self.hw = AttentionGate(temperature)

    def update_temperature(self):
        self.cw.updata_temperature()
        self.hc.updata_temperature()
        self.hw.updata_temperature()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        # print(x_out1.shape, 'ty44')
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            # print(x_out.shape, 'ty55')
            # x_out = x_out11
            # x_out = 1/3 * (x_out + x_out11 + x_out21)
            # x_out = 4 * x_out + 5 * x_out11 + 6 * x_out21
            x_out = self.w1 * x_out + self.w2 * x_out11 + self.w3 * x_out21
            # print(self.w1, self.w2, self.w3, 'w1,w2,w3')
            # print(x_out.shape, 'ty22')
        else:
            x_out = self.w1 * x_out11 + self.w2 * x_out21
        # return x_out, self.w1, self.w2, self.w3
        return x_out