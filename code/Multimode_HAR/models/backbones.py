import torch
from torch import nn
from collections import OrderedDict
from .transformer_attention import *
from .attention import *


class FCN(nn.Module):
    def __init__(self, n_channels, n_timesteps, n_classes, out_channels=128):
        super(FCN, self).__init__()

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.out_len = n_timesteps
        for _ in range(3):
            self.out_len = (self.out_len + 1) // 2 + 1

        self.out_channels = out_channels  # 128
        self.out_dim = self.out_len * self.out_channels

        self.logits = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units

        self.classifier = nn.Linear(LSTM_units, n_classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        out = self.classifier(x)
        return out


class TPN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        kernel_list = [24, 16, 8]
        channels_size = [32, 64, 96]

        self.out_dim = channels_size[2]
        convNet_layers = [
            ('conv1d_1', nn.Conv1d(in_channels, channels_size[0], kernel_size=(kernel_list[0],), stride=(1,))),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.1)),
            ('conv1d_2', nn.Conv1d(channels_size[0], channels_size[1], kernel_size=(kernel_list[1],), stride=(1,))),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.1)),
            ('conv1d_3', nn.Conv1d(channels_size[1], channels_size[2], kernel_size=(kernel_list[2],), stride=(1,))),
            ('relu3', nn.ReLU()),
            ('dropout3', nn.Dropout(p=0.1)),
            ('globalMaxPool', nn.AdaptiveMaxPool1d(output_size=1)),
            ('flatten', nn.Flatten()),
        ]
        self.convNet = nn.Sequential(OrderedDict(convNet_layers))
        self.classifier = nn.Linear(self.out_dim, n_classes)

    def forward(self, x, data_format='channel_last'):
        if data_format == 'channel_last':
            x = x.permute(0, 2, 1)
        x = self.convNet(x)
        out = self.classifier(x)
        return out


class Transformer(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super(Transformer, self).__init__()

        self.out_dim = dim
        self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, n_classes=n_classes, dim=dim,
                                           depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.transformer(x)
        out = self.classifier(x)
        return out


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )
        self.ca1 = ChannelAttention(128)
        self.sa1 = SpatialAttention()
        self.eca1 = SKNet(128)

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
        )
        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()
        self.eca2 = SKNet(256)

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
        )
        self.ca3 = ChannelAttention(384)
        self.sa3 = SpatialAttention()
        self.eca3 = SKNet(384)

        self.fc = nn.Sequential(
            nn.Linear(96000, 21)
        )

    def forward(self, x):
        x = x.unsqueeze(1)        # [64,1,350,25]
        h1 = self.Block1(x)       # [64,128,116,25]
        # r = self.shortcut1(x)     # [64, 128, 116, 25]
        # h1 = self.ca1(h1) * h1  # [64,128,1,1] * [64, 128, 116, 25] = [64, 128, 116, 25]
        # h1 = self.eca1(h1)
        # h1 = self.sa1(h1) * h1  # [64, 1, 116, 25] * [64,128,116,25] = [64,128, 116, 25]

        # h1 = h1 + r               # [64, 128, 116, 25]
        h2 = self.Block2(h1)      # [64, 256, 38, 25]
        # r = self.shortcut2(h1)    # [64, 256, 38, 25]
        # h2 = self.ca2(h2) * h2  # [64,256,1,1] * [64, 256, 38, 25] = [64, 256, 38, 25]
        # h2 = self.eca2(h2)
        # h2 = self.sa2(h2) * h2  # [64,1,38,25] * [64, 256, 38, 25] = [64, 256, 38, 25]

        # h2 = h2 + r               # [64, 256, 38, 25]
        h3 = self.Block3(h2)      # [64, 384, 12, 25]
        # r = self.shortcut3(h2)    # [64, 384, 12, 25]
        # h3 = self.ca3(h3) * h3  # [64, 384, 12, 25]
        # h3 = self.eca3(h3)
        # h3 = self.sa3(h3) * h3  # [64, 384, 12, 25]

        # h3 = h3 + r               # [64, 384, 12, 25]
        x = h3.view(h3.size(0), -1)  # [64, 115200]
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        return x


class ConvBlockFixup(nn.Module):
    """
    Fixup convolution block
    """

    def __init__(self, filter_width, input_filters, nb_filters, dilation):
        super(ConvBlockFixup, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1),
                               dilation=(self.dilation, 1),
                               bias=False, padding='same')
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1),
                               dilation=(self.dilation, 1),
                               bias=False, padding='same')
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out += identity
        out = self.relu(out)

        return out

class ConvBlockSkip(nn.Module):
    """
    Convolution block with skip connection
    """

    def __init__(self, window_size, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlockSkip, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1),
                               dilation=(self.dilation, 1), padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1),
                               dilation=(self.dilation, 1), padding='same')
        self.seq_len = window_size - (filter_width + 1) * 2
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        identity = x
        if self.batch_norm:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.norm1(out)
            out = self.conv2(out)
            out += identity
            out = self.relu(out)
            out = self.norm2(out)
        else:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out += identity
            out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    """
    Normal convolution block
    """

    def __init__(self, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1),
                               dilation=(self.dilation, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1),
                               dilation=(self.dilation, 1))
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)
        return out

class DC_shallow_LSTM(nn.Module):
    def __init__(self):
        """
        DeepConvLSTM model based on architecture suggested by Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115)

        :param config: config dictionary containing all settings needed to initialize DeepConvLSTM; these include:
            - no_lstm:              whether not to use an lstm
            - pooling:              whether to use pooling layer
            - reduce_layer:         whether to use reduce layer
            - reduce_layer_output:  size of output of reduce layer
            - pool_type:            type of pooling
            - pool_kernel_width:    width of pooling kernel
            - window_size:          number of samples contained in each sliding window
            - final_seq_len:        length of the sequence after the applying each convolution layer
            - nb_channels:          number of sensor channels used (i.e. number of features)
            - nb_classes:           number of classes which are to be predicted (e.g. 2 if binary classification problem)
            - nb_units_lstm:        number of units within each hidden layer of the LSTM
            - nb_layers_lstm:       number of hidden layers of the LSTM
            - nb_conv_blocks:       number of convolution blocks
            - conv_block_type:      type of convolution blocks used
            - nb_filters:           number of filters employed in convolution blocks
            - filter_width:         size of the filters (1 x filter_width) applied within the convolution layer
            - dilation:             dilation factor for convolutions
            - batch_norm:           whether to use batch normalization
            - drop_prob:            dropout probability employed after each dense layer (e.g. 0.5 for 50% dropout probability)
            - weights_init:         type of weight initialization used
            - seed:                 random seed employed to ensure reproducibility of results
        """
        super(DC_shallow_LSTM, self).__init__()
        # parameters
        self.no_lstm = False  # False
        self.pooling = False  # False
        self.reduce_layer = False  # False
        self.reduce_layer_output = 8  # 8
        self.pool_type = 'max'  # 'max'
        self.pool_kernel_width = 2  # 2
        self.window_size = 300  # 300
        self.drop_prob = 0.5  # 0.5
        self.nb_channels = 25  # 25 特征维度
        self.nb_classes = 21  # 21
        self.weights_init = 'xavier_normal'  # 'xavier_normal'
        self.seed = 1  # 1
        # convolution settings
        self.nb_conv_blocks = 2  # 2
        self.conv_block_type = 'normal'  # 'normal'
        if self.conv_block_type == 'fixup':
            self.use_fixup = True
        else:
            self.use_fixup = False
        self.nb_filters = 64  # 64
        self.filter_width = 5  # 11
        self.dilation = 1  # 1
        self.batch_norm = False  # False
        # lstm settings
        self.nb_units_lstm = 128  # 128
        self.nb_layers_lstm = 1  # 1

        # define conv layers
        self.conv_blocks = []
        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = 1
            else:
                input_filters = self.nb_filters
            if self.conv_block_type == 'fixup':
                self.conv_blocks.append(
                    ConvBlockFixup(self.filter_width, input_filters, self.nb_filters, self.dilation))
            elif self.conv_block_type == 'skip':
                self.conv_blocks.append(
                    ConvBlockSkip(self.window_size, self.filter_width, input_filters, self.nb_filters,
                                  self.dilation,
                                  self.batch_norm))
            elif self.conv_block_type == 'normal':
                self.conv_blocks.append(
                    ConvBlock(self.filter_width, input_filters, self.nb_filters, self.dilation, self.batch_norm))
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        # define max pool layer
        if self.pooling:
            if self.pool_type == 'max':
                self.pool = nn.MaxPool2d((self.pool_kernel_width, 1))
            elif self.pool_type == 'avg':
                self.pool = nn.AvgPool2d((self.pool_kernel_width, 1))
        if self.reduce_layer:
            self.reduce = nn.Conv2d(self.nb_filters, self.reduce_layer_output, (self.filter_width, 1))
        self.final_seq_len = self.window_size - (self.filter_width - 1) * (self.nb_conv_blocks * 2)
        # define lstm layers
        if not self.no_lstm:
            self.lstm_layers = []
            for i in range(self.nb_layers_lstm):
                if i == 0:
                    if self.reduce_layer:
                        self.lstm_layers.append(
                            nn.LSTM(self.nb_channels * self.reduce_layer_output, self.nb_units_lstm))
                    else:
                        self.lstm_layers.append(nn.LSTM(self.nb_channels * self.nb_filters, self.nb_units_lstm))
                else:
                    self.lstm_layers.append(nn.LSTM(self.nb_units_lstm, self.nb_units_lstm))
            self.lstm_layers = nn.ModuleList(self.lstm_layers)
        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        # define classifier
        if self.no_lstm:
            if self.reduce_layer:
                self.fc = nn.Linear(self.reduce_layer_output * self.nb_channels, self.nb_classes)
            else:
                self.fc = nn.Linear(self.nb_filters * self.nb_channels, self.nb_classes)
        else:
            self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, 1, self.window_size, self.nb_channels)  # [128, 1, 300, 25]
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)  # [128, 64, 260, 25]
        if self.pooling:
            x = self.pool(x)
            self.final_seq_len = x.shape[2]
        if self.reduce_layer:
            x = self.reduce(x)
            self.final_seq_len = x.shape[2]
        # permute dimensions and reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # [128, 260, 64, 25]
        if self.reduce_layer:
            x = x.reshape(-1, self.final_seq_len, self.nb_channels * self.reduce_layer_output)
        else:
            x = x.reshape(-1, self.final_seq_len, self.nb_filters * self.nb_channels)  # [128, 260, 64*25]
        if self.no_lstm:
            if self.reduce_layer:
                x = x.view(-1, self.nb_channels * self.reduce_layer_output)
            else:
                x = x.view(-1, self.nb_filters * self.nb_channels)
        else:
            for lstm_layer in self.lstm_layers:
                x, _ = lstm_layer(x)  # [128, 260, 128]
            # reshape data for classifier
            x = x.view(-1, self.nb_units_lstm)  # [33280, 128]
            x = self.dropout(x)
        x = self.fc(x)  # [33280, 21]
        # reshape data and return predicted label of last sample within final sequence (determines label of window)
        out = x.view(-1, self.final_seq_len, self.nb_classes)  # [128,260,21]

        return out[:, -1, :]  # [128, 21]

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetTA(nn.Module):
    def __init__(self):
        super(ResNetTA, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.att1 = TripletAttention()

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
        )
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.att2 = TripletAttention()

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.att3 = TripletAttention()

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
        )
        self.fc = nn.Sequential(
            nn.Linear(64000, 21)
        )

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        out1 = self.Block1(x)
        out1 = self.att1(out1)
        y1 = self.shortcut1(x)
        out = y1 + out1

        out2 = self.Block2(out)
        out2 = self.att2(out2)
        y2 = self.shortcut2(out)
        out = y2 + out2

        out3 = self.Block3(out)
        out3 = self.att3(out3)
        y3 = self.shortcut3(out)
        out = y3 + out3

        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda()
        return out


class CNN_TA(nn.Module):

    def __init__(self):
        super(CNN_TA, self).__init__()
        conv1 = nn.Conv2d(1, 64, (6, 1), (3, 1), padding=(1, 0))
        att1 = TripletAttention()

        conv2 = nn.Conv2d(64, 128, (6, 1), (3, 1), padding=(1, 0))
        att2 = TripletAttention()

        conv3 = nn.Conv2d(128, 256, (6, 1), (3, 1), padding=(1, 0))
        att3 = TripletAttention()

        self.conv_module = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            att1,

            conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            att2,

            conv3,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            att3
        )
        self.classifier = nn.Sequential(
            nn.Linear(64000, 21),
            # nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_module(x)
        x = torch.flatten(x, 1)
        # print(x.shape, 'x')
        x = self.classifier(x)
        return x
