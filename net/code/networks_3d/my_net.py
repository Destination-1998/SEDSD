import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res

class Mutil_scale(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        ############################student############################
        super(Mutil_scale, self).__init__()
        self.s_block_1 = nn.Sequential(
        # Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
        #                     mode_upsampling=up_type),
        # Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
        #                     mode_upsampling=up_type),
        # Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
        #                     mode_upsampling=up_type),
        # Upsampling_function(n_filters * 2, n_filters * 1, normalization=normalization,
        #                     mode_upsampling=up_type),
        nn.Conv3d(n_filters * 16, n_classes, 1, padding=0)
        )

        self.s_block_2 = nn.Sequential(
        # Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
        #                     mode_upsampling=up_type),
        # Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
        #                     mode_upsampling=up_type),
        # Upsampling_function(n_filters * 2, n_filters * 1, normalization=normalization,
        #                     mode_upsampling=up_type),
        nn.Conv3d(n_filters * 8, n_classes, 1, padding=0)
        )

        self.s_block_3 = nn.Sequential(
        # Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
        #                     mode_upsampling=up_type),
        # Upsampling_function(n_filters * 2, n_filters * 1, normalization=normalization,
        #                     mode_upsampling=up_type),
        nn.Conv3d(n_filters * 4, n_classes, 1, padding=0)
        )

        self.s_block_4 = nn.Sequential(
        # Upsampling_function(n_filters * 2, n_filters * 1, normalization=normalization,
        #                     mode_upsampling=up_type),
        nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)
        )

        self.s_block_5 = nn.Sequential(
        nn.Conv3d(n_filters, n_classes, 1, padding=0)
        )


        ############################teacher############################
        self.t_block_1 = nn.Sequential(
            # Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
            #                     mode_upsampling=up_type),
            # Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
            #                     mode_upsampling=up_type),
            # Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
            #                     mode_upsampling=up_type),
            # Upsampling_function(n_filters * 2, n_filters * 1, normalization=normalization,
            #                     mode_upsampling=up_type),
            nn.Conv3d(n_filters * 16, n_classes, 1, padding=0)
        )

        self.t_block_2 = nn.Sequential(
            # Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
            #                     mode_upsampling=up_type),
            # Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
            #                     mode_upsampling=up_type),
            # Upsampling_function(n_filters * 2, n_filters * 1, normalization=normalization,
            #                     mode_upsampling=up_type),
            nn.Conv3d(n_filters * 8, n_classes, 1, padding=0)
        )

        self.t_block_3 = nn.Sequential(
            # Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
            #                     mode_upsampling=up_type),
            # Upsampling_function(n_filters * 2, n_filters * 1, normalization=normalization,
            #                     mode_upsampling=up_type),
            nn.Conv3d(n_filters * 4, n_classes, 1, padding=0)
        )

        self.t_block_4 = nn.Sequential(
            # Upsampling_function(n_filters * 2, n_filters * 1, normalization=normalization,
            #                     mode_upsampling=up_type),
            nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)
        )

    def forward(self, f1, f2):
        feature1=[]
        feature1.append(self.t_block_2(f1[3]))
        feature1.append(self.t_block_3(f1[2]))
        feature1.append(self.t_block_4(f1[1]))

        feature2=[]
        feature2.append(self.s_block_2(f2[3]))
        feature2.append(self.s_block_3(f2[2]))
        feature2.append(self.s_block_4(f2[1]))

        return feature1, feature2


class Scale_Attention(nn.Module):
    def __init__(self, n_filters=16,):
        super(Scale_Attention, self).__init__()
        self.q = nn.Linear(n_filters * 2, n_filters * 2)
        self.k = nn.Linear(n_filters * 16, n_filters * 2)
        self.proj = nn.Linear(n_filters * 2, n_filters * 2)

    def forward(self, q, k, eps=.0001):
        b, c, h_k, w_k, d_k = k.shape
        k_squse = torch.flatten(k, -3).permute(0,2,1)
        k_squse = self.k(k_squse)#k_shape=b,n,c
        k_norm = torch.norm(k_squse, 2, 1, True)

        c, h_q, w_q, d_q= q.shape[1:]
        q = torch.flatten(q, -3).permute(0, 2, 1)
        q = self.q(q)
        q = q.reshape(b,h_q, w_q, d_q, c)
        q_matrix = torch.zeros_like(q)

        for i in range(h_q//h_k):
            for j in range(w_q//w_k):
                for k in range(d_q//d_k):
                    q_squse = q[:,(i * h_k):((i+1) * h_k),(j * w_k):((j+1) * w_k),(k * d_k):((k+1) * d_k),:]
                    q_squse = q_squse.reshape(b,h_k*w_k*d_k,c).permute(0,2,1) # q_shape=b,c,n
                    q_norm = torch.norm(q_squse, 2, 2, True)
                    similar_map = torch.bmm(k_squse, q_squse) / (torch.bmm(k_norm, q_norm) + eps)
                    similar_map = F.softmax(similar_map.permute(0,2,1), dim=-1)
                    similar_map = torch.bmm(similar_map, q_squse.permute(0,2,1))
                    similar_v = (similar_map - similar_map.min(1)[0].unsqueeze(1)) / (similar_map.max(1)[0].unsqueeze(1) - similar_map.min(1)[0].unsqueeze(1) + eps)
                    q_matrix[:, (i * h_k):((i + 1) * h_k), (j * w_k):((j + 1) * w_k), (k * d_k):((k + 1) * d_k), :] = similar_v.permute(0,2,1).reshape(b, h_k, w_k, d_k, c)
        q_matrix = self.proj(q_matrix)
        return q_matrix.permute(0,4,1,2,3)



class Decoder_5(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder_5, self).__init__()
        self.has_dropout = has_dropout

        # attention_block = Attention_Block
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)
        # self.block_six = attention_block(n_filters * 8, n_filters * 8)
        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # self.intra_attention = IntraSampleAttention(self.get_dim(), self.get_dim() * 4)
        self.inter_attention = InterSampleAttention(self.get_dim(), self.get_dim() * 4)

    def get_dim(self, ):
        return self.block_six.conv[6].weight.shape[0]

    def forward(self, features, with_feature=False, is_attention = False):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)

        if is_attention:
            # x6 = self.intra_attention(x6)
            x6 = self.inter_attention(x6)

        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)

        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        if with_feature:
            return out_seg, [x9,x8,x7,x6,x5]
        else:
            return out_seg


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet, self).__init__()

        self.encoder1 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        # self.encoder2 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        # self.encoder2 = EncoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)

        self.decoder1 = Decoder_5(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_5(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

        self.scale = Mutil_scale(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.s = torch.nn.Hardswish()
    def forward(self, input, is_training=False):
        features1 = self.encoder1(input)

        out_seg4, f1 = self.decoder1(features1, 1, is_attention=False)

        for i in range(len(features1)):
            features1[i] = features1[i] + self.s(f1[i] - features1[i]) * 0.3

        out_seg5, f2 = self.decoder2(features1, 1, is_attention=False)

        feature1, feature2 = self.scale(f1[:-1], f2[:-1])

        return feature1, feature2, out_seg4, out_seg5

        # return out_seg4, out_seg5



class InterSampleAttention(torch.nn.Module):
    """
        Implementation for inter-sample self-attention
        input size for the encoder_layers: [batch, h x w x d, dim]
    """
    def __init__(self, input_dim=256, hidden_dim=1024):
        super(InterSampleAttention, self).__init__()
        self.input_dim = input_dim
        self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)

    def forward(self, feature):
        if self.training:
            b, c, h, w, d = feature.shape
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(b, h * w * d, c)
            feature = self.encoder_layers(feature)
            feature = feature.view(b, h, w, d, c)
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature


class IntraSampleAttention(torch.nn.Module):
    """
    Implementation for intra-sample self-attention
    input size for the encoder_laye1rs: [h x w x d, batch, dim]
    """
    def __init__(self, input_dim=256, hidden_dim=1024):
        super(IntraSampleAttention, self).__init__()
        self.input_dim = input_dim
        self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)

    def forward(self, feature):
        if self.training:
            b, c, h, w, d = feature.shape
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(b, h * w * d, c)
            feature = feature.permute(1, 0, 2).contiguous()
            feature = self.encoder_layers(feature)
            feature = feature.permute(1, 0, 2).contiguous()
            feature = feature.view(b, h, w, d, c)
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature


class EncoderAuxiliary(nn.Module):
    """
    encoder for auxiliary model with CMA
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4):
        super(EncoderAuxiliary, self).__init__()
        self.insert_idx = insert_idx
        self.cma_type = cma_type

        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # print(self.get_dim(self.insert_idx))
        if self.cma_type == 'v2+':
            self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 4:
            return self.block_four.conv[6].weight.shape[0]

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        # cma layers
        if self.insert_idx == 4:
            if self.cma_type == "v2+":
                x4 = self.intra_attention(x4)
            x4 = self.inter_attention(x4)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class CAML3d_v1(nn.Module):
    """
    Use CMA on Encoder layer 4
    With different upsample
    """

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4, feat_dim=32):
        super(CAML3d_v1, self).__init__()
        self.cma_type = cma_type
        self.insert_idx = insert_idx
        assert self.insert_idx == 4
        self.encoder1 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.encoder2 = EncoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)

        self.decoder11 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder21 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

    def forward(self, input):
        features1 = self.encoder1(input)
        features2 = self.encoder2(input)
        out_seg1, embedding1 = self.decoder1(features1, with_feature=True)
        out_seg2, embedding2 = self.decoder2(features2, with_feature=True)
        return out_seg1, out_seg2, embedding1, embedding2







import torch
from torch import nn


class Channelblock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(Channelblock, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, 5, padding=2),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

        self.GlobalAveragePooling3D = nn.AdaptiveAvgPool3d(1)

        self.function = nn.Sequential(
            nn.Linear(n_filters_out * 2, n_filters_out),
            nn.BatchNorm1d(n_filters_out),
            nn.ReLU(inplace=True),
            nn.Linear(n_filters_out,n_filters_out),
            nn.Sigmoid()
        )

        self.convblock3 = nn.Sequential(
            nn.Conv3d(n_filters_out * 2, n_filters_out, 1, padding=0),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )


    def forward(self, data):
        conv1 = self.convblock1(data)
        conv2 = self.convblock2(data)

        data3 = torch.cat([conv1,conv2], dim=1)
        b, c, w, h, d = conv2.shape
        data3 = self.GlobalAveragePooling3D(data3)
        data3 = data3.flatten(1)

        a = self.function(data3)
        a = a.reshape(b, c, 1, 1, 1)
        a1 = 1 - a

        y = torch.mul(conv1, a)
        y1 = torch.mul(conv2, a1)

        data_all = torch.cat([y, y1], dim=1)
        data_all = self.convblock3(data_all)

        return data_all


class Spatialblock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(Spatialblock, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv3d(n_filters_out, n_filters_out, 1, padding=0),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv3d(n_filters_out * 2, n_filters_out, 1, padding=0),
            nn.BatchNorm3d(n_filters_out),
        )

        self.function = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(n_filters_out, 1, 1, padding=0),
            nn.Sigmoid()
        )


    def expend_as(tensor, data, repnum):
        my_repeat = data.repeat(1, repnum, 1, 1, 1)

        return my_repeat

    def forward(self, data, data_all):
        data = self.convblock1(data)
        data = self.convblock2(data)

        data3 = data + data_all
        data3 = self.function(data3)

        a = self.expend_as(data3, data_all.shape[1])
        y = torch.mul(a, data_all)

        a1 = 1-data3
        a1 = self.expend_as(a1, data_all.shape[1])
        y1 = torch.mul(a1, data)

        data_pro = torch.cat([y, y1], dim=1)
        data_pro = self.convblock3(data_pro)

        return data_pro


class Attention_Block(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(Attention_Block, self).__init__()
        self.conv11 = Channelblock(n_filters_in, n_filters_out)
        self.conv12 = Spatialblock(n_filters_in, n_filters_out)
        self.conv21 = Channelblock(n_filters_out, n_filters_out)
        self.conv22 = Spatialblock(n_filters_out, n_filters_out)


    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv12(x, x1)

        x3 = self.conv21(x2)
        x4 = self.conv22(x2, x3)
        return x4
