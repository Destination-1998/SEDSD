from networks_3d.unet_3d import UNet
from networks_3d.my_net import VNet, CAML3d_v1
from networks_3d.mutil_attention_model import CAML3d_v2
from networks_3d.teacher_student import Teacher, Student

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train", **kwargs):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "caml3d_v1" and mode == "train":
        net = CAML3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, **kwargs).cuda()
    elif net_type == "caml3d_v1" and mode == "test":
        net = CAML3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, **kwargs).cuda()
    elif net_type == "mutil_attention_model" and mode == "train":
        net = CAML3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, **kwargs).cuda()
    elif net_type == "student_teacher" and mode == "train":
        net_s = Student(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, **kwargs).cuda()
        net_t = Teacher(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, **kwargs).cuda()
        return net_s, net_t
    return net
