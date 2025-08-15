from networks_3d.net_factory_3d import net_factory
from utils import test_patch_MedicalImage_3D

if __name__ == '__main__':
    num_classes = 2
    patch_size = (16, 160, 160)
    model = net_factory(net_type='caml3d_v1', in_chns=1, class_num=num_classes, mode="test")
    dice_sample = test_patch_MedicalImage_3D.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='ACDC')