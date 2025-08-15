import os
import sys
import shutil
import os
import sys
import shutil
import argparse
import logging
import torch.nn as nn
import torch
# from albumentations import Compose, OneOf
from torch import sigmoid
from torchvision.transforms import Compose
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
# import albumentations as transforms
from adds.DistillKL import DistillKL
from adds.aggregator import Aggregator
from adds.cifar_sup_layer_mcl_meta_loss import Sup_MCL_Loss_Meta
from utils import ramps, losses, feature_memory, correlation, test_patch_MedicalImage_3D, test_patch_leafspot
from dataloaders.dataset import *
from networks_3d.net_factory_3d import net_factory
from adds.meta_network import LossWeightNetwork

def get_lambda_c(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_lambda_o(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency_o * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Pancreas', help='dataset_name')
parser.add_argument('--root_path', type=str, default=r'D:\lixin\data', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CAML', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=61,help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=12, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance supervised loss')
parser.add_argument('--consistency', type=float, default=1, help='lambda_c')
parser.add_argument('--consistency_o', type=float, default=0.05, help='lambda_s to balance sim loss')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--memory_num', type=int, default=256, help='num of embeddings per class in memory bank')
parser.add_argument('--embedding_dim', type=int, default=64, help='dim of embeddings to calculate similarity')
parser.add_argument('--num_filtered', type=int, default=12800,
                    help='num of unlabeled embeddings to calculate similarity')
parser.add_argument('--kd_T', type=float, default=3, help='temperature of KL-divergence')

args = parser.parse_args()

snapshot_path = "./model/{}_{}_{}_memory{}_feat{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}/{}".format(
    args.dataset_name,
    args.exp,
    args.labelnum,
    args.memory_num,
    args.embedding_dim,
    args.num_filtered,
    args.consistency,
    args.consistency_rampup,
    args.consistency_o,
    args.max_iteration,
    args.seed,
    args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + r'\LA'
train_data_path = args.root_path

if args.dataset_name == "Pancreas":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path + '/Pancreas'
train_data_path = args.root_path


if args.dataset_name == "zaoyibing":
    patch_size = (32, 32)
    args.root_path = args.root_path + args.dataset_name
train_data_path = args.root_path


if args.dataset_name == "ACDC":
    patch_size = (16, 160, 160)
    args.root_path = args.root_path + '/ACDC'
train_data_path = args.root_path


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),
                           with_idx=True)

    if args.dataset_name == "Pancreas":
        db_train = Pancreas(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),
                           with_idx=True)

    if args.dataset_name == "ACDC":
        db_train = ACDC(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),
                           with_idx=True)

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    criterion_cls = losses.Binary_dice_loss
    criterion_div = losses.mse_loss

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, idx = sampled_batch['image'].unsqueeze(1), sampled_batch['label'], sampled_batch['idx']
            volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()

            model.train()
            scale_pred_s, scale_pred, logits_s, logits = model(volume_batch, is_training=True)
            # logits_s, logits = model(volume_batch, is_training=True)

            logits_s = logits_s.softmax(1)
            logits = logits.softmax(1)

            loss_cls = torch.tensor(0.).cuda()
            loss_cls_mask = torch.tensor(0.).cuda()
            loss_logit_kd = torch.tensor(0.).cuda()

            for i in range(args.labeled_bs):
                loss_cls = loss_cls + criterion_cls(logits_s[i,1], label_batch[i])\
                                    + criterion_cls(logits[i,1], label_batch[i])

            for i in range(args.labeled_bs, args.batch_size):
                loss_cls_mask = loss_cls_mask + criterion_div(logits_s[i,1], logits[i,1])

            for i in range(len(scale_pred)):
                loss_logit_kd = loss_logit_kd + criterion_div(scale_pred_s[i].softmax(1)[:, 1], scale_pred[i].softmax(1)[:, 1])

            iter_num = iter_num + 1
            lambda_c = get_lambda_c(iter_num // 150)
            lambda_o = get_lambda_o(iter_num // 150)

            # loss = loss_cls / 2. + \
            #        loss_logit_kd * lambda_o + \
            #        loss_cls_mask * lambda_c / 2.

            loss = loss_cls / 2. + loss_cls_mask * lambda_c / 2. + loss_logit_kd * lambda_o / 2.

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f' % (
                iter_num, loss, loss_cls, loss_cls_mask, ))

            writer.add_scalar('Labeled_loss/loss_s', loss_cls, iter_num)
            writer.add_scalar('Co_loss/loss_c', loss_cls_mask, iter_num)


            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    metric1, metric2= test_patch_MedicalImage_3D.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                                   stride_xy=18, stride_z=4, dataset_name='LA')
                if args.dataset_name == "zaoyibing":
                    dice_sample = test_patch_leafspot.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                                   stride_xy=18, stride_z=4, dataset_name='zaoyibing')
                if args.dataset_name == "ACDC":
                    dice_sample = test_patch_MedicalImage_3D.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                                   stride_xy=18, stride_z=4, dataset_name='ACDC')
                if args.dataset_name == "Pancreas":
                    metric1, metric2 = test_patch_MedicalImage_3D.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                                   stride_xy=16, stride_z=16, dataset_name='Pancreas')
                # Notification!
                # Here we just save the best result to perform performance comparison with some SOTA methods that
                # report their best results during training obtained during training.
                # In our paper, we only use the model and corresponding results from the final training iteration.
                dice_sample1 = metric1[0]
                dice_sample2 = metric2[0]

                if dice_sample1 > best_dice or dice_sample2 > best_dice:

                    best_dice = max([dice_sample1, dice_sample2])
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice1_{}_dice2_{}.pth'.format(iter_num, dice_sample1, dice_sample2))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                else:
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice1_{}_dice2_{}.pth'.format(iter_num, dice_sample1, dice_sample2))
                    logging.info("save model to {}".format(save_mode_path))

                logging.info('iteration %d : dice1 : %03f, jc: %03f, 95hd: %03f, asd: %03f' % (
                    iter_num, metric1[0], metric1[1], metric1[2],metric1[3]))
                logging.info('iteration %d : dice2 : %03f, jc: %03f, 95hd: %03f, asd: %03f' % (
                    iter_num, metric2[0], metric2[1], metric2[2],metric2[3]))

                writer.add_scalar('Var_dice/Dice', dice_sample1, dice_sample2, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
            if iter_num >= max_iterations:
                iterator.close()
                break
    writer.close()

