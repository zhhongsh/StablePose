# the point only vision of 'train_patch_sym_relation2.py'
import torch.utils as utils
import argparse
import os
import random
import time
import numpy as np
import torch
import sys
sys.path.append('/home/lthpc/yifeis/pose/pose_est_tless_3d/')
# sys.path.append('/home/dell/yifeis/pose/pose_est_tless_3d/')
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.patch.dataset_triplet import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network_point import PatchNet, PoseRefineNet
from lib.loss_triplet_so import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='tless', help='tless or linemod')
parser.add_argument('--dataset_root', type=str, default='/data/yifeis/pose/',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=32, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.01, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.001, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max numbesr of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')#pose_model_2_193909.25539978288.pth
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
proj_dir = '/home/lthpc/yifeis/pose/pose_est_tless_3d/'
# proj_dir = '/home/dell/yifeis/pose/pose_est_tless_3d/'
torch.set_num_threads(32)

# proj_dir = '/home/demian/pose_est_tless/'
def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'tless':
        opt.num_objects = 30  # number of object classes in the dataset
        opt.num_points = 2000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/triplet/3_so_lthpc'  # folder to save trained models
        opt.log_dir = proj_dir + 'experiments/logs/triplet/3_so_lthpc'  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 15
        opt.num_points = 1000
        opt.outf = proj_dir +'trained_models/linemod/'
        opt.log_dir =  proj_dir +'experiments/logs/linemod/'
        opt.repeat_epoch = 2
    else:
        print('Unknown dataset')
        return
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    estimator = PatchNet(num_obj=opt.num_objects)
    # estimator = torch.nn.DataParallel(estimator)
    estimator = estimator.cuda()
    # estimator = torch.nn.parallel.DistributedDataParallel(estimator,find_unused_parameters=True)

    total_params = sum(p.numel() for p in estimator.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in estimator.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # print(estimator)
    refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects)
    refiner.cuda()
    # utils.print_network(estimator)
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = False  # True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=0.01)

    if opt.dataset == 'tless':
        dataset = PoseDataset_ycb('train', opt.num_points, False, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, False, opt.dataset_root, opt.noise_trans,
                                      opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers,
                                             pin_memory=True)
    if opt.dataset == 'tless':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                                 pin_memory=True)

    opt.sym_list = dataset.get_sym_list()
    nosym_list = dataset.get_nosym_list()
    rot_list = dataset.get_rot_list()
    ref_list = dataset.get_ref_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list,rot_list,ref_list,nosym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        train_patch_avg = 0.0
        train_norm_avg = 0.0

        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target_rt,target_trans, idx, \
                choose_patchs,target_pt,model_points,normals,model_info,model_axis,_,_ = data
                # if idx[0].item() not in ref_list:
                #     continue
                points, choose, img, target_rt, target_trans,idx,\
                target_pt, model_points,normals,model_axis = Variable(points).cuda(), \
                                                 Variable(choose).cuda(), \
                                                 Variable(img).cuda(), \
                                                 Variable(target_rt).cuda(), \
                                                 Variable(target_trans).cuda(),\
                                                 Variable(idx).cuda(), \
                                                 Variable(target_pt).cuda(),\
                                                 Variable(model_points).cuda(),\
                                                 Variable(normals).cuda(),\
                                                 Variable(model_axis).cuda()

                normal_ls = []
                for patch_id in range(len(choose_patchs)):
                    normal_ls.append(normals[0][choose_patchs[patch_id][0]])

                pred_r, pred_t, pred_choose = estimator(img, points, choose, choose_patchs, idx)
                loss, dis, norm_loss, patch_loss, r_pred, t_pred, _ = criterion(pred_r, pred_t, pred_choose, target_rt,
                                                            target_trans, idx, points,opt.w,
                                                            target_pt,model_points,
                                                            model_info)
                if opt.refine_start:
                    dis.backward()
                else:
                    loss.backward()

                torch.cuda.empty_cache()

                train_dis_avg += dis.item()
                train_patch_avg += patch_loss.item()
                train_norm_avg += norm_loss.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info(
                        'Train time {0} Epoch {1} Batch {2} Frame {3}  idx:{7} Avg_dis:{4} Avg_norm:{5} Avg_patch:{6}'.format(
                            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch,
                            int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size,
                                                                            train_norm_avg / opt.batch_size,
                                                                            train_patch_avg / opt.batch_size,
                                                                            idx))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0
                    train_norm_avg = 0
                    train_patch_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_patch = 0.0
        test_norm = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target_rt, target_trans, idx, \
            choose_patchs, target_pt, model_points, normals, model_info, model_axis, _, _ = data

            points, choose, img, target_rt, target_trans, idx, \
            target_pt, model_points, normals, model_axis = Variable(points).cuda(), \
                                                           Variable(choose).cuda(), \
                                                           Variable(img).cuda(), \
                                                           Variable(target_rt).cuda(), \
                                                           Variable(target_trans).cuda(), \
                                                           Variable(idx).cuda(), \
                                                           Variable(target_pt).cuda(), \
                                                           Variable(model_points).cuda(), \
                                                           Variable(normals).cuda(), \
                                                           Variable(model_axis).cuda()

            normal_ls = []
            for patch_id in range(len(choose_patchs)):
                normal_ls.append(normals[0][choose_patchs[patch_id][0]])

            pred_r, pred_t, pred_choose = estimator(img, points, choose, choose_patchs, idx)

            loss, dis, norm_loss, patch_loss, r_pred, t_pred, _ = criterion(pred_r, pred_t, pred_choose, target_rt,
                                                                            target_trans, idx, points, opt.w,
                                                                            target_pt, model_points,
                                                                            model_info)
            # if opt.refine_start:
            #     for ite in range(0, opt.iteration):
            #         pred_r, pred_t = refiner(new_points, emb, idx)
            #         dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, idx,
            #                                                        new_points)

            test_dis += dis.item()
            test_norm += norm_loss.item()
            test_patch += patch_loss.item()
            logger.info('Test time {0} Test Frame No.{1} idx:{5} dis:{2} norm_loss:{3} patch_loss:{4}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis, norm_loss,
                patch_loss,idx))

            test_count += 1

        test_dis = test_dis / test_count
        test_norm = test_norm / test_count
        test_patch = test_patch / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2} avg norm: {3} avg patch: {4}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis, test_norm, test_patch))
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
                                          opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
                                              opt.refine_start)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0,
                                                   opt.refine_start)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                         num_workers=opt.workers)

            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print(
                '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
                    len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

def displayPoint(data,target,view,title):
    # 解决中文显示问题
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams['axes.unicode_minus'] = False
    # 点数量太多不予显示
    while len(data[0]) > 20000:
        print("too much point")
        exit()
    # 散点图参数设置
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(title)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r', marker='.')
    ax.scatter3D(target[:, 0], target[:, 1], target[:, 2], c='b', marker='.')
    ax.scatter3D(view[:, 0], view[:, 1], view[:, 2], c='g', marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.close()
if __name__ == '__main__':
    main()
