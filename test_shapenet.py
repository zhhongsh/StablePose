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
from datasets.shapenet.dataset_shapenet_eval import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network_noid import PatchNet
from lib.loss_noid import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
import open3d as o3d
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='tless', help='tless or linemod')
parser.add_argument('--dataset_root', type=str, default='/home/lthpc/yifeis/symmetry_pose_mount/render/render_dy/render_wt_pt_proj/data/syn_images_20views1',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
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
parser.add_argument('--resume_posenet', type=str, default='pose_model_1_0.5772390158245474.pth', help='resume PoseNet model')#pose_model_2_193909.25539978288.pth
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
proj_dir = '/home/lthpc/yifeis/pose/pose_est_tless_3d/'
# proj_dir = '/home/dell/yifeis/pose/pose_est_tless_3d/'
torch.set_num_threads(64)

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'tless':
        opt.num_objects = 30  # number of object classes in the dataset
        opt.num_points = 2000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/shapenet/nobatchnorm/'  # folder to save trained models
        opt.log_dir = proj_dir + 'experiments/logs/shapenet/nobatchnorm/'  # folder to save logs
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
    # refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects)
    # refiner.cuda()
    # utils.print_network(estimator)
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        # refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = False  # True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        # optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=0.01)

    if opt.dataset == 'tless':
        dataset = PoseDataset_ycb('train_100', opt.num_points, False, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, False, opt.dataset_root, opt.noise_trans,
                                      opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers,
                                             pin_memory=True)
    if opt.dataset == 'tless':
        test_dataset = PoseDataset_ycb('test_100', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                                 pin_memory=True)

    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\n'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh))

    criterion = Loss(opt.num_points_mesh)
    # criterion_refine = Loss_refine(opt.num_points_mesh)

    best_test = np.Inf
    st_time = time.time()

    train_count = 0
    train_dis_avg = 0.0
    train_patch_avg = 0.0
    train_norm_avg = 0.0

    # if opt.refine_start:
    #     estimator.eval()
    #     refiner.train()
    # else:
    #     estimator.train()
    estimator.eval()
    optimizer.zero_grad()
    result_list = []
    scene_id_ls = []
    im_id_ls = []
    obj_id_ls = []
    r_ls = []
    score_ls = []
    t_ls = []
    time_ls = []
    dis_ls = []
    occ_ls = []
    fit_before = []
    rmse_before = []
    fit_after = []
    rmse_after = []
    with torch.no_grad():
        for j, data in enumerate(testdataloader, 0):
            time_st = time.time()
            points, target_rt, choose_patchs, target_pt, model_points, target_mode, target_s,data_path = data

            points, target_rt, target_pt, model_points, target_mode, target_s = Variable(points).cuda(), \
                                                                                Variable(target_rt).cuda(), \
                                                                                Variable(target_pt).cuda(), \
                                                                                Variable(model_points).cuda(), \
                                                                                Variable(target_mode).cuda(), \
                                                                                Variable(target_s).cuda()

            pred_r, pred_t, pred_choose, pred_a = estimator(points, choose_patchs)
            loss, dis, norm_loss, patch_loss, r_pred, t_pred, pred = criterion(pred_r, pred_t, pred_a, pred_choose,
                                                                               target_rt,
                                                                               points, target_pt, model_points,
                                                                               target_mode, target_s)
            target_mode =target_mode.item()
            if target_mode==1:
                r_pred = r_pred.detach().cpu().numpy().T.reshape(9).tolist()
                r_pred_s = str(r_pred)[1:-1].replace(',', ' ')
                t_pred = t_pred.view(3).detach().cpu().numpy().reshape(3) * 1000
                t_pred_s = str(t_pred.tolist())[1:-1].replace(',', ' ')
                score = 1
                # occ = occ.numpy()[0]
                dis = dis.detach().cpu().numpy()
                pred = pred.view(-1, 3).detach().cpu().numpy()
                view_point = points.detach().cpu().numpy().reshape(-1, 3)
                target = target_pt.view(-1, 3).detach().cpu().numpy()
                model = model_points.view(-1, 3).detach().cpu().numpy()
                r_array = np.array(r_pred).reshape(3, 3)
                t_array = np.array(t_pred).reshape(3, 1) / 1000
                attach = np.array([0, 0, 0, 1]).reshape(1, 4)
                trans_init = np.append(r_array, t_array, axis=1)
                trans_init = np.append(trans_init, attach, axis=0)

                model_max_x = np.max(model[:, 0]) - np.min(model[:, 0])
                model_max_y = np.max(model[:, 1]) - np.min(model[:, 1])
                model_max_z = np.max(model[:, 2]) - np.min(model[:, 2])
                model_d = max([model_max_x, model_max_y, model_max_z])
                mindis = 0.1 * model_d

                pcd_model = o3d.geometry.PointCloud()
                pcd_model.points = o3d.utility.Vector3dVector(model)
                pcd_pred = o3d.geometry.PointCloud()
                pcd_pred.points = o3d.utility.Vector3dVector(pred)
                pcd_view = o3d.geometry.PointCloud()
                pcd_view.points = o3d.utility.Vector3dVector(view_point)

                evaluation = o3d.registration.evaluate_registration(pcd_model, pcd_view, mindis, trans_init)
                fitness = evaluation.fitness
                inlier_rmse = evaluation.inlier_rmse
                fit_before.append(fitness)
                score = fitness
                rmse_before.append(inlier_rmse)
                # print('dis=', dis, 'fitness=', fitness, 'inlier_rmse=', inlier_rmse)

                reg_p2p = o3d.registration.registration_icp(pcd_model, pcd_view, mindis, trans_init,
                                                            o3d.registration.TransformationEstimationPointToPoint(),
                                                            o3d.registration.ICPConvergenceCriteria(max_iteration=3000))
                fit_after.append(reg_p2p.fitness)
                score = reg_p2p.fitness
                rmse_after.append(reg_p2p.inlier_rmse)
                print(reg_p2p)

                transform = reg_p2p.transformation

                r_pred = transform[:3, :3].reshape(9).tolist()
                r_pred_s = str(r_pred)[1:-1].replace(',', ' ')
                t_pred = transform[:3, 3].reshape(3)
                t_pred_s = str(t_pred.tolist())[1:-1].replace(',', ' ')

                im_id_ls.append(data_path[0])
                r_ls.append(r_pred_s)
                t_ls.append(t_pred_s)
                score_ls.append(score)
                time_ls.append(-1)
                dis_ls.append(dis)
            else:
                choose_r = pred_choose.detach().cpu().numpy().reshape(-1)
                r_pred = r_pred.detach().cpu().numpy()
                t_pred = t_pred.view(3).detach().cpu().numpy().reshape(3) * 1000
                r_pred_ = np.transpose(r_pred, [0, 2, 1])
                pred_ = pred.view(3, -1, 3).detach().cpu().numpy()
                dis = dis.detach().cpu().numpy()
                fit_ls = []
                for r_id in range(3):
                    r_pred = r_pred_[r_id, :, :]
                    r_pred = r_pred.reshape(9).tolist()
                    r_pred_s = str(r_pred)[1:-1].replace(',', ' ')
                    t_pred_s = str(t_pred.tolist())[1:-1].replace(',', ' ')
                    score = 1
                    pred = pred_[r_id, :, :]
                    view_point = points.detach().cpu().numpy().reshape(-1, 3)
                    model = model_points.view(-1, 3).detach().cpu().numpy()
                    r_array = np.array(r_pred).reshape(3, 3)
                    t_array = np.array(t_pred).reshape(3, 1) / 1000
                    attach = np.array([0, 0, 0, 1]).reshape(1, 4)
                    trans_init = np.append(r_array, t_array, axis=1)
                    trans_init = np.append(trans_init, attach, axis=0)

                    model_max_x = np.max(model[:, 0]) - np.min(model[:, 0])
                    model_max_y = np.max(model[:, 1]) - np.min(model[:, 1])
                    model_max_z = np.max(model[:, 2]) - np.min(model[:, 2])
                    model_d = max([model_max_x, model_max_y, model_max_z])
                    mindis = 0.1 * model_d

                    pcd_model = o3d.geometry.PointCloud()
                    pcd_model.points = o3d.utility.Vector3dVector(model)
                    pcd_pred = o3d.geometry.PointCloud()
                    pcd_pred.points = o3d.utility.Vector3dVector(pred)
                    pcd_view = o3d.geometry.PointCloud()
                    pcd_view.points = o3d.utility.Vector3dVector(view_point)

                    evaluation = o3d.registration.evaluate_registration(pcd_model, pcd_view, mindis, trans_init)
                    fitness = evaluation.fitness
                    inlier_rmse = evaluation.inlier_rmse
                    fit_before.append(fitness)
                    score = fitness
                    rmse_before.append(inlier_rmse)
                    # print('dis=', dis, 'fitness=', fitness, 'inlier_rmse=', inlier_rmse)

                    reg_p2p = o3d.registration.registration_icp(pcd_model, pcd_view, mindis, trans_init,
                                                                o3d.registration.TransformationEstimationPointToPoint(),
                                                                o3d.registration.ICPConvergenceCriteria(
                                                                    max_iteration=3000))
                    fit_after.append(reg_p2p.fitness)
                    score = reg_p2p.fitness
                    rmse_after.append(reg_p2p.inlier_rmse)
                    print(reg_p2p)
                    transform = reg_p2p.transformation

                    r_pred = transform[:3, :3].reshape(9).tolist()
                    r_pred_s = str(r_pred)[1:-1].replace(',', ' ')
                    t_pred = transform[:3, 3].reshape(3)
                    t_pred_s = str(t_pred.tolist())[1:-1].replace(',', ' ')
                    fit_ls.append(reg_p2p.fitness)

                    im_id_ls.append(data_path[0])
                    r_ls.append(r_pred_s)
                    t_ls.append(t_pred_s)
                    score_ls.append(score)
                    time_ls.append(-1)
                    dis_ls.append(dis)
                print('time:', time.time()-time_st)

    dataframe = pd.DataFrame({'im_id': im_id_ls, 'score': score_ls,'R': r_ls, 't': t_ls, 'time': time_ls})
    dataframe.to_csv("/home/lthpc/yifeis/pose/pose_est_tless_3d/tools/shapenet/shapenet-all-ours.csv", index=False, sep=',')
    print('mean_fitness_before=', np.mean(fit_before))
    print('mean_rmse_before=', np.mean(rmse_before))
    print('mean_fitness_after=', np.mean(fit_after))
    print('mean_rmse_after=', np.mean(rmse_after))
    print('mean_dis=', np.mean(dis_ls))

if __name__ == '__main__':
    main()
