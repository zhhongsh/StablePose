# dataloader of augmented original t-less dataset
import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import random
import numpy.ma as ma
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import heapq
proj_dir = os.getcwd()

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train':
            self.mode = 'train'
            self.path = proj_dir + '/datasets/tless/dataset_config/bop_train_list.txt'
        elif mode == 'test':
            self.mode = 'test'
            self.path = proj_dir + '/datasets/tless/dataset_config/bop_final_test_list.txt'
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.model_root = self.root
        # self.model_root = '/home/dell/yifeis/pose/bop2020/tless/'
        self.list = []
        # self.real = []
        self.syn = []
        input_file = open(self.path)  # open data folder(include png,depth,label mat)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            # if input_line[:5] == 'train_primesense/':
            #     self.real.append(input_line)
            # else:
            #     self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        # self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        model_info_file = open('{0}/models_reconst/models_info.json'.format(self.model_root), 'r', encoding='utf-8')
        self.model_info = json.load(model_info_file)

        self.cld = {}
        for class_id in range(1, 31):
            self.cld[class_id] = []
            mesh = o3d.io.read_triangle_mesh('{0}/models_reconst/obj_{1}.ply'.format(self.model_root, str(class_id).zfill(6)))
            pcd = mesh.sample_points_uniformly(number_of_points=10000)
            pcd = np.asarray(pcd.points)
            # displayPoint(pcd, pcd,k)
            self.cld[class_id] = pcd

        if self.mode == 'train':
            self.xmap = np.array([[j for i in range(400)] for j in range(400)])
            self.ymap = np.array([[i for i in range(400)] for j in range(400)])
            self.rt_list = []
            self.patch_num_list = []
            self.crop_size_list = []

            self.gt_list = []
            self.info_list = []
            self.cam_list = []
            N = 31
            for i in range(1, N):
                datadir = 'train_primesense/' + str(i).zfill(6)
                info_file = open('{0}/{1}/scene_gt_info.json'.format(self.root, datadir), 'r', encoding='utf-8')
                gt_file = open('{0}/{1}/scene_gt.json'.format(self.root, datadir), 'r', encoding='utf-8')
                cam_file = open('{0}/{1}/scene_camera.json'.format(self.root, datadir), 'r', encoding='utf-8')
                info = json.load(info_file)
                gt = json.load(gt_file)
                cam = json.load(cam_file)
                self.info_list.append(info)
                self.gt_list.append(gt)
                self.cam_list.append(cam)
                print('loading training ' + str(i) + 'json files')

        else:
            self.xmap = np.array([[j for i in range(720)] for j in range(540)])
            self.ymap = np.array([[i for i in range(720)] for j in range(540)])
            self.gt_list = []
            self.info_list = []
            self.cam_list = []
            self.patch_num_list = []
            ############# load json
            for i in range(1, 21):
                datadir = 'test_primesense/' + str(i).zfill(6)
                info_file = open('{0}/{1}/scene_gt_info.json'.format(self.root, datadir), 'r', encoding='utf-8')
                gt_file = open('{0}/{1}/scene_gt.json'.format(self.root, datadir), 'r', encoding='utf-8')
                cam_file = open('{0}/{1}/scene_camera.json'.format(self.root, datadir), 'r', encoding='utf-8')
                info = json.load(info_file)
                gt = json.load(gt_file)
                cam = json.load(cam_file)
                self.info_list.append(info)
                self.gt_list.append(gt)
                self.cam_list.append(cam)
                print('loading testing '+str(i)+' yml files')

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 100
        self.trans = transforms.ToTensor()
        self.norm1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm2 = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.symmetry_obj_idx = [1-1, 2-1, 3-1, 4-1, 13-1, 14-1, 15-1, 16-1, 17-1,
                                 24-1, 30-1,5-1, 6-1, 7-1, 8-1, 9-1, 10-1, 11-1,
                                 12-1, 19-1, 20-1, 23-1, 25-1, 26-1, 27-1, 28-1, 29-1]
        # self.rot_index = [1, 2, 3, 4, 13, 14, 15, 16, 17, 24, 30]
        # self.ref_index = [5, 6, 7, 8, 9, 10, 11, 12, 19, 20, 23, 25, 26, 27, 28, 29]
        # self.nosym_obj_idx = [18, 21, 22]
        self.rot_obj_idx = [1-1, 2-1, 3-1, 4-1, 13-1, 14-1, 15-1, 16-1, 17-1, 24-1, 30-1]
        self.ref_obj_idx = [5-1, 6-1, 7-1, 8-1, 9-1, 10-1, 11-1, 12-1, 19-1, 20-1,
                            23-1, 25-1, 26-1, 27-1, 28-1, 29-1]
        self.nosym_obj_idx = [18-1, 21-1, 22-1]

        self.num_pt_mesh_small = 2000
        self.num_pt_mesh_large = 1000
        self.refine = refine
        self.front_num = 2

        self.img_noise = True
        self.t_noise = True
        # print(len(self.list))

    def __getitem__(self, index):
        if self.mode == 'train':
            data_dir = self.list[index][:-7]
            dir_num = self.list[index][-13:-7]
            data_num = self.list[index][-6:]
            info = self.info_list[int(dir_num) - 1]
            gt = self.gt_list[int(dir_num) - 1]
            cam = self.cam_list[int(dir_num) - 1]
            choose_file = '{0}/{1}/{2}/{3}_choose.list'.format(self.root, data_dir, 'segmentation', data_num)
            label = Image.open(
                '{0}/{1}/{2}/{3}_{4}.png'.format(self.root, data_dir, 'mask_visib_occ', data_num, str(0).zfill(6)))
            img = Image.open('{0}/{1}/{2}/{3}.png'.format(self.root, data_dir, 'rgb_occ', data_num))
            depth = Image.open('{0}/{1}/{2}/{3}.png'.format(self.root, data_dir, 'depth', data_num))
            patch_file = Image.open('{0}/{1}/{2}/{3}.png'.format(self.root, data_dir, 'segmentation', data_num))
            normal_file = Image.open('{0}/{1}/{2}/{3}.png'.format(self.root, data_dir, 'normal', data_num))
            choose_file = '{0}/{1}/{2}/{3}_choose.list'.format(self.root, data_dir, 'segmentation', data_num)
            choose_ls = []
            stable_ls = []
            try:
                with open(choose_file) as f:
                    data = f.readlines()
                if len(data) > 1:
                    for ids in data:
                        choose_id = ids[:-1].split(',')[:-1]
                        stable = float(ids[:-1].split(',')[-1])
                        choose_ls.append([int(x) for x in choose_id])
                        stable_ls.append(stable)
                else:
                    if data[0] != '0':
                        choose_id = data[0].split(',')[:-1]
                        stable = float(data[0].split(',')[-1])
                        choose_ls.append([int(x) for x in choose_id])
                        stable_ls.append(stable)
                    else:
                        stable_ls.append(0)
            except(OSError):
                print('choose_list file not exist')
                stable_ls.append(0)
                choose_ls = []
                data = ['0']
            # if self.img_noise:
            #     img = self.trancolor(img)

            patch_label = np.array(patch_file)
            depth = np.array(depth)
            mask_occ = np.array(label)
            normal = np.array(normal_file)

            cam_k = cam[str(int(data_num))]['cam_K']
            depth_scale = cam[str(int(data_num))]['depth_scale']
            cam_k = np.array(cam_k).reshape(3, 3)
            obj_bb = info[str(int(data_num))][0]['bbox_visib']
            obj_id = gt[str(int(data_num))][0]['obj_id']

            model_info = self.model_info[str(obj_id)]

            depth_mask = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_not_equal(mask_occ, 0))
            mask = mask_label * depth_mask
            # mask_of_normal = mask.reshape(400, 400, 1).repeat(3, 2)
            # normal_masked = normal*mask_of_normal
            # mask_depth = mask * depth
            # mask_patch = mask * patch_label

            target_r = gt[str(int(data_num))][0]['cam_R_m2c']
            target_r = np.array(target_r).reshape(3, 3).T
            target_t = np.array(gt[str(int(data_num))][0]['cam_t_m2c'])
            target_t = target_t / 1000
            rt = np.append(target_r, target_t).reshape(1, 12)
            add = np.array([[0, 0, 0, 1]])
            target_trans = np.append(target_r.T, target_t.reshape(3, 1), axis=1)
            target_trans = np.append(target_trans, add, axis=0)

            rmin, rmax, cmin, cmax = get_bbox(mask_label)

            img_masked = self.trans(img)[:, rmin:rmax, cmin:cmax]
            img_masked = torch.zeros(img_masked.shape)
            # img_masked1 = self.norm1(img_masked[:, rmin:rmax, cmin:cmax])
            # img_masked2 = self.norm1(img_masked[:, rmin:rmax, cmin:cmax])
            # img = np.array(img)
            # img_masked_ = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > self.num_pt:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                if len(choose) == 0:
                    print(0)
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

            normal_maskd = normal[rmin:rmax, cmin:cmax].reshape(-1,3)[choose][:,:, np.newaxis].astype(np.float32)
            patch_masked = patch_label[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cam_cx = cam_k[0, 2]
            cam_cy = cam_k[1, 2]
            cam_fx = cam_k[0, 0]
            cam_fy = cam_k[1, 1]

            pt3 = patch_masked
            pt2 =depth_masked*depth_scale / 1000
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2, pt3), axis=1)

            nx = normal_maskd[:, 0] / 255.0 * 2 - 1
            ny = normal_maskd[:, 1] / 255.0 * 2 - 1
            nz = normal_maskd[:, 2] / 255.0 * 2 - 1
            normals = np.concatenate((nx, ny, nz), axis=1)

            dellist = [j for j in range(0, len(self.cld[obj_id]))]
            if self.refine:
                dellist = random.sample(dellist, len(self.cld[obj_id]) - self.num_pt_mesh_large)
            else:
                dellist = random.sample(dellist, len(self.cld[obj_id]) - self.num_pt_mesh_small)

            model_points = np.delete(self.cld[obj_id], dellist, axis=0)
            model_points = model_points / 1000

            target = np.dot(model_points, target_r)
            target = np.add(target, target_t)

            # displayPoint(cloud, target, index)

            # patch_occ = patch_label[rmin:rmax, cmin:cmax]
            # num_patch = np.max(patch_occ)
            # num_list = []
            # for n in range(1, num_patch + 1):
            #     num = str(patch_occ.tolist()).count(str(n))
            #     num_list.append(num)
            #
            # num_list_new = []
            # patch_id_list = []
            # for m in num_list:
            #     if m > 100:
            #         num_list_new.append(m)
            #         patch_id_list.append(num_list.index(m) + 1)

            im_id = 0
            scene_id = 0

        # if self.mode == 'test':
        else:
            data_dir = self.list[index][:23]
            dir_num = self.list[index][16:22]
            data_num = self.list[index][23:29]
            obj_order = self.list[index][30:]
            idx = int(obj_order)
            im_id = int(data_num)
            scene_id = int(dir_num)

            info = self.info_list[int(dir_num) - 1]
            gt = self.gt_list[int(dir_num) - 1]
            cam = self.cam_list[int(dir_num) - 1]
            obj_num = len(gt[str(int(data_num))])
            obj_id = gt[str(int(data_num))][idx]['obj_id']

            model_info = self.model_info[str(obj_id)]

            depth = np.array(Image.open('{0}/{1}/{2}/{3}.png'.format(self.root, data_dir, 'depth', data_num)))
            label = np.array(Image.open(
                '{0}/{1}/{2}/{3}_{4}.png'.format(self.root, data_dir, 'mask_visib', data_num, str(idx).zfill(6))))
            patch_label = np.array(Image.open(
                '{0}/{1}/{2}/{3}_{4}.png'.format(self.root, data_dir, 'segmentation', data_num, str(idx).zfill(6))))
            img = Image.open('{0}/{1}/{2}/{3}.png'.format(self.root, data_dir, 'rgb', data_num.zfill(6)))
            normal = np.array(Image.open(
                '{0}/{1}/{2}/{3}.png'.format(self.root, data_dir, 'normal', data_num, str(idx).zfill(6))))
            choose_file = '{0}/{1}/{2}/{3}_{4}_choose.list'.format(self.root, data_dir, 'segmentation', data_num, str(idx).zfill(6))
            with open(choose_file) as f:
                data = f.readlines()
            choose_ls = []
            stable_ls = []
            if len(data) > 1:
                for ids in data:
                    choose_id = ids[:-1].split(',')[:-1]
                    stable = float(ids[:-1].split(',')[-1])
                    choose_ls.append([int(x) for x in choose_id])
                    stable_ls.append(stable)
            else:
                if data[0] != '0':
                    choose_id = data[0].split(',')[:-1]
                    stable = float(data[0].split(',')[-1])
                    choose_ls.append([int(x) for x in choose_id])
                    stable_ls.append(stable)
                else:
                    stable_ls.append(0)
            depth_scale = cam[str(int(data_num))]['depth_scale']
            cam_k = cam[str(int(data_num))]['cam_K']
            cam_k = np.array(cam_k).reshape(3, 3)
            cam_cx = cam_k[0, 2]
            cam_cy = cam_k[1, 2]
            cam_fx = cam_k[0, 0]
            cam_fy = cam_k[1, 1]

            obj_bb = info[str(int(data_num))][idx]['bbox_visib']
            cmin = obj_bb[0]
            cmax = cmin + obj_bb[2]
            rmin = obj_bb[1]
            rmax = rmin + obj_bb[3]

            img_masked = self.trans(img)[:, rmin:rmax, cmin:cmax]
            img_masked = torch.zeros(img_masked.shape)
            # img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, 255))
            mask = mask_label * mask_depth
            mask_patch = mask * patch_label
            mask_num = len(mask.flatten().nonzero()[0])

            target_r = gt[str(int(data_num))][idx]['cam_R_m2c']
            target_r = np.array(target_r).reshape(3, 3).T
            target_t = np.array(gt[str(int(data_num))][idx]['cam_t_m2c'])
            target_t = target_t / 1000
            rt = np.append(target_r.reshape(-1), target_t.reshape(-1)).reshape(1, 12)
            add = np.array([[0,0,0,1]])
            target_trans = np.append(target_r.T,target_t.reshape(3,1),axis=1)
            target_trans = np.append(target_trans,add,axis=0)

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > self.num_pt:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                if len(choose) == 0:
                    print(0)
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

            normal_maskd = normal[rmin:rmax, cmin:cmax].reshape(-1, 3)[choose][:, :, np.newaxis].astype(np.float32)
            patch_masked = patch_label[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

            pt3 = patch_masked
            pt2 = depth_masked*depth_scale / 1000
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2, pt3), axis=1)

            nx = normal_maskd[:, 0] / 255.0 * 2 - 1
            ny = normal_maskd[:, 1] / 255.0 * 2 - 1
            nz = normal_maskd[:, 2] / 255.0 * 2 - 1
            normals = np.concatenate((nx, ny, nz), axis=1)

            dellist = [j for j in range(0, len(self.cld[obj_id]))]
            if self.refine:
                dellist = random.sample(dellist, len(self.cld[obj_id]) - self.num_pt_mesh_large)
            else:
                dellist = random.sample(dellist, len(self.cld[obj_id]) - self.num_pt_mesh_small)

            model_points = np.delete(self.cld[obj_id], dellist, axis=0)
            model_points = model_points / 1000

            target = np.dot(model_points, target_r)
            target = np.add(target, target_t)

        # displayPoint(cloud,target,index)
        patches = pt3.astype(int)
        num_patch = np.max(patches)
        num_list = []
        patch_list = patches.reshape(-1).tolist()
        for n in range(1, num_patch + 1):  # ordered num of point in each tless(from patch_1 to patch_n)
            num = str(patch_list).count(str(n))
            num_list.append(num)

        num_list_new = []
        patch_id_list_all = []
        for m in num_list:  # select patchs that num of points > 100
            if m > 100:
                num_list_new.append(m)
                patch_id_list_all.append(num_list.index(m) + 1)

        # stable_id_order = []
        # stable_score_order = sorted(stable_ls)
        # for i in stable_score_order:
        #     stable_id_order.append(stable_score_order.index(i))

        choose_triplet = []
        all_list = [i for i in range(0, 2000)]
        if data[0] != '0':
            # stable_ls_new = []
            # for score in temp:
            #     if score >0.5:
            #         stable_ls_new.append(score)
            if len(stable_ls) > 3:
                stable_ls_new = heapq.nlargest(3, stable_ls)
            else:
                stable_ls_new = stable_ls
            triplet_ls = []
            for item in stable_ls_new:
                triplet_ls.append(choose_ls[stable_ls.index(item)])
            for tri in triplet_ls:
                patch_idx = []
                for m in range(cloud.shape[0]):
                    if cloud[m, 3] in tri:
                        patch_idx.append(m)
                if len(patch_idx) >= 300:
                    choose_triplet.append(np.array(patch_idx))
                else:
                    choose_triplet.append(np.array(all_list))
        else:
            choose_triplet.append(np.array(all_list))
        if not choose_triplet:
            choose_triplet.append(np.array(all_list))
        cloud = cloud[:, :-1]
        model_axis = np.array([0.0,0.0,1.0])

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               img_masked, \
               torch.from_numpy(rt.astype(np.float32)), \
               torch.from_numpy(target_trans.astype(np.float32)), \
               torch.LongTensor([obj_id - 1]), \
               choose_triplet,\
               torch.from_numpy(target.astype(np.float32)),\
               torch.from_numpy(model_points.astype(np.float32)),\
               torch.from_numpy(normals.astype(np.float32)),\
               model_info,\
               torch.from_numpy(model_axis.astype(np.float32)),\
               scene_id,\
               im_id

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_nosym_list(self):
        return self.nosym_obj_idx

    def get_rot_list(self):
        return self.rot_obj_idx

    def get_ref_list(self):
        return self.ref_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


# border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400]

img_width = 400
img_length = 400


def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    # for tt in range(len(border_list)):
    #     if r_b > border_list[tt] and r_b < border_list[tt + 1]:
    #         r_b = border_list[tt + 1]
    #         break
    c_b = cmax - cmin
    # for tt in range(len(border_list)):
    #     if c_b > border_list[tt] and c_b < border_list[tt + 1]:
    #         c_b = border_list[tt + 1]
    #         break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def displayPoint(data,target,title):
    plt.rcParams['axes.unicode_minus'] = False
    while len(data[0]) > 50000:
        print("too much point")
        exit()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(title)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r', marker='.')
    ax.scatter3D(target[:, 0], target[:, 1], target[:, 2], c='b', marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.close()