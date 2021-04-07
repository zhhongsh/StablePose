import torch.utils.data as data
import cv2
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import numpy.ma as ma
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
import heapq
# proj_dir = '/home/dell/yifeis/pose/pose_est_tless_3d/'
proj_dir = '/home/lthpc/yifeis/pose/pose_est_tless_3d/'

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train_100':
            self.path = proj_dir + 'datasets/shapenet/dataset_config/train_ls_100.txt'
        elif mode == 'train_500':
            self.path = proj_dir + 'datasets/shapenet/dataset_config/train_ls_500.txt'
        elif mode == 'test_100':
            self.path = proj_dir + 'datasets/shapenet/dataset_config/test_ls_100.txt'
        elif mode == 'test_500':
            self.path = proj_dir + 'datasets/shapenet/dataset_config/test_ls_500.txt'

        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self.list = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]

            self.list.append(input_line)
        input_file.close()

        self.bad_depth = []
        self.length = len(self.list)

        self.xmap = np.array([[j for i in range(960)] for j in range(540)])  # 480*640, xmap[i,:]==i
        self.ymap = np.array([[i for i in range(960)] for j in range(540)])  # 480*640, ymap[j,:]==j

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trans = transforms.ToTensor()
        self.num_pt_mesh = 500  # num_point_mesh
        self.refine = refine
        self.front_num = 2
        self.name_list = np.loadtxt(proj_dir + 'datasets/shapenet/'+'name_list.txt', dtype=str, delimiter='\n')
        self.class_id = {
            '000': '02691156',
            '001': '02747177',
            '002': '02773838',
            '003': '02808440',
            '004': '02818832',
            '005': '02828884',
            '006': '02876657',
            '007': '02924116',
            '008': '02942699',
            '009': '02946921',
            '010': '02958343',
            '011': '03001627',
            '012': '03211117',
            '013': '03261776',
            '014': '03467517',
            '015': '03513137',
            '016': '03593526',
            '017': '03636649',
            '018': '03642806',
            '019': '03991062',
            '020': '04225987',
            '021': '04256520',
            '022': '04379243'}
        # print(len(self.list))

    def __getitem__(self, index):
        while 1:
            if os.path.exists('{0}/{1}-rt.txt'.format(self.root, self.list[index])) and \
                    os.path.exists('{0}/{1}-depth-crop-occlusion.png'.format(self.root, self.list[index])) and \
                    os.path.exists('{0}/{1}-k-crop.txt'.format(self.root, self.list[index])) and \
                    os.path.exists('{0}/{1}-color-crop-occlusion.png'.format(self.root, self.list[index])) and\
                    os.path.exists('{0}/{1}-color-crop-segmentation.png'.format(self.root, self.list[index])):

                rt = np.loadtxt('{0}/{1}-rt.txt'.format(self.root, self.list[index]))
                check_rt = np.zeros((4, 4))
                if os.path.exists('{0}/{1}-rt-all.txt'.format(self.root, self.list[index])):
                    rt_all = np.loadtxt('{0}/{1}-rt-all.txt'.format(self.root, self.list[index]))
                    check_rt_all = np.zeros(rt_all.shape)
                    if (rt_all == check_rt_all).all():
                        index+=1
                        continue
                check_depth = 255 * np.ones((540, 960, 3))
                # depth_ = cv2.imread('{0}/{1}-depth-crop-occlusion.png'.format(self.root, self.list[index]))
                try:
                    depth_ = Image.open('{0}/{1}-depth-crop-occlusion.png'.format(self.root, self.list[index]))
                except (OSError, NameError):
                    index += 1
                    # self.bad_depth.append('{0}/{1}-depth-crop-occlusion.png'.format(self.root, self.list[index]))
                    continue
                depth_ = np.array(depth_)
                cam_ = np.loadtxt('{0}/{1}-k-crop.txt'.format(self.root, self.list[index]))

                if cam_.reshape(-1).shape[0] != 9:
                    print('{0}/{1}-k-crop.txt'.format(self.root, self.list[index]))

                input_file = self.list[index]
                # class_key = input_file[20:23]
                # input_id = int(input_file[20:23])
                # ins_num = int(input_file[24:28])
                class_key = input_file[:3]
                input_id = int(class_key)
                ins_num = int(input_file[4:8])
                cls_idx = input_id
                class_name = self.class_id[class_key]
                instance_ls = self.name_list[cls_idx][1:-1].split(",")
                ins_name = instance_ls[ins_num][2:-1]
                sym_dir = '/home/lthpc/yifeis/shapenet_mount/'
                sym_file = sym_dir + class_name + '/' + ins_name + '/' + 'model_sym.txt'
                if os.path.exists(sym_file) == False:
                    index += 1
                    continue
                model_s = np.loadtxt(sym_file)
                model_file = sym_dir + class_name + '/' + ins_name + '/' + 'model.obj'
                verts, faces, _ = load_obj(model_file)
                model_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
                model_points = sample_points_from_meshes(model_mesh, 500)
                syms = model_s[1:, :]
                check_ = np.zeros((4, 3))
                check_sym = (syms != check_)
                nozero = np.nonzero(check_sym)
                row_id = nozero[0]
                if (rt == check_rt).all() or (depth_ == check_depth).all() :
                    index += 1
                elif (row_id.shape[0] != 0)  and (cam_.reshape(-1).shape[0] == 9):
                    break
                else:
                    index += 1
            else:
                index += 1
        choose_file = '{0}/{1}-choose.list'.format(self.root, self.list[index])
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
            print('-choose list file not exist')
            stable_ls.append(0)
            choose_ls = []
            data = ['0']
        input_file = self.list[index]
        class_key = input_file[:3]
        input_id = int(class_key)
        ins_num = int(input_file[4:8])
        img_ = cv2.imread('{0}/{1}-color-crop-occlusion.png'.format(self.root, self.list[index]))  # 540*960
        img = img_
        depth_ = cv2.imread('{0}/{1}-depth-crop-occlusion.png'.format(self.root, self.list[index]))
        depth = depth_[:, :, 0]
        seg_ = cv2.imread('{0}/{1}-color-crop-segmentation.png'.format(self.root, self.list[index]))
        seg = seg_[:, :, 0]
        cam = np.loadtxt('{0}/{1}-k-crop.txt'.format(self.root, self.list[index]))
        cam_cx = cam[0, 2]
        cam_cy = cam[1, 2]
        cam_fx = cam[0, 0]
        cam_fy = cam[1, 1]

        idx = input_id
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 255))
        mask_seg = ma.getmaskarray(ma.masked_not_equal(seg, 0))
        patch_label = mask_seg*mask_depth
        mask_label = mask_depth
        mask = mask_depth
        mask_real = len(mask.nonzero()[0])

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(img[:, :, :3], (2, 0, 1))
        img_masked = img[:, rmin:rmax, cmin:cmax]

        try:
            target_trans = rt_all.reshape(-1,4,4)
        except (OSError, NameError):
            target_trans = rt

        target_mode = 0
        center = model_s[0, :]
        check_line = np.zeros((1, 3))

        if (syms[-1] != check_line).any():
            target_mode = 1
        else:
            if target_trans.shape[0]==4:
                target_mode=2
            else:
                target_mode=0

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        patch_masked = patch_label[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)get masked depth
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)
        choose = np.array([choose])  # (1,1000)

        cam_scale = 100  # cam_scale = 10000
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)  # (1000,3)

        target_r = rt[:-1, :-1]
        target_t = rt[:-1, 3]
        target_rt = np.append(target_r.T, target_t).reshape(1, 12)

        patches = patch_masked.astype(int)
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
                    if patches[m] in tri:
                        patch_idx.append(m)
                if len(patch_idx) >= 300:
                    choose_triplet.append(np.array(patch_idx))
                else:
                    choose_triplet.append(np.array(all_list))
        else:
            choose_triplet.append(np.array(all_list))
        if not choose_triplet:
            choose_triplet.append(np.array(all_list))

        target = np.dot(model_points, target_r.T)
        target_pt = np.add(target, target_t)

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.from_numpy(target_rt.astype(np.float32)),\
               choose_triplet, \
               torch.from_numpy(target_pt.astype(np.float32)), \
               model_points[0],\
               torch.LongTensor([target_mode]),\
               torch.from_numpy(target_trans.astype(np.float32)),\
               self.list[index]


    def __len__(self):
        return self.length

    def get_num_points_mesh(self):
        return self.num_pt_mesh


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560]
img_width = 540
img_length = 960


def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list) - 1):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list) - 1):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
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
