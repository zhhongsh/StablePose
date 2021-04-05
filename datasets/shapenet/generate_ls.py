import numpy as np
import random
import os
import numpy.ma as ma
import cv2
from PIL import  Image
seg = cv2.imread('/home/dell/yifeis/pose_estimation/render/render_dy/render_wt_pt_proj/data/syn_images_20views1/000/0000/15-color-crop-segmentation.png')
patch_label = seg[:,:,0]
patch_mask = ma.getmaskarray(ma.masked_not_equal(patch_label, 0))
print()

def generate_ls():
    much = [0,3,5,7,10,11,12,14,17,19,21,22] #>600,12
    small = [1,4,6,8,9,15,16,18,20]#>100,9

    datadir = '/home/dell/yifeis/pose_estimation/render/render_dy/render_wt_pt_proj/data/syn_images_20views1/'
    all_ls = []
    train_list_all = []
    test_list_all = []
    train_list2 = []
    test_list1 = []
    train_list1 = []
    test_list2 = []
    for i in much:
        for k in range(500):
            for j in range(0,20):
                cat = str(i).zfill(3)
                ins = str(k).zfill(4)
                num = str(j)
                path = datadir+ cat+'/'+ins+'/'+num
                rgb = path+'-color-crop-occlusion.png'
                depth = path+'-depth-crop-occlusion.png'
                rt = path+'-rt.txt'
                cam = path + '-k-crop.txt'
                if os.path.exists(rgb) and os.path.exists(depth) and os.path.exists(rt) and os.path.exists(cam):
                    depth_ = np.array(Image.open(depth))
                    check_rt = np.zeros((4, 4))
                    pose = np.loadtxt(rt)
                    check_depth = 255 * np.ones((540, 960, 3))
                    if (pose == check_rt).all() or (depth_ == check_depth).all():
                        continue
                    else:
                        if k<400:
                            train_list1.append(cat+'/'+ins+'/'+num+'\n')
                            train_list_all.append(cat + '/' + ins + '/' + num + '\n')
                        else:
                            if j%4 and j!=0:
                                test_list1.append(cat + '/' + ins + '/' + num + '\n')
                                test_list_all.append(cat + '/' + ins + '/' + num + '\n')
                        print('writing:', cat+'/'+ins+'/'+num)

    for i in small:
        for k in range(100):
            for j in range(0,20):
                cat = str(i).zfill(3)
                ins = str(k).zfill(4)
                num = str(j)
                path = datadir+ cat+'/'+ins+'/'+num
                rgb = path+'-color-crop-occlusion.png'
                depth = path+'-depth-crop-occlusion.png'
                rt = path+'-rt.txt'
                cam = path + '-k-crop.txt'
                if os.path.exists(rgb) and os.path.exists(depth) and os.path.exists(rt) and os.path.exists(cam):
                    depth_ = np.array(Image.open(depth))
                    check_rt = np.zeros((4, 4))
                    pose = np.loadtxt(rt)
                    check_depth = 255 * np.ones((540, 960, 3))
                    if (pose == check_rt).all() or (depth_ == check_depth).all() :
                        continue
                    else:
                        if k<80:
                            train_list2.append(cat+'/'+ins+'/'+num+'\n')
                            train_list_all.append(cat + '/' + ins + '/' + num + '\n')
                        else:
                            if j%4 and j!=0:
                                test_list2.append(cat + '/' + ins + '/' + num + '\n')
                                test_list_all.append(cat + '/' + ins + '/' + num + '\n')
                        print('writing:', cat+'/'+ins+'/'+num)
    #
    # test_ids = random.sample(range(len(all_ls)), int(len(all_ls)/5))
    # all_ids = [m for m in range(len(all_ls))]
    # train_ids = set(all_ids)-set(test_ids)
    #
    # train_ls = [all_ls[n] for n in train_ids]
    # test_ls = [all_ls[q] for q in test_ids]
    #
    # print('length of test:', len(test_ls))
    # print('length of train:', len(train_ls))

    with open('./train_ls2_all.txt','w') as f:
        f.writelines(train_list_all)
    with open('./test_ls2_all.txt','w') as f:
        f.writelines(test_list_all)
    with open('./train_ls2_500.txt','w') as f:
        f.writelines(train_list1)
    with open('./test_ls2_500.txt','w') as f:
        f.writelines(test_list1)
    with open('./train_ls2_100.txt', 'w') as f:
        f.writelines(train_list2)
    with open('./test_ls2_100.txt', 'w') as f:
        f.writelines(test_list2)
    print()


