import json
import os
model_info_file = open('./test_targets_bop19.json','r', encoding='utf-8')
test_ls = json.load(model_info_file)
save_ls = []
# for itm in test_ls:
#     # scene = str(itm["scene_id"])
#     # im = str(itm["im_id"])
#     # int_count = str(itm["inst_count"]-1).zfill(2)
for scene in range(1,21):
    dir = '/home/dell/yifeis/pose/bop_datasets/test_primesense/'+str(scene).zfill(6)+'/'+'rgb/'
    # im_ls=[]
    # for root, dirs, files in os.walk(dir):
    #     for file in files:
    #         im_ls.append(file)
    # print(im_ls)
    files = os.walk(dir)
    for image in files:
        im_ls = image[2]
        for i in range(len(im_ls)):
            im = int(im_ls[i][:-4])
            data_path = '/home/dell/yifeis/pose/bop_datasets/test_primesense/'+ str(scene).zfill(6) +'/'+'scene_gt.json'
            scene_gt = json.load(open(data_path))
            count_all = len(scene_gt[str(im)])
            for j in range(count_all):
                save_ls.append('test_primesense/'+str(scene).zfill(6)+'/' + str(im).zfill(6) + '/' + str(j)+'\n')

with open('./target_test.txt','w') as f:
    f.writelines(save_ls)

    print()
