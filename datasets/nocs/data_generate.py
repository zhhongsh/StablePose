import numpy as np
mode = 1 # 0: train,1: test
if mode==0:
    scene_id = [i for i in range(1,8)]
    image_len = [805+1,623+1,573+1,597+1,540+1,646+1,588+1]
else:
    scene_id = [i for i in range(1, 7)]
    image_len = [389+ 1,629+ 1,524+ 1,499+ 1,515+ 1,569+ 1]

ls = []
for scene in scene_id:
    for image in range(image_len[scene-1]):
        ls.append('scene_'+str(scene)+'/'+str(image).zfill(4)+'\n')

with open('./dataset_config/nocs_real_test_ls.txt','w') as f:
    f.writelines(ls)

