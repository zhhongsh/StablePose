# StablePose

This repository includes:  
*lib: the core Python library for networks and loss  
** lib/loss_*dataset.py: symmetrynet loss caculation for respective dataset 
** lib/network_*dataset.py: network architecture for the respective dataset 

* datasets: the dataloader and training/testing lists  
** datasets/tless/dataset.py: the training dataloader for tless dataset  
** datasets/tless/dataset_eval.py: the evaluation dataloader for tless dataset  
** datasets/tless/dataset_config/*.txt: training and testing splits for tless dataset

** datasets/shapenet/dataset.py: the training dataloader for shapnet dataset  
** datasets/shapenet/dataset_eval.py: the evaluation dataloader for shapnet dataset  
** datasets/shapenet/dataset_config/*.txt: training and testing splits for shapenet dataset 

** datasets/linemod/dataset.py: the training dataloader for linemod dataset  
** datasets/linemod/dataset_eval.py: the evaluation dataloader for linemod dataset  
** datasets/linemod/dataset_config/*.txt: training and testing splits for linemod dataset

** datasets/nocs/dataset.py: the training dataloader for nocs dataset  
** datasets/nocs/dataset_eval.py: the evaluation dataloader for nocs dataset  
** datasets/nocs/dataset_config/*.txt: training and testing splits for nocs dataset

To train StablePose on T-LESS dataset, run train_tless.py  
To train StablePose on ShapeNet dataset, run train_shapenet.py  
To train StablePose on NOCS-REAL275 dataset, run train_nocs.py  

To test StablePose on T-LESS dataset, run test_tless.py  
To test/evaluate StablePose on ShapeNet dataset, run test_shapenet.py  
To test/evaluate StablePose on NOCS-REAL275 dataset, run test_nocs.py  

To evaluate instace-level datasets: T-LESS and Linemod, use the code here https://github.com/thodan/bop_toolkit.  
