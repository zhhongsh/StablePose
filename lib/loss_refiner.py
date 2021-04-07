from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
# from lib.knn.__init__ import KNearestNeighbor
from pytorch3d.ops import knn_points

def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    # knn = KNearestNeighbor(1)
    pred_r = pred_r.view(1, 1, -1)
    pred_t = pred_t.view(1, 1, -1)
    bs, num_p, _ = pred_r.size()
    num_input_points = len(points[0])

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
    
    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
    target_ori = target
    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t

    pred = torch.add(torch.bmm(model_points, base), pred_t)
    pred_ori = torch.mean(pred, dim=1)
    if idx[0].item() in sym_list:
        target_model = target_ori.view(bs, -1, 3)
        pred_model = pred_ori.view(bs, -1, 3)
        knn = knn_points(pred_model, target_model)
        dis = knn.dists.view(bs, -1)
    else:
        dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

    t = ori_t[0]
    points = points.view(1, num_input_points, 3)

    ori_base = ori_base[0].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_input_points, 1).contiguous().view(1, bs * num_input_points, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    r_pred = ori_base
    t_pred = t

    # print('------------> ', dis.item(), idx[0].item())

    return dis, new_points.detach(), new_target.detach(),r_pred.detach(),t_pred.detach()


class Loss_refine(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list


    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        return loss_calculation(pred_r, pred_t, target, model_points, idx, points, self.num_pt_mesh, self.sym_list)
