# MIT License

# Copyright (c) 2022 Yaqiang Sun

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel
import cv2
from qpth.qp import QPFunction
from tqdm import trange
from model.utils import count_acc
from model.transformers.attention import MultiHeadAttention,ScaledDotProductAttention

def emd_inference_qpth(distance_matrix, weight1, weight2, form='QP', l2_strength=0.0001):
    """
    to use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
    :param distance_matrix: nbatch * element_number * element_number
    :param weight1: nbatch  * weight_number
    :param weight2: nbatch  * weight_number
    :return:
    emd distance: nbatch*1
    flow : nbatch * weight_number *weight_number

    """

    weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(1).unsqueeze(1)
    weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(1).unsqueeze(1)

    nbatch = distance_matrix.shape[0]
    nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
    nelement_weight1 = weight1.shape[1]
    nelement_weight2 = weight2.shape[1]

    Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()

    if form == 'QP':
        # version: QTQ
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().cuda() + 1e-4 * torch.eye(
            nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
        p = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
    elif form == 'L2':
        # version: regularizer
        Q = (l2_strength * torch.eye(nelement_distmatrix).double()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        p = distance_matrix.view(nbatch, nelement_distmatrix).double()
    else:
        raise ValueError('Unkown form')

    h_1 = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
    h_2 = torch.cat([weight1, weight2], 1).double()
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
    G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().cuda()
    # sum_j(xij) = si
    for i in range(nelement_weight1):
        G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
    # sum_i(xij) = dj
    for j in range(nelement_weight2):
        G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
    #xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
    G = torch.cat((G_1, G_2), 1)
    A = torch.ones(nbatch, 1, nelement_distmatrix).double().cuda()
    b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    emd_score = torch.sum((1 - Q_1).squeeze() * flow, 1)
    return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow

def emd_inference_opencv_test(distance_matrix,weight1,weight2):
    distance_list = []
    flow_list = []

    for i in range (distance_matrix.shape[0]):
        cost,flow=emd_inference_opencv(distance_matrix[i],weight1[i],weight2[i])
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    emd_distance = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()

    return emd_distance,flow

def multi_emd(similarity_dist_map,num_query,num_proto,weight_1,weight_2):
    flows = np.zeros_like(similarity_dist_map)
    for i in range(num_query):
        for j in range(num_proto):
            cost_matrix = similarity_dist_map[i, j, :, :]
            weight1 = weight_1[i, j, :]
            weight2 = weight_2[j, i, :]
            cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
            flows[i,j] = flow
    return flows

class SeparateEMD(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        elif args.backbone_class == 'Transformer':
            hdim = 640
        elif args.backbone_class == 'Swin':
            hdim = 768
        else:
            raise ValueError('')
        # no_avg_pool
        self.hdim = hdim
        self.w = 5
        self.h = 5
        if args.deepemd == "sampling":
            self.w = 2*2
            self.h = 5
       
        self.out_slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.attention = ScaledDotProductAttention(temperature=np.power(hdim, 0.5),attn_dropout=0)

        
        self.learnable_scale_attention = torch.nn.Parameter(torch.FloatTensor(1).fill_(1.0),requires_grad=True)
        

        
    def _forward(self, instance_embs, support_idx, query_idx):
        
        # no_avg_pool
        if self.args.no_avg_pool:
            
            
            instance_embs = instance_embs.view(instance_embs.size(0),self.hdim,5,5,-1).contiguous()
            instance_embs_avg = instance_embs.mean(dim=3).unsqueeze(3)
            instance_embs = torch.cat([instance_embs,instance_embs_avg],3)
            instance_embs = instance_embs.mean(dim=2)
            instance_embs = instance_embs.view(instance_embs.size(0),self.hdim,2,3,-1)
            instance_embs = instance_embs.mean(dim=3)
            instance_embs = instance_embs.view(instance_embs.size(0),self.hdim,self.w,self.h)
            
        # organize support/query datasupport_idx.shape
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (self.hdim,self.w,self.h)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + (self.hdim,self.w,self.h)))
        
        proto_dist = None


        if self.args.shot>1:
           
            query = query.view(-1,self.hdim,self.w,self.h)
           
            logits,proto_dist = self.all_emd_forward_5shot(support, query)


        else:
            proto = support.mean(dim=1) # Ntask x NK x d
            query = query.view(-1,self.hdim,self.w,self.h)
            logits = self.attention_emd_forward_1shot(proto, query)

        
        # for regularization
        if self.training:          
            logits_reg = None
            return logits, logits_reg,proto_dist            
        else:
            return logits, proto_dist


    def get_weight_vector(self, A, B):

        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination
    
    def attention_emd_forward_1shot(self, proto, query):
        proto = proto.squeeze(0)

        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_attention_similiarity_map(proto, query)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance_faster(similarity_map,weight_1, weight_2,solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
        return logits

    def all_emd_forward_5shot(self, support, query):
        support = support.to('cuda:3')
        query = query.to('cuda:3')  
        
        support = support.view(self.args.shot,self.args.way,self.hdim,self.w,self.h)
        support = support.permute(1,2,3,4,0).contiguous()
        proto = support.view(self.args.way,self.hdim,self.w,self.h*self.args.shot)
        query = query*self.args.shot
       
        # proto in class attn
        proto_attn = proto.view(self.args.way,self.hdim,-1)
        proto_attn = proto_attn.permute(0,2,1)
        proto_attn, _, _ =self.attention(proto_attn,proto_attn,proto_attn)
        proto_attn = proto_attn.permute(0,2,1).contiguous()
        proto_attn = proto_attn.view(proto.size())
        proto_attn = F.adaptive_avg_pool2d(proto_attn, [1, 1])
        
        # proto mean
        proto_mean = F.adaptive_avg_pool2d(proto, [1, 1])

        in_proto_dist = None
        
        proto = proto + proto_attn + (self.learnable_scale_attention*(proto_mean.to(self.learnable_scale_attention.device))).to(proto.device)

        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(proto, query)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance_faster(similarity_map,weight_1, weight_2,solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
        return logits,in_proto_dist

    def get_emd_distance_faster(self, similarity_map, weight_1, weight_2, solver='opencv'):
        
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver
            weight_1 = F.relu(weight_1) + 1e-5
            weight_2 = F.relu(weight_2) + 1e-5

            weight_1 = weight_1 * ((weight_1.size()[-1]/weight_1.sum(-1)).unsqueeze(-1))
            weight_2 = weight_2 * ((weight_2.size()[-1]/weight_2.sum(-1)).unsqueeze(-1))
            similarity_dist_map = (1 - similarity_map).detach().cpu().numpy()
            weight_1 = weight_1.unsqueeze(-1).detach().cpu().numpy()
            weight_2 = weight_2.unsqueeze(-1).detach().cpu().numpy()
            
            
            flows = multi_emd(similarity_dist_map,num_query,num_proto,weight_1,weight_2)
            similarity_map =(similarity_map.cuda())*(torch.from_numpy(flows).cuda())

            temperature=(self.args.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            return logitis

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

            temperature=(self.args.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            return logitis

        elif solver == 'qpth':
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form=self.args.form, l2_strength=self.args.l2_strength)

            logitis=(flows*similarity_map).view(num_query, num_proto,flows.shape[-2],flows.shape[-1])
            temperature = (self.args.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) *  temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x


    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

    def get_attention_similiarity_map(self, proto, query):
            way = proto.shape[0]
            num_query = query.shape[0]
            query = query.view(query.shape[0], query.shape[1], -1)
            proto = proto.view(proto.shape[0], proto.shape[1], -1)

            proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
            query = query.unsqueeze(1).repeat([1, way, 1, 1])
            proto = proto.permute(0, 1, 3, 2)
            query = query.permute(0, 1, 3, 2)
            feature_size = proto.shape[-2]

            if self.args.metric == 'cosine':
                proto = proto.unsqueeze(-3)
                query = query.unsqueeze(-2)
                query = query.repeat(1, 1, 1, feature_size, 1)
                similarity_map = F.cosine_similarity(proto, query, dim=-1)
            if self.args.metric == 'l2':
                proto = proto.unsqueeze(-3)
                query = query.unsqueeze(-2)
                query = query.repeat(1, 1, 1, feature_size, 1)
                similarity_map = (proto - query).pow(2).sum(-1)
                similarity_map = 1 - similarity_map

            return similarity_map
