### place our DFGS code in the sampler package and import it for use. ###

import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
### import this class ###
from ..common import CommDataset
from collections import deque


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, domain = zip(*batch)
    domains = torch.tensor(domain, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, domains


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


### PK sampler ###
class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, data in enumerate(self.data_source):
            self.index_dic[data[1]].append(index)
        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        pids = self.pids
        for pid in pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        print(len(final_idxs))
        return iter(final_idxs)

    def __len__(self):
        return self.length


### DFGS on image encoder ###
class DepthFirstGraphSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances, model,pid2label,h_size=224, w_size=224,epoch=60):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(dict)

        for index, data in enumerate(self.data_source):
            if data[2] in self.index_dic[data[1]].keys():
                self.index_dic[data[1]][data[2]].append(index)
            else:
                self.index_dic[data[1]][data[2]] = [index]
        self.pids = list(self.index_dic.keys())
        self.pidlabel = torch.tensor([pid2label[pid] for pid in self.pids])
        self.get_feat_flag = True
        self.dist_mat = None
        self.length = 0
        self.epoch = epoch
        self.num_epoch = 0
        for pid in self.pids:
            num = sum([len(self.index_dic[pid][key]) for key in self.index_dic[pid].keys()])
            self.length += num

        self.eval_transforms = T.Compose([
            T.Resize((h_size, w_size)),
            T.ToTensor(),
            nn.InstanceNorm2d(3),
        ])

    # sort for camera
    def sort_dic_cam(self, s):
        ks = list(s.keys())
        len_k = np.array([len(s[k]) for k in s.keys()])
        ix = len_k.argsort()[::-1]
        return {ks[i]: s[ks[i]] for i in ix}

    def __iter__(self):
        self.num_epoch+=1
        batch_idxs_dict = defaultdict(list)
        pids = copy.deepcopy(self.pids)
        random.shuffle(pids)
        print("Graph Structure Updating...")
        for pid in pids:
            dic_tmp = copy.deepcopy(self.index_dic[pid])
            cids = list(dic_tmp.keys())
            for cid in cids:
                random.shuffle(dic_tmp[cid])
            idxs = []
            while cids:
                num = 0
                dic_tmp = self.sort_dic_cam(dic_tmp)
                for cid in cids:
                    num += 1
                    idxs.append(dic_tmp[cid].pop())
                    if len(dic_tmp[cid]) == 0:
                        cids.remove(cid)
                    if num == self.num_instances:
                        break
            if len(idxs) <= 1:
                continue
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        avai_pids = copy.deepcopy(pids)
        final_idxs = []

        model = copy.deepcopy(self.model).cuda().eval()
        index_dic = defaultdict(list)
        for index, data in enumerate(self.data_source):
            index_dic[data[1]].append(index)
        pids = list(index_dic.keys())
        inex_dic = {k: index_dic[k][random.randint(0, len(index_dic[k]) - 1)] for k in pids}
        choice_set = CommDataset([self.data_source[i] for i in list(inex_dic.values())], self.eval_transforms,
                                 relabel=True)
        choice_loader = DataLoader(
            choice_set, batch_size=128, shuffle=False, num_workers=8,
            collate_fn=val_collate_fn
        )

        feats = torch.tensor([]).cuda()
        for i, (img, pid, domain) in enumerate(choice_loader):
            with torch.no_grad():
                img = img.cuda()
                feat = model(img)
                feats = torch.cat((feats, feat), dim=0)

        dist_mat = euclidean_dist(feats, feats)


        for i in range(len(dist_mat)):
            dist_mat[i][i] = float("inf")

        self.dist_mat = dist_mat
        self.get_feat_flag = False

        graph = {}
        for i, feat in enumerate(self.dist_mat):
            graph[pids[i]] = []
            loc = torch.argsort(feat)
            for j in range(15):
                locj = int(loc[j].cpu())
                graph[pids[i]].append(pids[locj])
        print("Graph Structure Update Completed!")
        
        
        
        print("Building Iteration...")
        batch_idxs = []
        is_pid = set()

        stack = deque([random.choice(list(avai_pids))])
        i = len(avai_pids)
        while stack and i:
            k = stack.pop()
            if k not in avai_pids:
                continue
            # If multiple consecutive PIDs already exist in the current batch, exit the loop.
            if k in is_pid:
                stack.appendleft(k)
                i -= 1
                continue
            # Reset the count of 'i' whenever a new PID enters the batch.
            i = len(avai_pids)
            batch_idxs.extend(batch_idxs_dict[k].pop(0))
            if len(batch_idxs_dict[k]) == 0:
                avai_pids.remove(k)
            # Mark the PID 'k' already exists in this batch.
            is_pid.add(k)
            # At the end of a batch, start counting anew for the next round.
            if len(batch_idxs) == self.batch_size:
                final_idxs.extend(batch_idxs)
                batch_idxs = []
                is_pid = set()
            # At `the completion of a batch, start a new round.
            avai_k = list(graph[k])
            # Depth-first
            for v in avai_k[::-1]:
                if v in avai_pids:
                    stack.append(v)
        torch.cuda.empty_cache()
        print("Completion of Iteration!")
        print(len(final_idxs))
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


### DFGS on text encoder###
class DepthFirstGraphSamplerText(Sampler):
    def __init__(self, data_source, batch_size, num_instances, model,pid2label,h_size=224, w_size=224,epoch=60,log=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(dict)

        for index, data in enumerate(self.data_source):
            if data[2] in self.index_dic[data[1]].keys():
                self.index_dic[data[1]][data[2]].append(index)
            else:
                self.index_dic[data[1]][data[2]] = [index]
        self.pids = list(self.index_dic.keys())
        self.pidlabel = torch.tensor([pid2label[pid] for pid in self.pids])
        self.length = 0
        self.log = log
        self.get_feat = True
        self.dist_mat = None

        for pid in self.pids:
            num = sum([len(self.index_dic[pid][key]) for key in self.index_dic[pid].keys()])
            self.length += num

        self.eval_transforms = T.Compose([
            T.Resize((h_size, w_size)),
            T.ToTensor(),
            nn.InstanceNorm2d(3),
        ])

    # sort for camera
    def sort_dic_cam(self, s):
        ks = list(s.keys())
        len_k = np.array([len(s[k]) for k in s.keys()])
        ix = len_k.argsort()[::-1]
        return {ks[i]: s[ks[i]] for i in ix}

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        pids = copy.deepcopy(self.pids)
        random.shuffle(pids)
        self.log.info("Graph Structure Updating...")
        for pid in pids:
            dic_tmp = copy.deepcopy(self.index_dic[pid])
            cids = list(dic_tmp.keys())
            for cid in cids:
                random.shuffle(dic_tmp[cid])
            idxs = []
            while cids:
                num = 0
                dic_tmp = self.sort_dic_cam(dic_tmp)
                for cid in cids:
                    num += 1
                    idxs.append(dic_tmp[cid].pop())
                    if len(dic_tmp[cid]) == 0:
                        cids.remove(cid)
                    if num == self.num_instances:
                        break
            if len(idxs) <= 1:
                continue
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        avai_pids = copy.deepcopy(pids)
        final_idxs = []

        model = copy.deepcopy(self.model).cuda().eval()

        i_ter = len(pids) // self.batch_size
        feats = torch.tensor([]).cuda()
        if self.get_feat:
            self.get_feat = False
            for i in range(i_ter):
                if i+1 != i_ter:
                    target_label = self.pidlabel[i * self.batch_size:(i + 1) * self.batch_size]
                else:
                    target_label = self.pidlabel[i * self.batch_size:len(pids)]
                with torch.no_grad():
                    feat = model(label=target_label, get_text=True, domain=None, cam_label=None, getdomain=False,getcls=False)
                    feats = torch.cat((feats, feat), dim=0)
            dist_mat = euclidean_dist(feats, feats)
            for i in range(len(dist_mat)):
                dist_mat[i][i] = float("inf")

            self.dist_mat = dist_mat

        graph = {}
        range_arr = [i for i in range(2,12)]

        for i, feat in enumerate(self.dist_mat):
            graph[pids[i]] = []
            loc = torch.argsort(feat)
            random.shuffle(range_arr)
            for j in range_arr:
                locj = int(loc[j].cpu())
                graph[pids[i]].append(pids[locj])
        self.log.info("Graph Structure Update Completed!")



        self.log.info("Building Iteration...")
        batch_idxs = []
        is_pid = set()

        stack = deque([random.choice(list(avai_pids))])
        i = len(avai_pids)
        while stack and i:
            k = stack.pop()
            if k not in avai_pids:
                continue
            # If multiple consecutive PIDs already exist in the current batch, exit the loop.
            if k in is_pid:
                stack.appendleft(k)
                i -= 1
                continue
            # Reset the count of 'i' whenever a new PID enters the batch.
            i = len(avai_pids)
            batch_idxs.extend(batch_idxs_dict[k].pop(0))
            if len(batch_idxs_dict[k]) == 0:
                avai_pids.remove(k)
            # Mark the PID 'k' already exists in this batch.
            is_pid.add(k)
            # At the end of a batch, start counting anew for the next round.
            if len(batch_idxs) == self.batch_size:
                final_idxs.extend(batch_idxs)
                batch_idxs = []
                is_pid = set()
            # At `the completion of a batch, start a new round.
            avai_k = list(graph[k])
            # Depth-first
            for v in avai_k[::-1]:
                if v in avai_pids:
                    stack.append(v)
        torch.cuda.empty_cache()
        self.log.info("Completion of Iteration!"+' '+str(len(final_idxs)))
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


    def __len__(self):
        return self.length
