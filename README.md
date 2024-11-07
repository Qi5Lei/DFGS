## CLIP-DFGS: A Hard Sample Mining Method for CLIP in Generalizable Person Re-Identification


![pipeline](DFS.png)






## Requirements

##### Our code is based on [fast-reid](https://github.com/JDAI-CV/fast-reid), [TransReid](https://github.com/damo-cv/TransReID) and [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID).

Thanks for their contributions.

*We will release the full code in the future.*

If you are using the above-mentioned framework, you just need to place our DFGS code in the sampler package and import it for use.

You can run the project by modifying the corresponding configuration (cfg) files and using the run.py script.




## Core Code

```python
class DepthFirstGraphSamplerText(Sampler):
    def __init__(self, data_source, batch_size, num_instances, model,pid2label,h_size=224, w_size=224,epoch=60,start=0,log=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(dict)
        self.start = start

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
        # index_dic = defaultdict(list)
        # for index, data in enumerate(self.data_source):
        #     index_dic[data[1]].append(index)
        # pids = list(index_dic.keys())

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
        range_arr = [i for i in range(self.start,self.start+10)]

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

```





## Experiment

Our method has achieved excellent results.

![experiments](experiments.png)


# Citation

```tex
@article{10.1145/3701036,
author = {Zhao, Huazhong and Qi, Lei and Geng, Xin},
title = {CLIP-DFGS: A Hard Sample Mining Method for CLIP in Generalizable Person Re-Identification},
year = {2024},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
}


```
