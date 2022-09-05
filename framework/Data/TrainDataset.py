from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import DataLoader
def get_train_dataloader(train_dataset):
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )
    return train_dataloader

class TrainDataset(Dataset):
    def __init__(self,in_path=None):
        self.in_path = in_path
        self.tri_file = in_path + "train2id.txt"
        self.ent_file = in_path + "entity2id.txt"
        self.rel_file = in_path + "relation2id.txt"
        self.head, self.tail, self.rel = self.__read_dataset()
        self.label_smooth = 0.1
        self.__count_htr()
        self.__tri_count()
        self.__constract_dataset()

    def __len__(self):
        return len(self.t_of_hr)
    def __getitem__(self, idx):

        e_in = 0 if int(self.Head[idx]) not in self.e_in_num else self.e_in_num[int(self.Head[idx])]
        e_out = self.e_out_num[int(self.Head[idx])]
        # e_in = torch.LongTensor(e_in)
        # e_out = torch.LongTensor(e_out)

        return self.Head[idx],  self.Rel[idx], self.Tail[idx],torch.tensor(e_in), torch.tensor(e_out)

    def __constract_dataset(self):
        Head = []
        Rel = []
        Tail = []
        for hr, tlist in self.t_of_hr.items():
            h, r = hr
            Head.append(h)
            Rel.append(r)
            # a = self.__onehot(tlist)
            tail_smooth = (1.0 - self.label_smooth) * self.__onehot(tlist) + (1.0 / self.ent_total)
            Tail.append(tail_smooth)

        self.Head = torch.LongTensor(Head)
        self.Rel = torch.LongTensor(Rel)
        self.Tail = torch.from_numpy(np.array(Tail)).float()
        # print(self.Tail)

    def __onehot(self, tlist):
        t_onehot = np.zeros([self.ent_total])
        # print(t_onehot)
        for t in tlist:
            t_onehot[t] = 1
        return t_onehot

    def __tri_count(self):
        e_out_num = {}
        e_in_num = {}
        for hr, tlist in self.t_of_hr.items():
            h, r = hr
            if h not in e_out_num:
                e_out_num[h] = len(tlist)
            else:
                e_out_num[h] += len(tlist)
        for tr, hlist in self.h_of_tr.items():
            t, r = tr
            if t not in e_in_num:
                e_in_num[t] = len(hlist)
            else:
                e_in_num[t] += len(hlist)
        self.e_out_num = e_out_num
        self.e_in_num = e_in_num

    def __read_dataset(self):
        with open(self.ent_file, "r") as f:
            self.ent_total = (int)(f.readline())
        with open(self.rel_file, "r") as f:
            self.rel_total = (int)(f.readline())

        head = []
        tail = []
        rel = []
# TODO 此处添加反关系
        with open(self.tri_file, "r") as f:
            triples_total = (int)(f.readline())
            for index in range(triples_total):
                h, t, r = f.readline().strip().split()

                head.append((int)(h))
                rel.append((int)(r))
                tail.append((int)(t))

                head.append((int)(t))
                rel.append((int)(r)+self.rel_total)
                tail.append((int)(h))
        return head,tail,rel

    def __count_htr(self):

        self.h_of_tr = {}
        self.t_of_hr = {}
        self.r_of_ht = {}
        self.h_of_r = {}
        self.t_of_r = {}
        # r出现的次数
        self.freqRel = {}

        triples = zip(self.head, self.tail, self.rel)
        for h, t, r in triples:
            if (h, r) not in self.t_of_hr:
                self.t_of_hr[(h, r)] = []
            self.t_of_hr[(h, r)].append(t)
            if (t, r) not in self.h_of_tr:
                self.h_of_tr[(t, r)] = []
            self.h_of_tr[(t, r)].append(h)
            if (h, t) not in self.r_of_ht:
                self.r_of_ht[(h, t)] = []
            self.r_of_ht[(h, t)].append(r)
            if r not in self.freqRel:
                self.freqRel[r] = 0
                self.h_of_r[r] = {}
                self.t_of_r[r] = {}
            self.freqRel[r] += 1.0
            #  数据的结构：{r{h:1}}
            self.h_of_r[r][h] = 1
            self.t_of_r[r][t] = 1
        # print("h_of_tr",self.h_of_tr)
        for t, r in self.h_of_tr:
            self.h_of_tr[(t, r)] = list(set(self.h_of_tr[(t, r)]))
        for h, r in self.t_of_hr:
            self.t_of_hr[(h, r)] = list(set(self.t_of_hr[(h, r)]))
        for h, t in self.r_of_ht:
            self.r_of_ht[(h, t)] = list(set(self.r_of_ht[(h, t)]))
    @ staticmethod
    def collate_fn(data):
        head = torch.stack([_[0] for _ in data], dim=0)
        rel = torch.stack([_[1] for _ in data], dim=0)
        tail = torch.stack([_[2] for _ in data], dim=0)
        e_in_num = torch.stack([_[3] for _ in data], dim=0)
        e_out_num = torch.stack([_[4] for _ in data], dim=0)

        return head, rel, tail, e_in_num, e_out_num

    def get_ent_tot(self):
        return self.ent_total
    def get_rel_tot(self):
        return self.rel_total




