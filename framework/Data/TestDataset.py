from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
def get_test_valid_dataset(mode='test',data='FB15K237'):
    tail_dataset = TestDataset(in_path="../benchmarks/"+data+'/',
                dataset=mode, mode='tail')
    head_dataset = TestDataset(in_path="../benchmarks/"+data+'/',
                               dataset=mode, mode='head')
    dataset = {'tail': tail_dataset,'head':head_dataset}
    return dataset
def get_test_valid_dataloader(test_valid_dataset):
    tail_dataloader = DataLoader(
                    dataset = test_valid_dataset['tail'],
                    batch_size = 128,
                    shuffle = True,
                    num_workers = 0,
                    collate_fn = test_valid_dataset['tail'].collate_fn,
                    pin_memory=True,
)
    head_dataloader = DataLoader(
                    dataset=test_valid_dataset['head'],
                    batch_size=128,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=test_valid_dataset['head'].collate_fn,
                    pin_memory=True,
    )
    dataloader = {'tail':tail_dataloader,'head':head_dataloader}
    return dataloader
class TestDataset(Dataset):
    def __init__(self,in_path=None,dataset='valid',mode ='tail'):
        self.mode = mode
        self.in_path = in_path
        self.tri_file = in_path + dataset+ '2id.txt'
        self.ent_file = in_path + "entity2id.txt"
        self.rel_file = in_path + "relation2id.txt"
        self.head, self.tail, self.rel = self.__read_dataset()
        self.__count_htr()
        self.__constract_dataset()

    def __len__(self):
        return len(self.head)
    def __getitem__(self, idx):
        return self.Head[idx],  self.Rel[idx], self.Tail[idx],self.Lable[idx]

    def __constract_dataset(self):
        Lable = []
        hr = zip(self.head,self.rel)
        for i in hr :
            tlist = self.t_of_hr[i]
            Lable.append(self.__onehot(tlist))
        self.Head = torch.LongTensor(self.head)
        self.Rel = torch.LongTensor(self.rel)
        self.Tail = torch.LongTensor(self.tail)
        #TODO 进行修改
        self.Lable = torch.Tensor.bool(torch.from_numpy(np.array(Lable)))
        # self.Lable = torch.from_numpy(np.array(Lable))
        # self.Lable = torch.Tensor.bool(torch.tensor(Lable,dtype=torch.bool))
        # torch.bool

    def __onehot(self, tlist):
        t_onehot = np.zeros([self.ent_total])
        for t in tlist:
            t_onehot[t] = 1
        return t_onehot

    def __read_dataset(self):
        with open(self.ent_file, "r") as f:
            self.ent_total = (int)(f.readline())
        with open(self.rel_file, "r") as f:
            self.rel_total = (int)(f.readline())

        head = []
        tail = []
        rel = []

        with open(self.tri_file, "r") as f:
            triples_total = (int)(f.readline())
            for index in range(triples_total):
                h, t, r = f.readline().strip().split()
                if self.mode == 'tail':
                    head.append((int)(h))
                    tail.append((int)(t))
                    rel.append((int)(r))
                else :
                    head.append((int)(t))
                    tail.append((int)(h))
                    rel.append((int)(r)+self.rel_total)
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
        lable = torch.stack([_[3] for _ in data], dim=0)

        return head, rel, tail, lable




