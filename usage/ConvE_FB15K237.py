from framework.Data.TrainDataset import TrainDataset
from framework.Data.TestDataset import TestDataset
from torch.utils.data import DataLoader
from framework.Module.convE import ConvE
from framework.Trainer.Trainer import Trainer
import os
from framework.Data.TrainDataset import get_train_dataloader
from framework.Data.TestDataset import get_test_valid_dataset,get_test_valid_dataloader
from framework.parameters.parameter import hyperparameter
import time

args = hyperparameter().get_parse()
train_dataset = TrainDataset(in_path = "../benchmarks/"+args.dataset+'/')
train_dataloader = get_train_dataloader(train_dataset)

valid_dataset = get_test_valid_dataset(mode='valid',data=args.dataset)
valid_dataloader = get_test_valid_dataloader(valid_dataset)
# print(valid_dataset['tail'][0])
test_dataset = get_test_valid_dataset(mode='test')
test_dataloader = get_test_valid_dataloader(test_dataset)

convE = ConvE(ent_tot=train_dataset.get_ent_tot(),
              rel_tot=train_dataset.get_rel_tot(),
              embedding_dim=200,
              input_drop=0.2,
              feature_map_drop=0.2,
              hidden_drop=0.3,
              reshape_high=20,
              reshape_wide=10,
              filter_num=32,
)
# alpha: 学习率
# data = iter(train_dataloader)
if not args.restore: args.name = args.name + '_' + time.strftime('%Y_%m_%d') + '_' + time.strftime('%H_%M_%S')

checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'save_model',args.name+'.pth')
logger_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'log',args.name+'.log')
data_loader = [train_dataloader,valid_dataloader,test_dataloader]
trainer = Trainer(opt_method='Adam',
                 model=convE,
                 data_loader=data_loader,
                 train_times=1000,
                 use_gpu=True,
                 alpha=args.lr,
                 save_steps=None,
                 checkpoint_dir=checkpoint_dir,
                  if_restore = False,
                  gamma=40,
                  logger_path = logger_path)
result = trainer.fit()
print(result)




# loss = trainer.train_epoch()
# result = trainer.eval(valid_dataloader)
# print(result)
# print(loss)
# for i in range(101):
# 	h = next(data)[0]
# 	r = next(data)[1]
# 	pred = convE(h,r)