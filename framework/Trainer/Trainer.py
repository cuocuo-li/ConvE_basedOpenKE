import torch
from torch import optim
import numpy as np
import logging
import os
import tqdm

class Trainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 use_gpu=True,
                 alpha=0.001,
                 opt_method="Adam",
                 save_steps=None,
                 checkpoint_dir=None,
                 gamma=40,
                 if_restore=False,
                 logger_path = None):
        self.logger_path = logger_path
        self.logger = self.__get_logger()

        self.opt_method = opt_method
        self.work_threads = 8
        self.train_times = train_times
        self.optimizer = None

        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.train_data_loader,self.valid_data_loader,self.test_data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.gamma = gamma
        self.if_restore = if_restore

        self.set_device()
        self.set_opt()

    def set_device(self):
        if self.use_gpu==True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info('use gpu for training!')
        else:
            self.device = torch.device("cpu")
            self.logger.info('use cpu for training!')
    def set_opt(self):
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
            self.logger.info('set Adagrad as optimizer!')
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.logger.info('set Adadelta as optimizer!')
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.logger.info('set Adam as optimizer!')
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.logger.info('set SGD as optimizer!')
    def read_data(self,batch,mode):
        if mode=='train':

            head,rel,tail,e_in,e_out = [_.to(self.device) for _ in batch]
            return head,rel,tail,e_in,e_out
        else:
            head, rel, tail, lable = [_.to(self.device) for _ in batch]
            return head, rel, tail, lable
    def train_epoch(self):
        losses = []
        self.model.to(self.device)
        self.model.train()
        for step ,batch in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            sub,rel,tail,e_in,e_out=self.read_data(batch,'train')

            pred = self.model.forward(sub,rel)
            loss = self.model.bceloss(pred,tail)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        loss = np.mean(losses)
        return loss
    def eval_test(self,mode):
        if mode == 'valid':
            dataloader = self.valid_data_loader
        else:
            dataloader = self.test_data_loader
        head_result = self.predict(dataloader['head'],mode=mode,direct='head')
        tail_result = self.predict(dataloader['tail'],mode=mode,direct='tail')
        results={}
        results['mrr'] = round((head_result['head_mrr']+tail_result['tail_mrr'])/2,5)
        results['mr'] = round((head_result['head_mr']+tail_result['tail_mr'])/2,5)
        for i in 1,3,10:
            results["hits@{}".format(i)] = head_result['head_hits@{}'.format(i)]+tail_result['tail_hits@{}'.format(i)]
        results.update(head_result)
        results.update(tail_result)
        self.logger.info(results)
        return results


    def predict(self,dataloader,mode,direct):
        self.model.to(self.device).eval()
        with torch.no_grad():
            results = {}
            for step, batch in enumerate(dataloader):
                sub,rel,tail, lable=self.read_data(batch,mode)
                pred = self.model.forward(sub,rel)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, tail]
                pred = torch.where(lable, -torch.ones_like(pred)*10000, pred)
                pred[b_range, tail] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, tail]

                ranks = ranks.float()
                results[direct+'_'+'count'] = torch.numel(ranks) + results.get(direct+'_'+'count', 0.0)
                results[direct+'_'+'mr'] = torch.sum(ranks).item() + results.get(direct+'_'+'mr', 0.0)
                results[direct+'_'+'mrr'] = torch.sum(1.0 / ranks).item() + results.get(direct+'_'+'mrr', 0.0)
                results[direct+'_'+'hits@1'] = torch.numel(ranks[ranks <= 1]) + results.get(direct+'_'+'hits@1', 0.0)
                results[direct + '_' + 'hits@3'] = torch.numel(ranks[ranks <= 3]) + results.get(direct + '_' + 'hits@3',0.0)
                results[direct + '_' + 'hits@10'] = torch.numel(ranks[ranks <= 10]) + results.get(direct + '_' + 'hits@10',0.0)

                if step % 100 == 0:
                    self.logger.info('[{}, Step {}]'.format(mode.title(), step))
            count = float(results.get(direct+'_'+'count'))
            results[direct+'_'+'mr'] = round(results[direct+'_'+'mr'] / count, 5)
            results[direct+'_'+'mrr'] = round(results[direct+'_'+'mrr'] / count, 5)
            results[direct+'_'+'hits@1'] = round(results[direct+'_'+'hits@1'] / count, 5)
            results[direct + '_' + 'hits@3'] = round(results[direct + '_' + 'hits@3'] / count, 5)
            results[direct + '_' + 'hits@10'] = round(results[direct + '_' + 'hits@10'] / count, 5)
            results.pop(direct+'_'+'count')
        return results

    def save_model(self):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_result': self.best_val_result,
            'best_epoch': self.best_epoch,
        }
        torch.save(state, self.checkpoint_dir)

    def load_model(self):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(self.checkpoint_dir)
        state_dict = state['state_dict']
        self.best_val_mrr = state['best_val_result']['mrr']
        self.best_val_result = state['best_val_result']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])


    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_val_mrr, self.best_val_result, self.best_epoch, val_mrr = 0., {}, 0, 0.

        if self.if_restore:
            self.load_model()
            self.logger.info('Successfully Loaded previous model')

        kill_cnt = 0
        for epoch in range(self.train_times):
            # print(self.train_times)
            train_loss = self.train_epoch()
            val_results = self.eval_test(mode='valid')

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val_result = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model()
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.gamma > 5:
                    self.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.gamma))
                if kill_cnt > 25:
                    self.logger.info("Early Stopping!!")
                    break

            self.logger.info(
                '[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model()
        test_results = self.eval_test(mode='test')
        return test_results

    def __get_logger(self):

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        consoleHandler = logging.StreamHandler()
        fileHandler = logging.FileHandler(self.logger_path, mode='a', )

        formatter = logging.Formatter("%(asctime)s|%(levelname)8s|%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        consoleHandler.setFormatter(formatter)
        fileHandler.setFormatter(formatter)

        logger.addHandler(consoleHandler)
        logger.addHandler(fileHandler)
        return logger


