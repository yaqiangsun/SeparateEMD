import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
    predict,
    set_seed
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm
import cv2
from model.trainer.loss import NCELoss,BCELoss,CrossEntropyLoss
from model.dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
     
class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        if args.not_random_val_task:
            print ('not fix val set for all epochs')
        # define test dataset used to eval in training.
        if args.dataset == 'MiniImageNet':
            # Handle MiniImageNet
            if args.deepemd == "fcn":
                from model.dataloader.mini_imagenet import MiniImageNet as Dataset
            if args.deepemd == "sampling":
                from model.dataloader.sampling.mini_imagenet import MiniImageNet as Dataset
        else:
            raise ValueError('Non-supported Dataset.')
        self.testset_training = Dataset('test', args)
        self.testsize =  args.num_eval_episodes # 1000
        self.test_sampler_training = CategoriesSampler(self.testset_training.label,
                        self.testsize, # args.num_eval_episodes,
                        args.eval_way, args.eval_shot + args.eval_query)
        self.test_loader_training = DataLoader(dataset=self.testset_training,
                                batch_sampler=self.test_sampler_training,
                                num_workers=args.num_workers,
                                pin_memory=True) 
        if args.not_random_val_task:
            print ('fix test set for all epochs')
            self.test_loader_training=[x for x in self.test_loader_training]

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        self.get_parameter_num()
        is_cuda_available = torch.cuda.is_available()
        for epoch in range(1, args.max_epoch + 1):
            start_time=time.time()
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            crossloss = Averager()
            pdloss = Averager()
            tqdm_gen = tqdm(self.train_loader)  

            start_tm = time.time()
            # for batch in self.train_loader:
            for i, batch in enumerate(tqdm_gen, 1):
                self.train_step += 1

                if is_cuda_available:
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]
               
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                logits, reg_logits, proto_dist = self.para_model(data)
                if reg_logits is not None:
                    loss = F.cross_entropy(logits, label)
                    total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                else:
                    loss = F.cross_entropy(logits, label)
                    total_loss = loss
                if proto_dist is not None:
                    total_loss = total_loss + proto_dist/1000
                    pdloss.add(proto_dist.item())
                    pass
                crossloss.add(loss.item())
                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'
                            .format(epoch, total_loss.item(), acc))   
                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()

            self.logger.add_histogram('train/logits', logits, self.train_epoch)
            if reg_logits is not None:
                self.logger.add_histogram('train/reg_logits', reg_logits, self.train_epoch)
            self.logger.add_scalar('train/cross_entropy_loss', float(crossloss.item()), self.train_epoch)
            self.logger.add_scalar('train/total_loss', float(tl1.item()), self.train_epoch)
            if proto_dist is not None:
                self.logger.add_scalar('train/proto_loss', float(pdloss.item()), self.train_epoch)
            if self.model.learnable_scale_attention:
                self.logger.add_scalar('train/learnable_scale_attention', float(self.model.learnable_scale_attention.item()), self.train_epoch)
            if self.model.learnable_scale_relative_attn:
                self.logger.add_scalar('train/learnable_scale_relative_attn', float(self.model.learnable_scale_relative_attn.item()), self.train_epoch)
            self.lr_scheduler.step()
            if args.max_epoch>30 and epoch > int(args.max_epoch*0.5):
                self.try_evaluate(epoch)
            # elif args.max_epoch<=30 and epoch>25 and epoch%1==0:
            #     self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )     
            # begin evaluate test dataset in training
            if (epoch>=0 and epoch<args.max_epoch*1.0 and epoch%1==0):
                self.evaluate_intraining()
            self.save_model('max_acc_step'+str(epoch))
            print ('This epoch takes %d seconds'%(time.time()-start_time),'\nstill need %.2f hour to finish'%((time.time()-start_time)*(args.max_epoch-epoch)/3600))

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            tqdm_gen = tqdm(data_loader)
            # for i, batch in enumerate(data_loader, 1):
            for i, batch in enumerate(tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits, proto_dist = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

        self.logger.add_histogram('val/logits', logits, self.train_epoch)
        if proto_dist is not None:
            self.logger.add_scalar('val/pd_loss', float(proto_dist.item()), self.train_epoch)
        
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        tqdm_gen.set_description('epo {}, val, loss={:.4f} acc={:.4f}'.format(self.train_epoch, vl, va))
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((args.num_test_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader, 1))
            for i, batch in pbar:
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits, proto_dist = self.model(data)
                prob = F.softmax(logits,dim=1)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                va, vap = compute_confidence_interval(record[:,1])
                pbar.set_description("ACC:"+str(va/(i)*args.num_test_episodes*100)[0:7])
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap
        
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            

    def evaluate_debug(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        # self.model.load_state_dict(torch.load(self.args.init_weights)['params'])
        self.model.eval()
        record = np.zeros((args.num_test_episodes*self.args.query//args.eval_query, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader, 1))
            for i, batch in pbar:
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                logits, _ = self.model(data)
                loss = F.cross_entropy(logits, label)
                pred = predict(logits, label)
                prob = F.softmax(logits,dim=1)
                self.tensorShow(data,pred,i,prob=prob)

                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

                va, vap = compute_confidence_interval(record[:,1])
                pbar.set_description("ACC:"+str(va/(i)*args.num_test_episodes*self.args.query//args.eval_query)[0:5])
                # print('test_acc:',va/(i+1)*10000)
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap

    def evaluate_intraining(self):
        # restore model args
        args = self.args
        # evaluation mode
        # self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((self.testsize*self.args.query//args.eval_query, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        
  
        with torch.no_grad():
            tqdm_gen = tqdm(self.test_loader_training)
            for i, batch in enumerate(tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                logits, proto_dist = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                tqdm_gen.set_description("Test Acc="+str(acc)[0:6])
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        self.logger.add_scalar('test/test_acc', float(va.item()), self.train_epoch)
        # self.logger.add_scalar('test/test_acc_interval', float(vap.item()), self.train_epoch)
        self.logger.add_scalar('test/test_loss', float(vl.item()), self.train_epoch)
        if proto_dist is not None:
            self.logger.add_scalar('test/pd_loss', float(proto_dist.item()), self.train_epoch)
        tqdm_gen.set_description('epo {}, test, loss={:.4f} acc={:.4f}'.format(self.train_epoch, vl, va))
        print("test_acc:",float(va.item()))

        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        return vl, va, vap

    def tensorShow(self,data,result=None,name=0,prob=None):
        data = data.clone()
        data = data.cpu().permute(0,2,3,1)
        data = data.numpy()
        data = np.array((data+2)*60,np.uint8)
        data = [data[i].copy() for i in range(data.shape[0])]
        result = result.clone().cpu()
        predict = result.numpy()
        predict = predict.astype(int)

        for i in range(predict.shape[0]):
            label = i % 5
            if predict[i] == label:# right
                img = data[i+25]
                # data[i+25] = cv2.rectangle(img, (5,5), (80,80), (0,255,0), 2)
            else:# wrong
                img = data[i+25]
                img = cv2.rectangle(img, (5,5), (80,80), (255,0,0), 2)
                data[i+25] = cv2.putText(img, str(predict[i]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            if prob is not None:
                predict_prob = prob[i][predict[i]]
                label_prob = prob[i][label]
                if predict[i] != label:# right
                    data[i+25] = cv2.putText(data[i+25], '%.2f'%predict_prob, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                data[i+25] = cv2.putText(data[i+25], '%.2f'%label_prob, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                pass
        data = np.array(data)
        images = data.reshape(data.shape[0]//5,5,data.shape[1],data.shape[2],data.shape[3])
        imgs=[cv2.hconcat(images[i]) for i in range(images.shape[0])] #水平拼接
        img=cv2.vconcat(imgs) #垂直拼接
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #
        cv2.imwrite('./debug/image_%05d.png' % name, img)