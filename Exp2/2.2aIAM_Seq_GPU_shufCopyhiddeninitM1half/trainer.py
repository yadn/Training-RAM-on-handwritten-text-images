import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import time
import shutil
import pickle

from tqdm import tqdm
from utils import AverageMeter
from model import RecurrentAttention
from tensorboard_logger import configure, log_value

from utils import plot_images

from matplotlib import pyplot as plt
import numpy as np


class Trainer(object):
    
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            #self.num_train = len(self.train_loader.sampler.indices)
            #self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 83
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        
        self.trainSamplesSize=len(self.train_loader.trainSamples)
        
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = 'ram_{}_{}x{}_{}'.format(
            config.num_glimpses, config.patch_size,
            config.patch_size, config.glimpse_scale
        )

        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size, self.num_patches, self.glimpse_scale,
            self.num_channels, self.loc_hidden, self.glimpse_hidden,
            self.std, self.hidden_size, self.num_classes,
        )
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # # initialize optimizer and scheduler
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=self.momentum,
        # )
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer, 'min', patience=self.lr_patience
        # )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=3e-4,
        )

    def reset(self):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        h_t = torch.zeros(self.batch_size, self.hidden_size)
        h_t[:,1:2:self.hidden_size] = -1
        h_t = Variable(h_t).type(dtype)
        l_t = torch.ones(self.batch_size,2)
        l_t[:,0] *= -1
        l_t[:,1] *= 0
        #l_t = torch.stack([, torch.zeros(self.batch_size,1)], dim=1)
        #print(l_t, l_t.shape)
        #l_t = torch.Tensor(self.batch_size, 2)#.uniform_(-1, 1)
        l_t = Variable(l_t).type(dtype)

        return h_t, l_t

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

#        print("\n[*] Train on {} samples, validate on {} samples".format(
#            self.num_train, self.num_valid)
#        )

        
        #self.trainDataset(1)
        
        for epoch in range(self.start_epoch, self.epochs):
         
            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss, train_acc = self.trainDataset(epoch)#self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validateDataset(epoch)#self.train_one_epoch(epoch)

            # evaluate on validation set
            #valid_loss, valid_acc = 0,0
            #valid_loss, valid_acc = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            # self.scheduler.step(valid_loss)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 }, is_best
            )
    
    
    def trainDataset(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.trainSamplesSize) as pbar:
	        self.train_loader.trainSet()
	        #rew = torch.linspace(1,0.1,self.num_glimpses,dtype=float)
	        i = 0    
	        while self.train_loader.hasNext():
	            #if(i>2): break
	            i += 1
	            iterInfo = self.train_loader.getIteratorInfo()
	            batch=self.train_loader.getNext()
	            x = batch.imgs
	            y = batch.gtTexts
	            x,y = torch.tensor(x),torch.tensor(y)
	            x = x[:,None,:,:]
	            x = x.type(torch.FloatTensor)
	            self.batch_size = x.shape[0]
	            bmax = 0
	            #print("y0",y)
	            for ib in range(self.batch_size):
	                #print((y[ib] != 82))
	                #print((y[ib] != 82).nonzero())
	                bmax=max(bmax,len((y[ib] != 82).nonzero()))
	            #print("bmax",bmax)
	            y = y[:,:bmax]
	            #print("y1",y)
	            x,y = Variable(x),Variable(y)
	            if self.use_gpu:
	                x, y = torch.tensor(x.clone().detach()).cuda(), torch.tensor(y.clone().detach()).cuda()
	            #y=y-1 #adjusting to 0-25
	            #x=x.T
	            #X = x.numpy()
	            #X = np.transpose(X, [0, 2, 3, 1])
	                            
	            #plot_images(x, y)
	            #print(x.shape,y)
	            #print("\n",i,"*************************************")
	            #
	            #plot = False
	            #if (epoch % self.plot_freq == 0) and (i == 0):
	            #    plot = True

	            # initialize location vector and hidden state
	            h_t, l_t = self.reset()#returns uniform(-1,1) x,y

	            # save images
	            imgs = []
	            imgs.append(x[0:4])
	            

	            # extract the glimpses
	            locs = []
	            locs.append(l_t[0:4])
	            log_pi = []
	            baselines = []
	            log_probas_list = []
	            predicted_list = []
	            R_list = []
	            baselines_list = []
	            #print("y0", y0)
	            y0new = []
	            y0 = []
	            onecharglimpse = 8
	            Rdist = []
	            #print("no_glimpse", self.num_glimpses)
	            #print(bmax*onecharglimpse)
	            for t in range(bmax*onecharglimpse):#self.num_glimpses): #- 1):
	                    # forward pass through model
	                    # h_t, l_t, b_t, p = self.model(x, l_t, h_t)
	                if t%(onecharglimpse) == 0:
	                    y0 = y[:,t//(onecharglimpse)]
	                    #for b in range(self.batch_size):
	                        #y0.append(y[b][t//(self.num_glimpses)])#first element for 8 glimpses in the batch #[:,t//sel...]Loop can be removed 
	                    #y0 = torch.tensor(y0)
	                    y0new.append(y0)#will be 32X22
	                #y0 = torch.tensor(y0).cuda()
	                l_t_Prev = l_t
	                h_t, l_t, b_t, log_probas1, p = self.model(x, l_t, h_t, last=True)
	                if (t+1)%(onecharglimpse) == 0:
	                    log_probas_list.append(log_probas1)
	                    predicted_list.append(torch.max(log_probas1, 1)[1])#22X32X83
	                    predicted1 = torch.max(log_probas1, 1)[1]
	                    R1 = (predicted1.detach() == y0).float()
	                    R1 = R1.unsqueeze(1).repeat(1, onecharglimpse)
	                    R_list.append(R1)#22X32X8
	                locs.append(l_t[0:4])
	                baselines.append(b_t)#
	                #Rdist.append(-1*(torch.dist(l_t_Prev,l_t,2)))
	                Rdist.append(-1*(torch.norm(l_t_Prev-l_t,p=2,dim=1)))
	                log_pi.append(p)
	            print(len(Rdist))
	            print(Rdist[0].shape)
	            R1 = R_list[0]
	            for R2 in R_list[1:]:
	                R1 = torch.cat((R1,R2),1)
	            #print(R1.shape)
	            Rdist = torch.stack(Rdist).transpose(1, 0)
	            baselines = torch.stack(baselines).transpose(1, 0)#32X176
	            #print(baselines.shape)
	            log_pi = torch.stack(log_pi).transpose(1, 0)

	            loss_action = F.nll_loss(log_probas_list[0], y0new[0])
	            for l in range(1,len(y0new)):
	                loss_action += F.nll_loss(log_probas_list[l], y0new[l])
	            loss_baseline = F.mse_loss(baselines, R1)
	            loss_baseline += F.mse_loss(baselines, Rdist)
	            #print("predicted_list", predicted_list)
	            # compute accuracy
	            # compute reinforce loss
	            # summed over timesteps and averaged across batch
	            adjusted_reward = R1 - baselines.detach()
	            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
	            loss_reinforce = torch.mean(loss_reinforce, dim=0)

	            # sum up into a hybrid loss
	            loss = loss_action + loss_baseline + loss_reinforce
	            predicted_list,y0new = torch.cat(predicted_list),torch.cat(y0new)
	            correct = (predicted_list == y0new).float()
	            acc = 100 * (correct.sum() / len(y0new))
	            #acc = 100 * ((correct[(y0new != 82).nonzero()]).sum() / len((y0new != 82).nonzero()))
	            # store
	            #losses.update(loss.data[0], x.size()[0])
	            #accs.update(acc.data[0], x.size()[0])
	            #print("loss", loss)
	            losses.update(loss.data.item(), x.size()[0])
	            accs.update(acc.data.item(), x.size()[0])

	            # compute gradients and update SGD
	            self.optimizer.zero_grad()
	            loss.backward()
	            self.optimizer.step()

	            # measure elapsed time
	            toc = time.time()
	            batch_time.update(toc-tic)

	            pbar.set_description(
	                (
	                    "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
	                        (toc-tic), loss.data.item(), acc.data.item()
	                    )
	                )
	            )
	            pbar.update(self.batch_size)

	            # dump the glimpses and locs
	            if (1):#plot:
	                if self.use_gpu:
	                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
	                    locs = [l.cpu().data.numpy() for l in locs]
	                else:
	                    imgs = [g.data.numpy().squeeze() for g in imgs]
	                    locs = [l.data.numpy() for l in locs]
	                pickle.dump(
	                    imgs, open(
	                        self.plot_dir + "g_{}.p".format(epoch+1),
	                        "wb"
	                    )
	                )
	                pickle.dump(
	                    locs, open(
	                        self.plot_dir + "l_{}.p".format(epoch+1),
	                        "wb"
	                    )
	                )
	            if self.use_tensorboard:
	                iteration = epoch*len(self.train_loader) + i
	                log_value('train_loss', losses.avg, iteration)
	                log_value('train_acc', accs.avg, iteration)
        return losses.avg, accs.avg
    def validateDataset(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        self.train_loader.validationSet()
        i = 0    
        while self.train_loader.hasNext():
            #if(i>2): break
            i += 1
            iterInfo = self.train_loader.getIteratorInfo()
            batch=self.train_loader.getNext()
            
            x = batch.imgs
            y = batch.gtTexts
            x,y = torch.tensor(x),torch.tensor(y)
            self.batch_size = x.shape[0]
            bmax = 0
            #print("y0",y)
            for ib in range(self.batch_size):
                #print((y[ib] != 82))
                #print((y[ib] != 82).nonzero())
                bmax=max(bmax,len((y[ib] != 82).nonzero()))
            #print("bmax",bmax)
            y = y[:,:bmax]
            #x = x.type(torch.cuda.FloatTensor)
            x = x[:,None,:,:]
            x = x.type(torch.FloatTensor)
            x,y = Variable(x),Variable(y)
            if self.use_gpu:
                x, y = torch.tensor(x).cuda(), torch.tensor(y).cuda()
            self.batch_size = x.shape[0]
            #x = x.repeat(self.M, 1, 1, 1)

            #print(x.shape)
            # initialize location vector and hidden state
            
            h_t, l_t = self.reset()

            # extract the glimpses
            log_pi = []
            locs = []
            baselines = []
            log_probas_list = []
            predicted_list = []
            R_list = []    
            
            y0new = []
            y0 = []
            onecharglimpse = 8
            #print("no_glimpse", self.num_glimpses)
            for t in range(bmax*onecharglimpse):
                     # forward pass through model
                    # h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                if t%(onecharglimpse) == 0:
                    y0 = y[:,t//(onecharglimpse)]
                    '''for b in range(self.batch_size):
                        y0.append(y[b][t//(self.num_glimpses)])'''#first element for 8 glimpses in the batch #[:,t//sel...]Loop can be removed 
                    y0new.append(y0)#will be 32X22
                #y0 = torch.tensor(y0).cuda()
                h_t, l_t, b_t, log_probas1, p = self.model(x, l_t, h_t, last=True)
                if (t+1)%(onecharglimpse) == 0:
                    log_probas_list.append(log_probas1)
                    predicted_list.append(torch.max(log_probas1, 1)[1])
                    predicted1 = torch.max(log_probas1, 1)[1]
                    R1 = (predicted1.detach() == y0).float()
                    R1 = R1.unsqueeze(1).repeat(1, onecharglimpse)
                    R_list.append(R1)#22X32X8
                    # store
                locs.append(l_t[0:9])
                baselines.append(b_t)
                log_pi.append(p)
            R1 = R_list[0]
            for R2 in R_list[1:]:
                R1 = torch.cat((R1,R2),1)
            #print(R1.shape)                    
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas_list[0], y0new[0])
            for l in range(1,len(y0new)):
                loss_action += F.nll_loss(log_probas_list[l], y0new[l])
            loss_baseline = F.mse_loss(baselines, R1)
            predicted_list,y0new = torch.cat(predicted_list),torch.cat(y0new)
            # compute reinforce loss
            adjusted_reward = R1 - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce

            # compute accuracy
            correct = (predicted_list == y0new).float()
            #acc = 100 * ((correct[(y0new != 82).nonzero()]).sum() / len((y0new != 82).nonzero()))
            acc = 100 * (correct.sum() / len(y0new))

#gb changes*********************************************************************************************            
            # store
            #losses.update(loss.data[0], x.size()[0])
            #accs.update(acc.data[0], x.size()[0])
            
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch*len(self.valid_loader) + i
                log_value('valid_loss', losses.avg, iteration)
                log_value('valid_acc', accs.avg, iteration)

        return losses.avg, accs.avg
    def testData(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        i = 0    
        while self.train_loader.hasNext():
            #if(i>2): break
            i += 1
            iterInfo = self.train_loader.getIteratorInfo()
            batch=self.train_loader.getNext()
            
            x = batch.imgs
            x = torch.tensor(x)
            #x = x.type(torch.cuda.FloatTensor)
            x = x[:,None,:,:]
            x = x.type(torch.FloatTensor)
            
            y = batch.gtTexts
            
            self.batch_size = x.shape[0]
            
            h_t, l_t = self.reset()

            log_pi = []
            locs = []
            baselines = []
            log_probas_list = []
            predicted_list = []
            R_list = []    
            
            y0new = []
            y0 = []
            # extract the glimpses
            # extract the glimpses
            for t in range(self.num_glimpses):
                # forward pass through model
                if(t%8 == 0):
                    y0=[]
                    for b in range(self.batch_size):
                        print(b,t//8,t)
                        y0.append(y[b][t//8])
                    y0new+=y0
                
                y0=torch.tensor(y0)
                
                h_t, l_t, b_t, log_probas1, p = self.model(x, l_t, h_t,last=True)
                if(t+1)%8==1:
                    log_probas_list.append(log_probas1)
                    predicted_list.append(torch.max(log_probas1,1)[1])
                            
                locs.append(l_t[0:9])
                baselines.append(b_t)
                log_pi.append(p)
                
                predicted1=torch.max(log_probas1, 1)[1]
                R1 = (predicted1.detach() == y0).float()
                R_list.append(R1)
                    
                # store
                #baselines.append(b_t)
                #log_pi.append(p)

#            # last iteration
#            h_t, l_t, b_t, log_probas, p = self.model(
#                x, l_t, h_t, last=True
#            )
#            log_pi.append(p)
#            baselines.append(b_t)

            R = R_list
            R = torch.stack(R).transpose(1,0)

            pred = log_probas_list.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct) / (self.num_test)
        error = 100 - perc
        print(
            '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
                correct, self.num_test, perc, error)
        )

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, y) in enumerate(self.test_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x, volatile=True), Variable(y)
            y=y-1
            # duplicate 10 times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(
                x, l_t, h_t, last=True
            )

            log_probas = log_probas.view(
                self.M, -1, log_probas.shape[-1]
            )
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct) / (self.num_test)
        error = 100 - perc
        print(
            '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
                correct, self.num_test, perc, error)
        )

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )
