import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from methods import gnnnet_copy
from methods.gnnnet import GnnNet

from methods import gnnnet
from methods import gnn
 


import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.dampnet import DampNet
from methods import dampnet_full
from methods import dampnet_full_class
from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file
from datasets import miniImageNet_few_shot



def train(base_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')     


    if not params.fine_tune:
      for epoch in range(start_epoch,stop_epoch):
          #print(epoch)
          model.train()
          if params.method in ["gnnnet", "gnnnet_maml", "gnnnet_neg_margin"]:
            if params.n_shot != 50:
              model.train_loop2(epoch, base_loader,  optimizer )
            else:
              model.train_loop50(epoch, base_loader,  optimizer )
          elif params.method in ["dampnet_full_sparse","dampnet_full","dampnet_full_class", "protonet_damp"]:
            model.train_loop_full(epoch, base_loader, optimizer, stop_epoch)
          else:
            model.train_loop(epoch, base_loader,  optimizer )
          if not os.path.isdir(params.checkpoint_dir):
              os.makedirs(params.checkpoint_dir)

          if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
              outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
              torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      for epoch in range(start_epoch,stop_epoch):
          #print(epoch)
          model.train()
          model.train_loop_finetune(epoch, base_loader,  optimizer ) 
          if epoch == (stop_epoch-1):
            model.MAML_update()
          if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
              outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
              torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
         
            
     
    return model

if __name__=='__main__':
    print("HELLO")
    
    params = parse_args('train')
    print(params.method)
    if not params.start_epoch > 0:
      np.random.seed(10) #original was 10

    image_size = 224
    optimization = 'Adam'

    if params.method in ['baseline'] :

        if params.dataset == "miniImageNet":
            #print('hi')
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            #print("bye")
            base_loader = datamgr.get_data_loader(aug = params.train_aug )
            #print("loaded")
        elif params.dataset == "CUB":

            base_file = configs.data_dir['CUB'] + 'base.json' 
            base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
       
        elif params.dataset == "cifar100":
            base_datamgr    = cifar_few_shot.SimpleDataManager("CIFAR100", image_size, batch_size = 16)
            base_loader    = base_datamgr.get_data_loader( "base" , aug = True )
                
            params.num_classes = 100

        elif params.dataset == 'caltech256':
            base_datamgr  = caltech256_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader = base_datamgr.get_data_loader(aug = False )
            params.num_classes = 257

        elif params.dataset == "DTD":
            base_datamgr    = DTD_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( aug = True )

        else:
           raise ValueError('Unknown dataset')
        
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)
        model           = BaselineTrain( model_dict[params.model], params.num_classes)

    elif params.method in ['dampnet_full_class','dampnet_full_sparse','protonet_damp','maml','relationnet','dampnet_full','dampnet','protonet', 'gnnnet', 'gnnnet_maml', 'metaoptnet', 'gnnnet_normalized', 'gnnnet_neg_margin']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        if params.dataset == "miniImageNet":
            print("loading")
            datamgr            = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            base_loader        = datamgr.get_data_loader(aug = params.train_aug)
            #datamgr         = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 64)
            #data_loader     = datamgr.get_data_loader(aug = False )
            
            print("BYE")

        else:
           raise ValueError('Unknown dataset')

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'protonet_damp':
            model           = protonet_damp.ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'relationnet':
            feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
            model           = RelationNet(feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method == 'maml':
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model           = MAML( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'metaoptnet':
            model           = MetaOptNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'gnnnet':
            if params.n_shot != 50:
                model           = GnnNet( model_dict[params.model], **train_few_shot_params)
            else:
                model           = gnnnet_copy.GnnNet( model_dict[params.model], **train_few_shot_params)
        
        
        elif params.method == 'gnnnet_maml':
            gnnnet.GnnNet.maml  = True
            gnn.Gconv.maml  = True
            gnn.Wcompute.maml = True
            model = gnnnet.GnnNet(model_dict[params.model], **train_few_shot_params)
            print(model.maml)
        elif params.method == 'gnnnet_neg_margin':
            model = gnnnet_neg_margin.GnnNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'gnnnet_normalized':
            model = gnnnet_normalized.GnnNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'dampnet':
            model = DampNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'dampnet_full':
            model = dampnet_full.DampNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'dampnet_full_sparse':
            model = dampnet_full_sparse.DampNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'dampnet_full_class':
            model = dampnet_full_class.DampNet(model_dict[params.model], **train_few_shot_params)
       
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    save_dir =  configs.save_dir
    print("WORKING")

    

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    print(params.checkpoint_dir)
  
    if params.start_epoch > 0:
      resume_file = get_assigned_file(params.checkpoint_dir, params.start_epoch -1)
      if resume_file is not None:
          tmp = torch.load(resume_file)
          
          state = tmp['state']
          state_keys = list(state.keys())
          for _, key in enumerate(state_keys):
              if "feature2." in key:
                  state.pop(key)
              if "feature3." in key:
                  state.pop(key)
          
          
      model.load_state_dict(state)

    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params)