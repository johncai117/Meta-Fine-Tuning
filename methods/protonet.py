# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import copy
import backbone

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.first = True


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )

    def MAML_update(self):
      names = []
      for name, param in self.feature.named_parameters():
        if param.requires_grad:
          #print(name)
          names.append(name)
      
      names_sub = names[:-9]
      if not self.first:
        for (name, param), (name1, param1), (name2, param2) in zip(self.feature.named_parameters(), self.feature2.named_parameters(), self.feature3.named_parameters()):
          if name not in names_sub:
            dat_change = param2.data - param1.data ### Y - X
            new_dat = param.data - dat_change ### (Y- V) - (Y-X) = X-V
            param.data.copy_(new_dat)


    def set_forward_finetune(self,x):
        x = x.cuda()

    
        # get feature using encoder
        batch_size = 4
        support_size = self.n_way * self.n_support 

        for name, param  in self.feature.named_parameters():
          param.requires_grad = True

        x_var = Variable(x)
          
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() # (25,)

        #print(y_a_i)
        self.MAML_update() ## call MAML update
        
        x_b_i = x_var[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) 
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) # (25, 3, 224, 224)
        feat_network = copy.deepcopy(self.feature)
        classifier = Classifier(self.feat_dim, self.n_way)
        delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, feat_network.parameters()), lr = 0.01)
        loss_fn = nn.CrossEntropyLoss().cuda() ##change this code up ## dorop n way
        classifier_opt = torch.optim.Adam(classifier.parameters(), lr = 0.01, weight_decay=0.001) ##try it with weight_decay
        
        names = []
        for name, param in feat_network.named_parameters():
          if param.requires_grad:
            #print(name)
            names.append(name)
        
        names_sub = names[:-9] ### last Resnet block can adapt

        for name, param in feat_network.named_parameters():
          if name in names_sub:
            param.requires_grad = False    
      
          
        total_epoch = 5

        classifier.train()
        feat_network.train()

        classifier.cuda()
        feat_network.cuda()

        for epoch in range(total_epoch):
              rand_id = np.random.permutation(support_size)

              for j in range(0, support_size, batch_size):
                  classifier_opt.zero_grad()
                  
                  delta_opt.zero_grad()

                  #####################################
                  selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
                  
                  z_batch = x_a_i[selected_id]
                  y_batch = y_a_i[selected_id] 
                  #####################################

                  output = feat_network(z_batch)
                  scores  = classifier(output)
                  loss = loss_fn(output, y_batch)
                  #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)
                  #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

                  #####################################
                  loss.backward() ### think about how to compute gradients and achieve a good initialization

                  classifier_opt.step()
                  delta_opt.step()
        

        #feat_network.eval() ## fix this
        #classifier.eval()
        #self.train() ## continue training this!
        if self.first == True:
          self.first = False
        self.feature2 = copy.deepcopy(self.feature)
        self.feature3 = copy.deepcopy(feat_network) ## before the new state_dict is copied over
        self.feature.load_state_dict(feat_network.state_dict())
        
        for name, param  in self.feature.named_parameters():
            param.requires_grad = True
        
        z_support = self.feature(x_a_i.cuda()).view(self.n_way, self.n_support, -1)
        z_query = self.feature(x_b_i.cuda()).view(self.n_way,self.n_query,-1)

        
        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores
        
    def set_forward_loss_finetune(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward_finetune(x)

        return self.loss_fn(scores, y_query )


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
