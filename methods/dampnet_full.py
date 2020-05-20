import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from methods.meta_template import MetaTemplate
from methods.gnn import GNN_nl
import backbone
import math
from torch.autograd import Variable

class DampNet(MetaTemplate):
  maml=False
  def __init__(self, model_func,  n_way, n_support):
    super(DampNet, self).__init__(model_func, n_way, n_support)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.gnn_dim = 128
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, self.gnn_dim), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(self.gnn_dim + self.n_way, 96, self.n_way)
    self.method = 'DampNet'
    
    self.num_ex = 20 ##making change to 50?
    #self.meta_store_mean = torch.zeros((self.num_ex,self.feat_dim))
    #self.meta_store_std = torch.zeros((self.num_ex,self.n_support*self.n_way,self.feat_dim))
    #self.corruption = torch.from_numpy(np.diag(np.ones(self.feat_dim)))


    ### comparison / recovery network

    self.W_R = nn.Bilinear(self.feat_dim, self.feat_dim, 300, bias = False).cuda()
    self.V_R = nn.Linear(self.feat_dim*2, 300).cuda()

    self.W_R_std = nn.Bilinear(self.feat_dim, self.feat_dim, 300, bias = False).cuda()
    self.V_R_std = nn.Linear(self.feat_dim*2, 300).cuda()

    ## MLP
    self.tanh = nn.Tanh()
    self.layer1 = nn.Linear(300*2, 500)
    self.layer2 = nn.Linear(500, 500)
    self.layer3 = nn.Linear(500, self.feat_dim)
    self.layer1_add = nn.Linear(300*2, 500)
    self.layer2_add = nn.Linear(500, 500)
    self.layer3_add = nn.Linear(500, self.feat_dim)


    self.final_meta_prototype = torch.zeros(self.feat_dim)
    self.final_meta_prototype_std = torch.zeros(self.feat_dim)
    self.final_meta_prototypes_initialized = False
    self.final_all_feats = torch.zeros(5,100,self.n_way*self.n_support, self.feat_dim) ##replace first and second dim with desired

    #self.meta_prototype_mean = torch.mean((1, self.feat_dim))
    #self.meta_prototype_std = torch.mean((1, self.feat_dim))
    self.call_count = 150 ##if restart
    self.first = True    

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

    self.cuda()

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    #self.meta_store_mean.cuda()
    #self.meta_store_std.cuda()
    self.W_R.cuda()
    self.V_R.cuda()
    self.W_R_std.cuda()
    self.V_R_std.cuda()
    self.tanh.cuda()
    self.layer1.cuda()
    self.layer2.cuda()
    self.layer3.cuda()
    self.layer1_add.cuda()
    self.layer2_add.cuda()
    self.layer3_add.cuda()
    self.final_all_feats.cuda()
    self.final_meta_prototype.cuda()
    self.final_meta_prototype_std.cuda()
    self.support_label = self.support_label.cuda()
    return self
  
  def get_all_feat(self, all_feat):
    all_feat = all_feat.cuda().detach()
    self.final_meta_prototype = torch.mean(all_feat, axis = 0).cuda().detach()
    self.final_meta_prototype_std = all_feat.std(axis = 0).cuda().detach()
    self.final_meta_prototypes_initialized = True
    return self

  def set_forward(self,x,is_feature=False, domain_shift = False):
    x = x.cuda()
    if domain_shift == False:
      if is_feature:
        # reshape the feature tensor: n_way * n_s + 15 * f
        assert(x.size(1) == self.n_support + 15)
        z = self.fc(x.view(-1, *x.size()[2:]))
        z = z.view(self.n_way, -1, z.size(1))
      else:
        # get feature using encoder ## brought it to higher level
        x2 = x.view(self.n_way, -1, x.size(1))
        x_mean = torch.mean(x2[:,:self.n_support,:], axis = (0,1)).detach()
        x_std = x2[:,:self.n_support,:].std(axis = (0,1)).detach()
        
    

      # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
      if self.final_meta_prototypes_initialized == False:
        self.fc[0].weight.requires_grad = True
        self.fc[0].bias.requires_grad = True
        self.gnn = self.gnn.train()
        
        z = self.fc(x)
        z = z.view(self.n_way, -1, z.size(1))
        #z_mean = torch.mean(z[:,:self.n_support,:], axis = (0,1), keepdim = True)
        #print("AVG SHAPE")
        #z = z - z_mean
        #z_norm = torch.norm(z, dim = 2, keepdim = True)
        #z = z / z_norm
        z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
        assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
        scores = self.forward_gnn(z_stack)
        idx = self.call_count % self.num_ex
        #self.meta_store_mean[idx] =  x_mean
        #self.meta_store_std[idx] = x2[:,:self.n_support,:].detach().reshape(-1,self.feat_dim)
        self.call_count += 1
        return scores
      elif self.call_count % 2 == 1:
        ### corruption vector
        a = 0.5
        b = 0.8
        perc = (b - a) * np.random.random_sample() + a
        perc_zeros = perc/2
        a2 = 1.5
        b2 = 4
        m_fac = (b2 - a2) * np.random.random_sample() + a2
        meta_prototype_mean = self.final_meta_prototype
        meta_prototype_std = self.final_meta_prototype_std
        one_zeros = np.concatenate((np.ones(self.feat_dim - math.floor(self.feat_dim * perc_zeros)), np.zeros(math.floor(self.feat_dim * perc_zeros))))
        np.random.shuffle(one_zeros)
        corruption = torch.from_numpy(np.diag(one_zeros)).cuda().float()
        corruption_bias = torch.from_numpy(np.zeros(self.feat_dim)).cuda().float()
        temp = np.asarray(list(range(0,self.feat_dim)))
        random_idx = np.random.choice(temp, math.floor(perc*self.feat_dim))
        random_idx2 = np.random.choice(temp, math.floor(perc*self.feat_dim))
        rand_idx_col = np.random.choice(random_idx2, 1)
        ad_sub = np.concatenate((np.ones(self.feat_dim - math.floor(self.feat_dim*0.5) ), - np.ones(math.floor(self.feat_dim * 0.5))))
        np.random.shuffle(ad_sub)
        t_sample = m_fac * np.reshape(np.random.standard_t(5, self.feat_dim * self.feat_dim), (self.feat_dim, self.feat_dim))
        t_sample_bias = np.random.standard_t(5, self.feat_dim) + ad_sub
        t_sample_bias = torch.from_numpy(-np.squeeze(t_sample[:,rand_idx_col]) +t_sample_bias).cuda().float()
        t_sample = torch.from_numpy(t_sample).cuda().float()
        corruption[random_idx,random_idx2] += t_sample[random_idx, random_idx2]
        corruption_bias[random_idx2] += t_sample_bias[random_idx2]
        corrupt_x = torch.matmul(x,corruption).detach().cuda() ## new input
        corrupt_x += (m_fac * corruption_bias)
        corrupt_x2 = corrupt_x.view(self.n_way, -1, x.size(1))
        corrupt_x_mean = torch.mean(corrupt_x2[:,:self.n_support,:], axis = (0,1)).detach()
        corrupt_x_std = corrupt_x2[:,:self.n_support,:].std(axis = (0,1)).detach()
        
        W_out_m = self.W_R(meta_prototype_mean, corrupt_x_mean).cuda()
        V_out_m = self.V_R(torch.cat((meta_prototype_mean, corrupt_x_mean)))
        NTN_out = W_out_m + V_out_m

        W_out_m_std = self.W_R_std(meta_prototype_std.cuda(), corrupt_x_std.cuda()).cuda()
        V_out_m_std = self.V_R_std(torch.cat((meta_prototype_std.cuda(), corrupt_x_std.cuda())).cuda())
        NTN_out_std = W_out_m_std + V_out_m_std

        compare_input = self.tanh(torch.cat((NTN_out, NTN_out_std)))
        mult_ = F.relu(self.layer1(compare_input))
        mult_ = F.relu(self.layer2(mult_))
        #upper = torch.tensor([1]).float().cuda()
        #lower = torch.tensor([1]).float().cuda()
        mult_ = self.layer3(mult_)

        add_ = F.relu(self.layer1_add(compare_input))
        add_ = F.relu(self.layer2_add(add_))
        add_ = self.layer3_add(add_) ## sparse add

        recovered_x = torch.mul(corrupt_x.detach(), mult_) + add_
        self.fc[0].weight.requires_grad = False
        self.fc[0].bias.requires_grad = False
        self.gnn = self.gnn.eval()
        r_z = self.fc(recovered_x)

        r_z = r_z.view(self.n_way, -1, r_z.size(1))
        #r_z_mean = torch.mean(r_z[:,:self.n_support,:], axis = (0,1), keepdim = True)
        #print("AVG SHAPE")
        #r_z = r_z - r_z_mean
        #r_z_norm = torch.norm(r_z, dim = 2, keepdim = True)
        #r_z = r_z / r_z_norm
        r_z_stack = [torch.cat([r_z[:, :self.n_support], r_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, r_z.size(2)) for i in range(self.n_query)]
        assert(r_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
        
        r_scores = self.forward_gnn(r_z_stack)
        scores = r_scores
        #self.meta_store_mean[idx] =  x_mean
        #self.meta_store_std[idx] = x2[:,:self.n_support,:].detach().reshape(-1,self.feat_dim)
        self.call_count += 1
        return scores
      elif self.call_count % 2 == 0:
        meta_prototype_mean = self.final_meta_prototype
        meta_prototype_std = self.final_meta_prototype_std
        self.fc[0].weight.requires_grad = True
        self.fc[0].bias.requires_grad = True
        self.gnn = self.gnn.train()
        
        W_out_m = self.W_R(meta_prototype_mean, x_mean.detach()).cuda()
        V_out_m = self.V_R(torch.cat((meta_prototype_mean, x_mean.detach())))
        NTN_out = W_out_m + V_out_m

        W_out_m_std = self.W_R_std(meta_prototype_std.cuda(), x_std.cuda()).cuda()
        V_out_m_std = self.V_R_std(torch.cat((meta_prototype_std.cuda(), x_std.cuda())).cuda())
        NTN_out_std = W_out_m_std + V_out_m_std

        compare_input = self.tanh(torch.cat((NTN_out, NTN_out_std)))
        mult_ = F.relu(self.layer1(compare_input))
        mult_ = F.relu(self.layer2(mult_))
        mult_ = self.layer3(mult_)

        add_ = F.relu(self.layer1_add(compare_input))
        add_ = F.relu(self.layer2_add(add_))
        #thresh_add = (add_ > 1).float() * 1
        add_ = self.layer3_add(add_)

        recovered_x = torch.mul(x, mult_) + add_ ### use back normal x
        r_z = self.fc(recovered_x)

        r_z = r_z.view(self.n_way, -1, r_z.size(1))
        #r_z_mean = torch.mean(r_z[:,:self.n_support,:], axis = (0,1), keepdim = True)
        #print("AVG SHAPE")
        #r_z = r_z - r_z_mean
        #r_z_norm = torch.norm(r_z, dim = 2, keepdim = True)
        #r_z = r_z / r_z_norm
        r_z_stack = [torch.cat([r_z[:, :self.n_support], r_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, r_z.size(2)) for i in range(self.n_query)]
        assert(r_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
        r_scores = self.forward_gnn(r_z_stack)
        scores = r_scores
        idx = self.call_count % self.num_ex
        #self.meta_store_mean[idx] =  x_mean
        #self.meta_store_std[idx] = x2[:,:self.n_support,:].detach().reshape(-1,self.feat_dim)
        self.call_count += 1
        return scores
    elif domain_shift == True:
      #print("DOMAIN SHIFT")
      if is_feature:
        assert(x.size(1) == self.n_support + 15)
        x = x.view(-1, *x.size()[2:])
        x2 = x.view(self.n_way, -1, x.size(1))
        ### LOAD PROTOTYPES
        meta_prototype_mean = self.final_meta_prototype
        meta_prototype_std = self.final_meta_prototype_std
        x_mean = torch.mean(x2[:,:self.n_support,:], axis = (0,1)).detach()
        x_std = x2[:,:self.n_support,:].std(axis = (0,1)).detach()
        
        W_out_m = self.W_R(meta_prototype_mean, x_mean).cuda()
        V_out_m = self.V_R(torch.cat((meta_prototype_mean, x_mean)))
        NTN_out = W_out_m + V_out_m

        W_out_m_std = self.W_R_std(meta_prototype_std.cuda(), x_std.cuda()).cuda()
        V_out_m_std = self.V_R_std(torch.cat((meta_prototype_std.cuda(), x_std.cuda())).cuda())
        NTN_out_std = W_out_m_std + V_out_m_std

        compare_input = self.tanh(torch.cat((NTN_out, NTN_out_std)))
        mult_ = F.relu(self.layer1(compare_input))
        mult_ = F.relu(self.layer2(mult_))
        mult_ = self.layer3(mult_)

        add_ = F.relu(self.layer1_add(compare_input))
        add_ = F.relu(self.layer2_add(add_))
        add_ = self.layer3_add(add_)

        recovered_x = torch.mul(x, mult_) + add_ ### use back normal x
        r_z = self.fc(recovered_x)

        r_z = r_z.view(self.n_way, -1, r_z.size(1))
        #r_z_mean = torch.mean(r_z[:,:self.n_support,:], axis = (0,1), keepdim = True)
        #print("AVG SHAPE")
        #r_z = r_z - r_z_mean
        #r_z_norm = torch.norm(r_z, dim = 2, keepdim = True)
       # r_z = r_z / r_z_norm
        r_z_stack = [torch.cat([r_z[:, :self.n_support], r_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, r_z.size(2)) for i in range(self.n_query)]
        assert(r_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
        r_scores = self.forward_gnn(r_z_stack)
        scores = r_scores
        idx = self.call_count % self.num_ex

        return scores
      else:
        print("NOT IMPLEMENTED YET")
  
  def set_forward_unsup(self,x,x_u_mean, x_u_std,is_feature=False, domain_shift = True):
    x = x.cuda()
    assert (domain_shift == True)
    if domain_shift == True:
      #print("DOMAIN SHIFT")
      if is_feature:
        assert(x.size(1) == self.n_support + 15)
        x = x.view(-1, *x.size()[2:])
        #x2 = x.view(self.n_way, -1, x.size(1))
        ### LOAD PROTOTYPES
        meta_prototype_mean = self.final_meta_prototype
        meta_prototype_std = self.final_meta_prototype_std
        #x_mean = torch.mean(x2[:,:self.n_support,:], axis = (0,1)).detach()
        #x_std = x2[:,:self.n_support,:].std(axis = (0,1)).detach()
        
        W_out_m = self.W_R(meta_prototype_mean, x_u_mean).cuda()
        V_out_m = self.V_R(torch.cat((meta_prototype_mean, x_u_mean)))
        NTN_out = W_out_m + V_out_m

        W_out_m_std = self.W_R_std(meta_prototype_std.cuda(), x_u_std.cuda()).cuda()
        V_out_m_std = self.V_R_std(torch.cat((meta_prototype_std.cuda(), x_u_std.cuda())).cuda())
        NTN_out_std = W_out_m_std + V_out_m_std

        compare_input = self.tanh(torch.cat((NTN_out, NTN_out_std)))
        mult_ = F.relu(self.layer1(compare_input))
        mult_ = F.relu(self.layer2(mult_))
        mult_ = self.layer3(mult_)

        add_ = F.relu(self.layer1_add(compare_input))
        add_ = F.relu(self.layer2_add(add_))
        add_ = self.layer3_add(add_)

        recovered_x = torch.mul(x, mult_) + add_ ### use back normal x
        r_z = self.fc(recovered_x)

        r_z = r_z.view(self.n_way, -1, r_z.size(1))
        #r_z_mean = torch.mean(r_z[:,:self.n_support,:], axis = (0,1), keepdim = True)
        #print("AVG SHAPE")
        #r_z = r_z - r_z_mean
        #r_z_norm = torch.norm(r_z, dim = 2, keepdim = True)
       # r_z = r_z / r_z_norm
        r_z_stack = [torch.cat([r_z[:, :self.n_support], r_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, r_z.size(2)) for i in range(self.n_query)]
        assert(r_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
        r_scores = self.forward_gnn(r_z_stack)
        scores = r_scores
        idx = self.call_count % self.num_ex

        return scores

    else:
      print("NOT IMPLEMENTED")
      
  

  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return loss
  
  def train_loop_full(self, epoch, train_loader, optimizer, final_epoch):
        print_freq = 10
        num_reset = 5
        avg_loss=0
        start = 206

        if epoch == 208:
          print(self.final_all_feats)
          print(self.W_R)
          print(self.V_R)
          print(self.W_R_std)
          print(self.V_R_std)

        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            x = self.feature(x.view(-1, *x.size()[2:]).cuda())
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()
            feats = x.detach()
            feats = feats.view(self.n_way, -1, feats.size(1))
            feats = feats[:,:self.n_support,:]
            #print(feats.shape)
            feats = feats.reshape(self.n_way * self.n_support, self.feat_dim)
            if i % print_freq==0:
              #print(optimizer.state_dict()['param_groups'][0]['lr'])
              print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
            if i == 0:
              all_feats = torch.zeros(len(train_loader), feats.shape[0], feats.shape[1])
            all_feats[i] = feats
        #if epoch % 5 == 0:
        self.final_all_feats[(epoch % 5)] = all_feats
        if epoch >= start:
          self = self.get_all_feat(self.final_all_feats.view(5 * len(train_loader) * feats.shape[0], feats.shape[1]))
        if epoch == (final_epoch-1):
          proto_numpy = self.final_meta_prototype.detach().cpu().numpy()
          proto_numpy_std = self.final_meta_prototype_std.detach().cpu().numpy()
          name1 = "proto_numpy_"+ str(epoch) + ".npy"
          name2 = "proto_numpy_std_" + str(epoch) + ".npy"
          np.save(name1, proto_numpy)
          np.save(name2, proto_numpy)

  def set_forward_adaptation_full(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        x = x.cuda()
        original_x = x.cuda()
        x = x.view(-1, *x.size()[2:])
        x2 = x.view(self.n_way, -1, x.size(1))
        ### LOAD PROTOTYPES
        meta_prototype_mean = self.final_meta_prototype
        meta_prototype_std = self.final_meta_prototype_std
        x_mean = torch.mean(x2[:,:self.n_support,:], axis = (0,1)).detach()
        x_std = x2[:,:self.n_support,:].std(axis = (0,1)).detach()
        
        W_out_m = self.W_R(meta_prototype_mean, x_mean).cuda()
        V_out_m = self.V_R(torch.cat((meta_prototype_mean, x_mean)))
        NTN_out = W_out_m + V_out_m

        W_out_m_std = self.W_R_std(meta_prototype_std.cuda(), x_std.cuda()).cuda()
        V_out_m_std = self.V_R_std(torch.cat((meta_prototype_std.cuda(), x_std.cuda())).cuda())
        NTN_out_std = W_out_m_std + V_out_m_std

        compare_input = self.tanh(torch.cat((NTN_out, NTN_out_std)))
        mult_ = F.relu(self.layer1(compare_input))
        mult_ = F.relu(self.layer2(mult_))
        mult_ = self.layer3(mult_)

        add_ = F.relu(self.layer1_add(compare_input))
        add_ = F.relu(self.layer2_add(add_))
        add_ = self.layer3_add(add_)

        recovered_x = torch.mul(original_x, mult_) + add_ ### use back normal x
        z_support, z_query  = self.parse_feature(recovered_x.detach(),is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
          

        

            
