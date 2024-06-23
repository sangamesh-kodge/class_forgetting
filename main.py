### Source -> https://github.com/pytorch/examples/blob/main/mnist/main.py


from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from utils import  get_retain_forget_partition, get_dataset, get_model, test, SVC_MIA
import copy
import os
import random 
from scrub_thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from scrub_thirdparty.repdistiller.distiller_zoo import DistillKL
from scrub_thirdparty.repdistiller.helper.loops import train_distill, validate
from torch import nn
from collections import OrderedDict, Counter

from multiprocessing import Pool 
from multiprocessing import Process, Value, Array
from functools import partial
import torch.multiprocessing as mp

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['ieee', 'science', 'grid'])
textwidth = 3.31314
aspect_ratio = 6/8
scale = 1.5
width = textwidth * scale
height = width * aspect_ratio
fig = plt.figure(figsize=(width, height))
def likelihood(score, mean, var):
    nll = - ( ( (score - mean)**2) / (2 * (var ** 2) ) ) - 0.5 * torch.log(var ** 2) - 0.5 * torch.log(4 * torch.acos(torch.zeros(1)))
    likelihood_val = torch.exp(nll)
    return likelihood_val

def get_likelihood_ratio(test_features, model_parameters):
    in_likelihood = likelihood(
        test_features, 
        model_parameters['mean_unlearn'].view(-1,1), 
        model_parameters['std_unlearn'].view(-1,1) + 1e-32)

    out_likelihood = likelihood(
        test_features, 
        model_parameters['mean_retrain'].view(-1,1), 
        model_parameters['std_retrain'].view(-1,1)+ 1e-32)
    likelihood_ratio = in_likelihood / (out_likelihood + 1e-32)
    return likelihood_ratio



def get_salun_mask(args, model, device, forget_loader):
    
    if args.unlearn_method != "salun":
        return None 
    mask = {}
    for name, param in model.named_parameters():
        mask[name] = 0
    model.train()
    for data, target in forget_loader:
            model.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        mask[name] += param.grad.data
    with torch.no_grad():
        for name in mask:
            mask[name] = torch.abs_(mask[name])

    
    sorted_dict_positions = {}
    hard_dict = {}

    # Concatenate all tensors into a single tensor
    all_elements = - torch.cat([tensor.flatten() for tensor in mask.values()])

    # Calculate the threshold index for the top 10% elements
    threshold_index = int(len(all_elements) * args.salun_threshold)

    # Calculate positions of all elements
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    start_index = 0
    for key, tensor in mask.items():
        num_elements = tensor.numel()
        # tensor_positions = positions[start_index: start_index + num_elements]
        tensor_ranks = ranks[start_index : start_index + num_elements]

        sorted_positions = tensor_ranks.reshape(tensor.shape)
        sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        hard_dict[key] = threshold_tensor
        start_index += num_elements    
    return hard_dict


def salun_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, optimizer, epochs, **kwargs):
    mask = get_salun_mask(args, model, device, forget_loader)     
    model.train()
    train_loss= 0
    train_forget_acc =100.0
    steps = 0
    # Random relabeling
    forget_dataset = copy.deepcopy(forget_loader.dataset)
    forget_dataset.update_labels(np.random.randint(0, args.num_classes, len(forget_dataset) ))
    forget_dataset, _ = torch.utils.data.random_split(forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples])
    
    #Subsample retain set. 
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples])

    train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])

    salun_train_loader= torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True)

    max_steps = epochs*(len(train_dataset)//args.batch_size)

    for epoch in range(epochs):
        for data, target in salun_train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            steps+=1
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            optimizer.step()
            train_loss += loss.detach().item()

 
            if args.dry_run:
                break        
        if args.dry_run:
            break    
    return model


def getRetrainLayers(m, name, ret):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        ret.append((m, name))
        #print(name)
    for child_name, child in m.named_children():
        ret = getRetrainLayers(child, f'{name}.{child_name}', ret)
    return ret

def _reinit(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def resetFinalResnet(model, num_retrain, reinit=True):
    for param in model.parameters():
        param.requires_grad = False
    done = 0
    ret = getRetrainLayers(model, 'M', [])
    ret.reverse()
    for idx in range(len(ret)):
        if reinit:
            if isinstance(ret[idx][0], nn.Conv2d) or isinstance(ret[idx][0], nn.Linear):
                _reinit(ret[idx][0])
                logger.info(f'Reinitialized layer: {ret[idx][1]}')
        if isinstance(ret[idx][0], nn.Conv2d) or isinstance(ret[idx][0], nn.Linear):
            done += 1
        for param in ret[idx][0].parameters():
            param.requires_grad = True
        if done >= num_retrain:
            break
    return model
    

def goel_last_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, optimizer, epochs, **kwargs):
    train_loss= 0
    train_forget_acc =100.0
    steps = 0    
    #Subsample retain set. 
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples])
    
    goel_train_loader= torch.utils.data.DataLoader( retain_dataset, batch_size=args.batch_size, shuffle=True)
  
    model.train()
    model = resetFinalResnet(model, 1, reinit=args.goel_exact)
    
    for epoch in range(epochs):
        for data, target in goel_train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            steps+=1
            optimizer.step()
            train_loss += loss.detach().item()
                    
            if args.dry_run:
                break        
        if args.dry_run:
            break  
    return model



def calc_importance(model, device, optimizer, dataloader) :
        """
        https://github.com/if-loops/selective-synaptic-dampening
        """
        # criterion = torch.nn.CrossEntropyLoss()
        importances = dict(
                            [
                                (k, torch.zeros_like(p, device=p.device))
                                for k, p in model.named_parameters()
                            ]
                        )
                        
        for x, y  in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.nll_loss(out, y)
            # loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances



def ssd_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, optimizer, **kwargs):
    #Subsample dataset. 
    forget_dataset = forget_loader.dataset
    forget_dataset, _ = torch.utils.data.random_split(forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples])
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples])
    ssd_retain_loader= torch.utils.data.DataLoader( retain_dataset, batch_size=args.batch_size , shuffle=True)
    train_dataset = torch.utils.data.ConcatDataset([forget_loader.dataset,retain_dataset])
    ssd_train_loader= torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True)

    # Hyperparameters
    lower_bound = 1  # unused
    exponent = 1  # unused
    dampening_constant =  args.ssd_lambda # Lambda from paper
    selection_weighting = args.ssd_alpha # Alpha from paper

    model.eval() # to ensure batch statistics do not change

    # Calculation of the forget set importances
    forget_importance = calc_importance(model, device, optimizer, forget_loader)
    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importance = calc_importance(model, device, optimizer,ssd_train_loader)
    # Dampen selected parameters
    with torch.no_grad():
        for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
            model.named_parameters(),
            original_importance.items(),
            forget_importance.items(),
        ):
            # Synapse Selection with parameter alpha
            oimp_norm = oimp.mul(selection_weighting)
            locations = torch.where(fimp > oimp_norm)

            # Synapse Dampening with parameter lambda
            weight = ((oimp.mul(dampening_constant)).div(fimp)).pow(
                exponent
            )
            update = weight[locations]
            # Bound by 1 to prevent parameter values to increase.
            min_locs = torch.where(update > lower_bound)
            update[min_locs] = lower_bound
            p[locations] = p[locations].mul(update)  
    return model


def scrub_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, **kwargs):
    #Subsample dataset. 
    forget_dataset = forget_loader.dataset
    forget_dataset, _ = torch.utils.data.random_split(forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples])
    scrub_forget_loader= torch.utils.data.DataLoader( forget_dataset, batch_size=args.scrub_del_bsz , shuffle=True)
    
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples])
    scrub_retain_loader= torch.utils.data.DataLoader( retain_dataset, batch_size=args.scrub_sgda_bsz , shuffle=True)

    ### Include in argsparse later!
    args.optim = 'sgd'
    args.gamma = 0.99
    args.alpha = 0.001
    args.beta = 0
    args.smoothing = 0.0
    args.msteps = args.scrub_msteps
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 4
    args.distill = 'kd'

    args.sgda_epochs = args.scrub_epochs
    args.sgda_learning_rate = args.lr #0.0005
    args.lr_decay_epochs = [3,5,9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9

    model_t = copy.deepcopy(model)
    model_s = copy.deepcopy(model)
    beta = 0.1
    def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
        1 - beta) * averaged_model_parameter + beta * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(
        model_s, avg_fn=avg_fn)
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                            lr=args.sgda_learning_rate,
                            momentum=args.sgda_momentum,
                            weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam": 
        optimizer = optim.Adam(trainable_list.parameters(),
                            lr=args.sgda_learning_rate,
                            weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = optim.RMSprop(trainable_list.parameters(),
                            lr=args.sgda_learning_rate,
                            momentum=args.sgda_momentum,
                            weight_decay=args.sgda_weight_decay)
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        swa_model.cuda()       
    model_s.train()
        
    for epoch in range(1, args.sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)
        # DO NOT EVAL DURING Unlearning
        # model_s.eval()
        # train_retain_acc, train_forget_acc, train_metric = test(model_s, device, train_loader, args.unlearn_class, args.class_label_names, args.num_classes,
        #     job_name = args.unlearn_method, set_name="Train Set")
        # test(model_s, device, test_loader,args.unlearn_class, args.class_label_names, args.num_classes,
        #     job_name = args.unlearn_method, set_name="Test Set") 
        maximize_loss = 0
        if epoch <= args.msteps:
            maximize_loss = train_distill(epoch, scrub_forget_loader, module_list, swa_model, criterion_list, optimizer, args, "maximize")
        train_acc, train_loss = train_distill(epoch, scrub_retain_loader, module_list, swa_model, criterion_list, optimizer, args, "minimize")
        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)
        # print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
    return model_s




# defining the noise structure
class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad = True)
        
    def forward(self):
        return self.noise

def tarun_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, train_dataset, val_index = None, **kwargs):
    # Hyperparam
    batch_size = args.batch_size
    impair_lr = args.tarun_impair_lr
    repair_lr =  args.lr

    # get  images of each class other than unlearning class
    index_list = []
    targets = np.array(train_dataset.targets)
    for i in range(args.num_classes):
        if i !=  args.unlearn_class[0]:
            class_i_index = np.intersect1d(np.where(i == targets)[0], val_index)
            index_list.extend(class_i_index[:int(args.tarun_samples_per_class) ])
    small_retain_loader = torch.utils.data.DataLoader( torch.utils.data.Subset(train_dataset, index_list), batch_size=batch_size , shuffle=True)
    
    # Learn Noise
    classes_to_forget = args.unlearn_class
    noises = {}
    for cls_num in classes_to_forget:
        print("Optiming loss for class {}".format(cls_num))
        if "imagenet" in args.dataset:
            noises[cls_num] = Noise(batch_size, 3, 224, 224).cuda()
        else:
            noises[cls_num] = Noise(batch_size, 3, 32, 32).cuda()
        opt = torch.optim.Adam(noises[cls_num].parameters(), lr = 0.1)

        num_epochs = 5
        num_steps = 8
        class_label = cls_num
        for epoch in range(num_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls_num]()
                labels = torch.zeros(batch_size).cuda()+class_label
                outputs = model(inputs)
                loss = -F.nll_loss(outputs, labels.long()) + 0.1*torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            print("Loss: {}".format(np.mean(total_loss)))

    batch_size = 256
    num_batches = 20
    # class_num = 0
    noisy_data = []
    for cls_num in classes_to_forget:
        for i in range(num_batches):
            batch = noises[cls_num]().cpu().detach()
            for i in range(batch[0].size(0)):
                noisy_data.append((batch[i], torch.tensor(cls_num)))

    other_samples = []
    retain_samples =  small_retain_loader.dataset
    for i in range(len( retain_samples)):
        other_samples.append((retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][1])))
    noisy_data += other_samples
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=batch_size, shuffle = True)

    # -> Impair step.
    print("-"*100)
    print("Impair step on Forget Model")
    print("-"*100)
    optimizer = torch.optim.Adam(model.parameters(), lr = impair_lr)
    for epoch in range(args.epochs_or_steps):  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(),torch.tensor(labels).cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(),dim=1)
            assert out.shape==labels.shape
            running_acc += (labels==out).sum().item()
        print(f"Train loss {epoch+1}: {running_loss/len(noisy_data)},Train Acc:{running_acc*100/len(noisy_data)}%")

    # -> Repair step.
    print("-"*100)
    print("Repair step on Forget Model")
    print("-"*100)
    heal_loader = torch.utils.data.DataLoader(other_samples, batch_size=256, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr = repair_lr)


    for epoch in range(args.epochs_or_steps):  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(heal_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(),torch.tensor(labels).cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(),dim=1)
            assert out.shape==labels.shape
            running_acc += (labels==out).sum().item()
        print(f"Train loss {epoch+1}: {running_loss/len(other_samples)},Train Acc:{running_acc*100/len(other_samples)}%")

    return model 

def unlearn_naive(args, model, device, retain_loader, forget_loader, train_loader, test_loader, optimizer, max_steps, **kwargs):
    #Subsample dataset. 
    method = args.unlearn_method
    clip=args.grad_norm_clip
    forget_dataset = forget_loader.dataset
    forget_dataset, _ = torch.utils.data.random_split(forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples])
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples])
    naive_retain_loader= torch.utils.data.DataLoader( retain_dataset, batch_size=args.batch_size , shuffle=True)

    model.train()
    train_loss= 0
    train_forget_acc =100.0
    forget_iterator = iter(forget_loader)
    cur_step = 0
    while (cur_step <= max_steps):
        for batch_idx, (data, target) in enumerate(naive_retain_loader):
            optimizer.zero_grad()
            if "ascent" in method and train_forget_acc>1e-3:
                try:
                    data_f, target_f = next(forget_iterator)
                except:
                    forget_iterator = iter(forget_loader)
                    data_f, target_f = next(forget_iterator)
                data_f, target_f = data_f.to(device), target_f.to(device)
                output = model(data_f)
                loss = -1.0 * F.nll_loss(output, target_f)
                loss.backward()
                # for param in model.parameters():
                #     if param.grad is not None:
                #         param.grad.data *= -1.0
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            if "descent" in method:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
            cur_step+=1

            optimizer.step()
            train_loss += loss.detach().item()
            # DO NOT EVAL DURING Unlearning
            if cur_step % args.log_interval == 0:        
                model.eval()
                count = 0
                for data, target in forget_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    count += torch.sum( target ==output.argmax(dim=1, keepdim=True) ).item()
                train_forget_acc = count/len(forget_loader)
                
                model.train()

            if args.dry_run:
                break
    
    
    return model
    

def get_projection_matrix(device, Mr, Mf):
    update_dict = OrderedDict()
    for act in Mr.keys():
        mr = Mr[act] 
        mf = Mf[act] 
        I = torch.eye(mf.shape[0]).to(device)
        update_dict[act] =  I  - (mf - torch.mm(mf,mr) )
    return update_dict

def our_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, train_dataset, val_index = None, **kwargs):
    
    model.eval() # Ensures batch statistics do not change. 
    # get 100 images of each class other than unlearning class
    index_list = []
    targets = np.array(train_dataset.targets)
    for i in range(args.num_classes):
        if i !=  args.unlearn_class[0]:
            class_i_index = np.intersect1d(np.where(i == targets)[0], val_index)
            index_list.extend(class_i_index[:int(args.our_samples//(args.num_classes-1))])
    small_retain_loader = torch.utils.data.DataLoader( torch.utils.data.Subset(train_dataset, index_list), batch_size=args.our_samples , shuffle=True)
    small_forget_loader = torch.utils.data.DataLoader( forget_loader.dataset , batch_size=args.our_samples , shuffle=True)
    
    with torch.no_grad():
        for data, target in small_retain_loader:
        # for data, target in retain_loader
            data, target = data.to(device), target.to(device)
            # Rr = model.get_activations(data)
            Mr = model.get_scaled_projections(data, args.our_alpha_r, args.our_max_patches)
            break
        # print(Counter(target.tolist()))
        
        for data, target in small_forget_loader:
        # for data, target in forget_loader:
            data, target = data.to(device), target.to(device)
            # Rf = model.get_activations(data)
            Mf = model.get_scaled_projections(data, args.our_alpha_f, args.our_max_patches)
            break
        # print(Counter(target.tolist()))
    
    # model.project_weights(get_projection_matrix(device=device, Rr=Rr, Rf=Rf, alpha_r = args.our_alpha_r, alpha_f = args.our_alpha_f, update_dict=OrderedDict()) )
    model.project_weights(get_projection_matrix(device, Mr, Mf))
    return model


def train(args, model, device, train_loader, optimizer, epoch, mode = "descent", clip=None):
    model.train()
    train_loss= 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if mode == "ascent":
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data *= -1.0
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.detach().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"train_loss":loss.item() })
            if args.dry_run:
                break
    return train_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs-or-steps', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                        help='')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='LR',
                        help='')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.7) after 50 epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-train-transform', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--arch', type=str, default='vgg11_bn',
                        help='')
    parser.add_argument('--data-path', type=str, default='../data/',
                        help='')
    parser.add_argument('--val-set-mode', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--val-set-samples', type=int, default=10000, metavar='EPS',
                        help='')
    ### Unlearn parameters

    parser.add_argument('--num-retain-samples', type=int, default=45000,
                        help='') 
    parser.add_argument('--num-forget-samples', type=int, default=5000,
                        help='') 
    parser.add_argument('--grad-norm-clip', type=float, default=None,
                        help='')                        
    parser.add_argument('--unlearn-class', type=str, default="",
                        help='')
    parser.add_argument('--unlearn-method', type=str, default="retrain",
                        help='')

    parser.add_argument('--salun-threshold', type=float, default=0.1,
                        help='')

    parser.add_argument('--goel-exact', action="store_true", default=False,
                        help='')

    parser.add_argument('--ssd-lambda', type=float, default=1,
                        help='')
    parser.add_argument('--ssd-alpha', type=float, default=10,
                        help='')

    parser.add_argument('--scrub-del-bsz', type=int, default=512,
                        help='')
    parser.add_argument('--scrub-sgda-bsz', type=int, default=64,
                        help='')
    parser.add_argument('--scrub-msteps', type=int, default=2,
                        help='')
    parser.add_argument('--scrub-epochs', type=int, default=3,
                        help='')

    parser.add_argument('--our-alpha-r', type=int, default=100,
                        help='')
    parser.add_argument('--our-alpha-f', type=int, default=3,
                        help='')  
    parser.add_argument('--our-samples', type=int, default=900,
                        help='')  
    parser.add_argument('--our-max-patches', type=int, default=10000,
                        help='')  


    parser.add_argument('--tarun-impair-lr', type=float, default=2e-4,
                        help='')
    parser.add_argument('--tarun-samples-per-class', type=int, default=1000,
                        help='') 
    ### wandb parameters
    parser.add_argument('--project-name', type=str, default='baseline',
                        help='')
    parser.add_argument('--group-name', type=str, default='final',
                        help='')
    parser.add_argument('--multiclass', action='store_true', default=False,
                        help='For Saving the current Model')  
    parser.add_argument('--class-names', type=str, default=None,
                        help='')   
    parser.add_argument('--do-mia',action='store_true', default=False,
                        help='')   
    parser.add_argument('--do-mia-ulira',action='store_true', default=False,
                        help='')   
    parser.add_argument('--plot-mia-roc',action='store_true', default=False,
                        help='') 
    # args  = add_additional_args(parser)
    args = parser.parse_args()
    args.train_transform = not args.no_train_transform
    if args.unlearn_class:
        args.unlearn_class=[int(val) for val in args.unlearn_class.split(",")]
    else:
        args.unlearn_class = []

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 16,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # Load full dataset.
    dataset1, dataset2 = get_dataset(args)
    if args.class_names is not None and args.multiclass: 
        args.unlearn_class = [args.class_label_names.index(val) for val in args.class_names.split(",")] 
        print(args.unlearn_class)
    # Partition into retain and forget dataset.
    retain_dataset, forget_dataset = get_retain_forget_partition(args, dataset1, args.unlearn_class)
    
    
    val_index= np.arange(len(dataset1))
    if args.val_set_mode:
        np.random.shuffle(val_index)
        val_index = val_index[:args.val_set_samples]
    val_dataset = torch.utils.data.Subset(dataset1, val_index) 


    # Check trained model!
    model = get_model(args, device)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **train_kwargs)
    retain_loader = torch.utils.data.DataLoader(retain_dataset,**train_kwargs)
    forget_loader = torch.utils.data.DataLoader(forget_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if os.path.exists(f"./pretrained_models/baseline/{args.dataset}_{args.arch}_{args.unlearn_method}_{','.join([str(val) for val in args.unlearn_class])}.pt") and args.save_model:
        raise FileExistsError
    group_name = f"{args.group_name}-{args.unlearn_method}"
    if args.unlearn_method == "ssd":
        group_name = f"{group_name}-{args.ssd_lambda}-{args.ssd_alpha}"
    elif args.unlearn_method == "our":
        group_name = f"{group_name}-{args.our_alpha_r}-{args.our_alpha_f}"
    elif args.unlearn_method == "tarun":
        group_name = f"{group_name}-{args.lr}-{args.tarun_impair_lr}-{args.tarun_samples_per_class}"
    elif args.unlearn_method == "scrub":
        group_name = f"{group_name}-{args.scrub_del_bsz}-{args.scrub_sgda_bsz}-{args.lr}"
    elif args.unlearn_method == "goel":
        if args.goel_exact:
            group_name = f"{group_name}-exact"
    elif args.unlearn_method == "salun":
        group_name = f"{group_name}-{args.lr}-{args.salun_threshold}"
    else:
        group_name = f"{group_name}-{args.lr}"

    run = wandb.init(
            # Set the project where this run will be logged
            project=f"Class-{args.dataset}-{args.project_name}-{args.arch}",
            group= group_name,
            name=f"Class-{args.unlearn_class}" ,
            dir = os.environ["LOCAL_HOME"],
            # Track hyperparameters and run metadata
            config= vars(args)
            )

    # do what unlearn method requires here.
    if args.unlearn_method == "retrain":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.gamma)
        # Run some things here
        for epoch in range(1, args.epochs_or_steps + 1):
            train_loss = train(args, model, device, retain_loader, optimizer, epoch, "descent")
            if args.val_set_mode :
                train_retain_acc, train_forget_acc, train_metric = test(model, device, val_loader,args.unlearn_class, args.class_label_names, args.num_classes,
                    job_name = args.unlearn_method, set_name="Val Set")
            else:
                train_retain_acc, train_forget_acc, train_metric = test(model, device, train_loader,args.unlearn_class, args.class_label_names, args.num_classes,
                    job_name = args.unlearn_method, set_name="Train Set")
            test(model, device, test_loader,args.unlearn_class, args.class_label_names, args.num_classes,
                job_name = args.unlearn_method, set_name="Test Set")
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss)
                else:
                    scheduler.step()
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        unlearn_model = model
    else:
        unlear_func = {
            "rl" : salun_unlearn,
            "salun": salun_unlearn,
            "goel":goel_last_unlearn,
            "ssd":ssd_unlearn,
            "scrub":scrub_unlearn,
            "our":our_unlearn,
            "tarun":tarun_unlearn,
            "grad_ascent'" : unlearn_naive, # NegGrad
            "grad_descent'" : unlearn_naive, # Fine-tune on Retain Set
            "grad_ascent_descent" : unlearn_naive # NegGrad+
        }
        model.load_state_dict(torch.load( f"./pretrained_models/{args.dataset}_{args.arch}.pt") )
        print("Model Loaded")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        unlearn_model = unlear_func[args.unlearn_method](
                args = args,
                model = model,
                device = device,
                retain_loader= retain_loader,
                forget_loader = forget_loader,
                train_loader = val_loader if args.val_set_mode else train_loader,
                test_loader= test_loader,
                optimizer = optimizer,
                epochs = args.epochs_or_steps,
                max_steps = args.epochs_or_steps,
                train_dataset = dataset1,
                val_index =val_index
        )
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
    wandb.log({"run_time":elapsed_time_ms})
    print("-"*40)
    print(f"Model Unlearnt with {args.unlearn_method}")
    print("-"*40)
    
    unlearn_model.eval()
    # Comment train set evaluation (line below) for saving time for ImageNet.
    train_retain_acc, train_forget_acc, train_metric = test(unlearn_model, device, train_loader, args.unlearn_class, args.class_label_names, args.num_classes,
        job_name = args.unlearn_method, set_name="Final Train Set")
    test(unlearn_model, device, test_loader,args.unlearn_class, args.class_label_names, args.num_classes,
        job_name = args.unlearn_method, set_name="Final Test Set") 
    if args.do_mia_ulira:
        evaluation_result= {}
        
        forget_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=False)
        # Get Paths for in, out models
        model_path_set = set(os.listdir("pretrained_models/mia_ulira_models"))
        in_model_paths = set([val for val in model_path_set if ("_none.pt" in val.lower() and f"{args.dataset}_{args.arch}_" in val.lower())])
        out_model_paths = set([val for val in model_path_set if (f"_{args.unlearn_class[0]}.pt" in val.lower()and f"{args.dataset}_{args.arch}_" in val.lower())])
        reduction_func = torch.nn.Softmax(dim=1)
        # get retrain_scores
        retrain_scores = torch.zeros((len(forget_dataset), len(out_model_paths)))
        for idx,model_path in tqdm(enumerate(out_model_paths), desc='Retrain Set'):
            inference_model =  copy.deepcopy(model)
            inference_model.load_state_dict(torch.load( f"./pretrained_models/mia_ulira_models/{model_path}") )
            inference_model.do_log_softmax = False
            inference_model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(forget_loader):
                    data, target = batch[0].cuda(), batch[1].cuda()
                    output = inference_model(data)
                    logits = inference_model(data)
                    retrain_scores[batch_idx*args.batch_size:batch_idx*args.batch_size+logits.shape[0], idx] = reduction_func(logits).clone().detach().cpu()[:, args.unlearn_class[0]]
        
        # get unlearn_scores
        unlearn_scores = torch.zeros((len(forget_dataset), len(in_model_paths)))
        for idx,model_path in tqdm(enumerate(in_model_paths), desc='Unlearn Set') :
            inference_model =  copy.deepcopy(model)
            inference_model.load_state_dict(torch.load( f"./pretrained_models/mia_ulira_models/{model_path}") )
            inference_model.eval()
            optimizer = optim.SGD(inference_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            inference_model = unlear_func[args.unlearn_method](
                        args = args,
                        model = inference_model,
                        device = device,
                        retain_loader= retain_loader,
                        forget_loader = forget_loader,
                        train_loader = val_loader if args.val_set_mode else train_loader,
                        test_loader= test_loader,
                        optimizer = optimizer,
                        epochs = args.epochs_or_steps,
                        max_steps = args.epochs_or_steps,
                        train_dataset = dataset1,
                        val_index =val_index
            )
            inference_model.do_log_softmax = False
            with torch.no_grad():
                for batch_idx, batch in enumerate(forget_loader):
                    data, target = batch[0].cuda(), batch[1].cuda()
                    logits = inference_model(data)
                    unlearn_scores[batch_idx*args.batch_size:batch_idx*args.batch_size+logits.shape[0], idx] = reduction_func(logits).clone().detach().cpu()[:, args.unlearn_class[0]]
        
        
        # Build the MIA model using the above statistics
        # Logit scaling
        unlearn_scores = torch.log(unlearn_scores / (1 - unlearn_scores + 1e-32) + 1e-32)
        retrain_scores = torch.log(retrain_scores / (1 - retrain_scores + 1e-32) + 1e-32) 

        # Split into train and test sets
        train_unlearn_scores = unlearn_scores[:, :-1]
        train_retrain_scores= retrain_scores[:, :-1]
        val_unlearn_scores = unlearn_scores[:, -1]
        val_retrain_scores = retrain_scores[:, -1]
        
        # Extract Distribution parameters from train scores
        mean_unlearn = train_unlearn_scores.mean(dim=1)
        mean_retrain = train_retrain_scores.mean(dim=1)
        std_unlearn = train_unlearn_scores.std(dim=1)
        std_retrain = train_retrain_scores.std(dim=1)
        mia_parameters = {
            "mean_unlearn":mean_unlearn, 
            "mean_retrain":mean_retrain, 
            "std_unlearn":std_unlearn, 
            "std_retrain":std_retrain
            }

        # Test MIA on test scroes
        concat_val_features = torch.cat( (val_unlearn_scores.view(-1,1), val_retrain_scores.view(-1,1)), dim=1)
        y_true_val = torch.cat( (torch.ones_like(val_unlearn_scores).view(-1,1), torch.zeros_like(val_retrain_scores).view(-1,1)), dim=1)
        y_pred_val = get_likelihood_ratio(concat_val_features, mia_parameters)
        fpr, tpr, thr = roc_curve(y_true_val.flatten().numpy(), y_pred_val.flatten().numpy(), pos_label=1)
        auc = roc_auc_score(y_true_val.flatten().numpy(), y_pred_val.flatten().numpy())
        # Find the optimal threshold: where the sum of FPR and TPR is closest to 1
        # "Balanced accuracy is symmetric. That is, the metric
        # assigns equal cost to false-positives and to false-negatives."
        # - LiRA https://arxiv.org/pdf/2112.03570.pdf
        optimal_idx = np.argmin(np.abs(fpr + tpr - 1))
        optimal_threshold = thr[optimal_idx]

        y_pred_binarized_val = (y_pred_val.flatten().numpy() >= optimal_threshold).astype(int)
        # Calculate Balanced Accuracy
        balanced_val_accuracy = balanced_accuracy_score(y_true_val.flatten().numpy(), y_pred_binarized_val)
        evaluation_result["auc"] = auc
        evaluation_result["threshold"] = optimal_threshold
        evaluation_result["balanced_val_accuracy"] = balanced_val_accuracy
        
        
        # Test the unlearnt model with learnt statistics.
        unlearn_model.eval()
        unlearn_model.do_log_softmax = False
        test_model_scores = torch.zeros((len(forget_dataset) ))
        with torch.no_grad():
            for batch_idx, batch in enumerate(forget_loader):
                data, target = batch[0].cuda(), batch[1].cuda()
                logits = inference_model(data)
                test_model_scores[batch_idx*args.batch_size:batch_idx*args.batch_size+logits.shape[0]] = reduction_func(logits).clone().detach().cpu()[:, args.unlearn_class[0]]
        unlearn_model.do_log_softmax = True
        test_model_scores = torch.log(test_model_scores / (1 - test_model_scores + 1e-32) + 1e-32).view(-1,1)    
        y_true_test = torch.ones_like(test_model_scores).view(-1,1)  # model should predict this as out 
        y_pred_test = get_likelihood_ratio(test_model_scores, mia_parameters)
        y_pred_binarized_test = (y_pred_test.flatten().numpy() >= optimal_threshold).astype(int)
        balanced_test_accuracy = balanced_accuracy_score(y_true_test.flatten().numpy(), y_pred_binarized_test)
        evaluation_result["balanced_test_accuracy"] = balanced_test_accuracy
        print(f"AUC {args.unlearn_method}: {auc*100:.2f}, Acc {balanced_test_accuracy * 100:.2f}")
        if args.plot_mia_roc:
            # Generate 10 unique colors from the 'viridis' colormap
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            plt.loglog(fpr,tpr, label=f'{args.unlearn_method} AUC: {auc*100:.2f} - ACC: {balanced_test_accuracy * 100:.2f}', ls='solid', color=colors[idx], lw=2)
            plt.ylim(bottom=1e-4)
            plt.xlim(left=1e-4)
            plt.loglog(fpr, fpr, ls='dashed', color='gray')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend()
            if not os.path.exists("./images"):
                os.makedirs("./images")
            plt.savefig(os.path.join("./images", f"mia_{args.unlearn_method}_{args.dataset}_{args.arch}_{args.unlearn_class}_roc.png"))
            # plt.show()
        wandb.log({"MIA/ULIRA":evaluation_result})
    if args.do_mia:       
        evaluation_result = {}
        classes_for_mia = args.unlearn_class
        classes_for_mia = list(set(classes_for_mia))         
        model.do_log_softmax  = True
        
        train_retain_dataset, train_forget_dataset, train_retain_index, train_forget_index = get_retain_forget_partition(args, dataset1, classes_for_mia, return_ind = True)
        if args.dataset == "imagenet":
            random.shuffle(train_retain_index)
            small_train_retain_index = train_retain_index[:50000] 
            train_retain_dataset = torch.utils.data.Subset(dataset1, small_train_retain_index)
            train_forget_dataset = torch.utils.data.Subset(dataset1, train_forget_index)
            train_retain_loader = torch.utils.data.DataLoader(train_retain_dataset,**train_kwargs)
            train_forget_loader = torch.utils.data.DataLoader(train_forget_dataset,**train_kwargs)
        else:
            train_retain_loader = torch.utils.data.DataLoader(train_retain_dataset,**test_kwargs)
            train_forget_loader = torch.utils.data.DataLoader(train_forget_dataset,**test_kwargs)


        _, test_forget_dataset = get_retain_forget_partition(args, dataset2, classes_for_mia)
        test_forget_loader = torch.utils.data.DataLoader(test_forget_dataset,**test_kwargs)
        evaluation_result["SVC_MIA_forget_efficacy"] = SVC_MIA(
                        shadow_train=train_retain_loader,
                        shadow_test=train_forget_loader,
                        target_train=None,
                        target_test=test_forget_loader,
                        model=model,
                    )

        wandb.log({"MIA/Simple":evaluation_result})
        
    
    

    wandb.finish()

    if args.save_model:
        if not os.path.exists("./pretrained_models/baseline/"):
            os.makedirs("./pretrained_models/baseline/")
        torch.save(model.state_dict(), f"./pretrained_models/baseline/{args.dataset}_{args.arch}_{args.unlearn_method}_{','.join([str(val) for val in args.unlearn_class])}.pt")


if __name__ == '__main__':
    main()