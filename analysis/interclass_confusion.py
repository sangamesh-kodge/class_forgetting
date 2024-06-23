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
from utils import  get_retain_forget_partition, get_dataset, get_model,  SVC_MIA, metric_function
import copy
import os
import random 
from scrub_thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from scrub_thirdparty.repdistiller.distiller_zoo import DistillKL
from scrub_thirdparty.repdistiller.helper.loops import train_distill, validate
from torch import nn
from collections import OrderedDict, Counter, defaultdict

from multiprocessing import Pool 
from multiprocessing import Process, Value, Array
from functools import partial
import torch.multiprocessing as mp

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['ieee', 'science', 'grid'])
# textwidth = 3.31314
# aspect_ratio = 6/8
# scale = 1.5
# width = textwidth * scale
# height = width * aspect_ratio
# fig = plt.figure(figsize=(width, height))




    
def test(model, device, data_loader,  unlearn_class_list, class_label_names, confusion_class_list=None, non_confusion_class_list=None, num_classes =10, \
                    plot_cm=False, job_name = "baseline", verbose=True, set_name = "Val Set"):
    model.eval()
    sample_loss = 0
    correct = 0
    cm = np.zeros((num_classes,num_classes))
    dict_classwise_acc={}
    dict_classwise_loss=defaultdict(float)
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            sample_loss = F.nll_loss(output, target, reduction='none')#.item()  # sum up batch loss
            for i in range(num_classes):
                dict_classwise_loss[i] +=  torch.where(target == i, sample_loss, torch.zeros_like(sample_loss)).sum()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            cm+=confusion_matrix(target.cpu().numpy(),pred.squeeze(-1).cpu().numpy(), labels=[val for val in range(num_classes)])
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss = sum(list(dict_classwise_loss.values()))
    total_loss /= len(data_loader.dataset)    
    classwise_acc = cm.diagonal()/cm.sum(axis=1)
    for i in range(0,num_classes):
        dict_classwise_acc[class_label_names[i]] =  100*classwise_acc[i]
    if unlearn_class_list:
        forget_loss = sum([float(dict_classwise_loss[key]) for key in dict_classwise_loss if key in unlearn_class_list])/ sum([float(val) for i, val in enumerate(cm.sum(axis=1)) if i in unlearn_class_list])
        forget_acc = sum([float(val) for i, val in enumerate(cm.diagonal()) if i in unlearn_class_list])/ sum([float(val) for i, val in enumerate(cm.sum(axis=1)) if i in unlearn_class_list])
        unlearn_class_name = [name for i, name in enumerate(class_label_names) if i in unlearn_class_list]
    else:
        forget_acc = 0
        forget_loss = 0
        unlearn_class_name = []
    
    confusion = 100*cm[unlearn_class_list[0]]/ sum(list(cm[unlearn_class_list[0]]))
    if confusion_class_list is None:
        #select top 5 confusing classes
        confusion_class_list = np.argsort(confusion)
        confusion_class_list = [val for val in np.flip(confusion_class_list) if (val!=unlearn_class_list[0] and 151<=val<=268)][:5]
        non_confusion_class_list = [val for val in confusion_class_list if (val!=unlearn_class_list[0] and 151<=val<=268 )][:5]
    confusion_acc = {}
    for i, class_idx in enumerate(confusion_class_list):
        confusion_acc[f"{i}-{class_idx}-{class_label_names[class_idx]}"] = 100*cm.diagonal()[class_idx]/ cm.sum(axis=1)[class_idx]
    non_confusion_acc = {}
    for i, class_idx in enumerate(non_confusion_class_list):
        non_confusion_acc[f"{i}-{class_idx}-{class_label_names[class_idx]}"] = 100*cm.diagonal()[class_idx]/ cm.sum(axis=1)[class_idx]

    not_retain_class_list = unlearn_class_list 
    retain_acc = sum([float(val) for i, val in enumerate(cm.diagonal()) if i not in not_retain_class_list])/ sum([float(val)  for i, val in enumerate(cm.sum(axis=1)) if i not in not_retain_class_list])
    retain_loss = sum([float(dict_classwise_loss[key]) for key in dict_classwise_loss if key not in not_retain_class_list])/ sum([float(val) for i, val in enumerate(cm.sum(axis=1)) if i not in not_retain_class_list])
    metric = metric_function(retain_acc,forget_acc)
    if plot_cm:
        fig,ax = plt.subplots()
        fig.set_size_inches(10,10)
        plt.style.use("seaborn-talk")
        cs = sns.color_palette("muted")
        short_labels = []
        for label in class_label_names:
            short_labels.append(label[0:4])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels )
        disp.plot(cmap='Greens', values_format='.0f')
        for labels in disp.text_.ravel():
            labels.set_fontsize(20)
        # plt.title(f"{set_name} Class removed {unlearn_class_name} : {job_name}")
        plt.xticks(rotation=45, ha='right', fontsize=25)
        plt.yticks(rotation=45, fontsize=25)
        plt.xlabel("Predicted Labels", fontsize=30)
        plt.ylabel("True Lables", fontsize=30)
        plt.tight_layout()
        plt.savefig(f"./class_removal/images/cm_{set_name}_{unlearn_class_name}_{job_name}.pdf")
        
        # plt.show()    

    print(f'{set_name}: Average loss: {total_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({(100. * correct / len(data_loader.dataset)):.0f}%)')
    if verbose:
        print('-'*30)
        print(f"{set_name} Confusion Matrix \n{cm}")
        print('-'*30)  
        print(f"{set_name} Class Wise ACC \n{dict_classwise_acc}")
        print('-'*30) 
        print(f"{set_name} Retain class acc(loss): {100*retain_acc}({retain_loss}) - Forget class acc(loss): {100*forget_acc}({forget_loss})\n")
        wandb.log({ #"test_loss":test_loss, 
                    f"{set_name}-Confusion/Acc":confusion_acc,
                    f"{set_name}-Confusion/Acc-avg":sum(list(confusion_acc.values()))/len(confusion_class_list),
                    f"{set_name}-NON-Confusion/Acc":non_confusion_acc,
                    f"{set_name}-NON-Confusion/Acc-avg":sum(list(non_confusion_acc.values()))/len(non_confusion_class_list),
                    f"{set_name}/acc-forget":100. * forget_acc,
                    f"{set_name}/acc-retained":100. * retain_acc,
                    f"{set_name}/metric":100*(metric) ,
                    f"{set_name}/loss-forget":forget_loss,
                    f"{set_name}/loss-retained":retain_loss,
                    f"{set_name}/class-acc/test_acc":dict_classwise_acc,
                    f"{set_name}/class-loss/test_loss":dict_classwise_loss
                    }
                    )
    return retain_acc, forget_acc, metric, confusion_class_list, non_confusion_class_list, classwise_acc, confusion




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
    group_name = f"{args.group_name}-{args.unlearn_method}-{args.our_alpha_r}-{args.our_alpha_f}"
    
    run = wandb.init(
            # Set the project where this run will be logged
            project=f"Class-{args.dataset}-{args.project_name}-{args.arch}",
            group= group_name,
            name=f"Class-{args.unlearn_class}" ,
            dir = os.environ["LOCAL_HOME"],
            # Track hyperparameters and run metadata
            config= vars(args)
            )
    model.load_state_dict(torch.load( f"./pretrained_models/{args.dataset}_{args.arch}.pt") )
    print("Model Loaded")
    confusion_class_list=None
    non_confusion_class_list =None
    model.eval()
    retain_acc, forget_acc, metric, confusion_class_list, non_confusion_class_list, org_classwise_acc, org_confusion = test(model, device, test_loader,args.unlearn_class, args.class_label_names,confusion_class_list,  non_confusion_class_list, args.num_classes,
        job_name = args.unlearn_method, set_name="Original Test Set") 

    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    unlearn_model = our_unlearn[args.unlearn_method](
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
    retain_acc, forget_acc, metric, confusion_class_list, non_confusion_class_list, unl_classwise_acc, unl_confusion = test(unlearn_model, device, test_loader,args.unlearn_class, args.class_label_names,confusion_class_list,  non_confusion_class_list, args.num_classes,
        job_name = args.unlearn_method, set_name="Unlearnt Test Set") 
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
        
    
    
    
    

    print("-"*40)
    print(f"Confusing classes for {args.class_label_names[args.unlearn_class[0]]} are {[args.class_label_names[val] for val in confusion_class_list]}")
    print(f"Non Confusing classes for {args.class_label_names[args.unlearn_class[0]]} are {[args.class_label_names[val] for val in non_confusion_class_list]}")
    print("-"*40)
    diff_acc =org_classwise_acc -  unl_classwise_acc
    sorted_index_acc = [val for val in np.argsort(diff_acc) if val!=args.unlearn_class[0]]# highest drop to lowest drop. 
    print("-"*40)
    print(f"Lowest Accuracy difference for {args.class_label_names[args.unlearn_class[0]]} ")
    for i in range(10):
        index = sorted_index_acc[i]
        print(f"CLASS-{index}-{args.class_label_names[index]} : Before-{org_classwise_acc[index]},After-{unl_classwise_acc[index]},Confusion-{org_confusion[index]}")
    print("-"*40)
    print("-"*40)
    print(f"Highest Accuracy difference for {args.class_label_names[args.unlearn_class[0]]} ")
    for i in range(10):
        index = sorted_index_acc[len(sorted_index_acc)-1-i]
        print(f"CLASS-{index}-{args.class_label_names[index]} : Before-{org_classwise_acc[index]},After-{unl_classwise_acc[index]},Confusion-{org_confusion[index]}")
    print("-"*40)

    # Sorted dog breeds
    sorted_index_dog_acc = [val for i, val in enumerate(sorted_index_acc) if (val!=200 and 151<=val<=268)]
    print("-"*40)
    print(f"Fixed- Confusing Breeds Accuracies {args.class_label_names[args.unlearn_class[0]]} ")
    for i in range(20):
        index = sorted_index_dog_acc[i]
        print(f"CLASS-{index}-{args.class_label_names[index]} : Before-{org_classwise_acc[index]},After-{unl_classwise_acc[index]},Confusion-{org_confusion[index]}")
    print("-"*40)
    print("-"*40)
    print(f"Fixed- Non Confusing Breeds Accuracies {args.class_label_names[args.unlearn_class[0]]} ")
    for i in range(20):
        index = sorted_index_dog_acc[len(sorted_index_dog_acc)-1-i]
        print(f"CLASS-{index}-{args.class_label_names[index]} : Before-{org_classwise_acc[index]},After-{unl_classwise_acc[index]},Confusion-{org_confusion[index]}")
    print("-"*40)

    wandb.finish()
    if args.save_model:
        torch.save(model.state_dict(), f"./pretrained_models/baseline/{args.dataset}_{args.arch}_{args.unlearn_method}_{','.join([str(val) for val in args.unlearn_class])}.pt")


if __name__ == '__main__':
    main()