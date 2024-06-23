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





def get_projection_matrix(device, alpha_r, Ur, Sr, alpha_f, Uf, Sf):
    update_dict = OrderedDict()
    for act in Ur.keys():
        ur = Ur[act]
        sr = Sr[act]
        sval_total = (sr**2).sum()
        sval_ratio = (sr**2)/sval_total
        importance_r =  torch.diag( alpha_r *sval_ratio/((alpha_r-1)*sval_ratio+1) )
        mr = torch.mm( ur, importance_r )
        mr = torch.mm( mr, ur.transpose(0,1) )

        uf = Uf[act]
        sf = Sf[act]
        sval_total = (sf**2).sum()
        sval_ratio = (sf**2)/sval_total
        importance_f =  torch.diag( alpha_f *sval_ratio/((alpha_f-1)*sval_ratio+1) )
        mf = torch.mm( uf, importance_f )
        mf = torch.mm( mf, uf.transpose(0,1) )
        I = torch.eye(mf.shape[0]).to(device)
        update_dict[act] =  I  - (mf - torch.mm(mf,mr) )
    return update_dict

def our_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, train_dataset, val_index = None):
    
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
            Ur, Sr = model.get_svd_directions(data, args.our_max_patches)
            break
        # print(Counter(target.tolist()))
        
        for data, target in small_forget_loader:
        # for data, target in forget_loader:
            data, target = data.to(device), target.to(device)
            # Rf = model.get_activations(data)
            Uf, Sf = model.get_svd_directions(data, args.our_max_patches)
            break
        # print(Counter(target.tolist()))
    
    model.eval()
    unlearn_model = model
    group_name = f"{args.group_name}-{args.unlearn_method}"
    
    run = wandb.init(
            # Set the project where this run will be logged
            project=f"Class-{args.dataset}-{args.project_name}-{args.arch}",
            group= group_name,
            name=f"Class-{args.unlearn_class}" ,
            dir = os.environ["LOCAL_HOME"],
            # Track hyperparameters and run metadata
            config= vars(args)
            )
    retain_acc, forget_acc, best_metric = test(model, device, test_loader,args.unlearn_class, args.class_label_names, args.num_classes,
        job_name = args.unlearn_method, set_name="Test Set") 
    wandb.finish()

    for alpha_r in [float(val) for val in args.our_alpha_r.split(",")]: 
        for alpha_f in [float(val) for val in args.our_alpha_f.split(",")]: 
            inference_model = copy.deepcopy(model)
            inference_model.eval()
            run = wandb.init(
                # Set the project where this run will be logged
                project=f"Class-{args.dataset}-{args.project_name}-{args.arch}",
                group= f"{group_name}-{alpha_r}-{alpha_f}",
                name=f"Class-{args.unlearn_class}" ,
                dir = os.environ["LOCAL_HOME"],
                # Track hyperparameters and run metadata
                config= vars(args)
            )
            inference_model.project_weights(get_projection_matrix(device,alpha_r, Ur, Sr, alpha_f, Uf, Sf))
            inference_model.eval()
            retain_acc, forget_acc, metric = test(inference_model, device, test_loader,args.unlearn_class, args.class_label_names, args.num_classes,
                job_name = args.unlearn_method, set_name="Test Set") 
            if metric > best_metric:
                best_metric = metric
                unlearn_model = inference_model
            wandb.finish()
    
    return unlearn_model


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
    parser.add_argument('--unlearn-method', type=str, default="our",
                        help='')
                        



    parser.add_argument('--our-alpha-r', type=str, default="100",
                        help='')
    parser.add_argument('--our-alpha-f', type=str, default="3",
                        help='')  
    parser.add_argument('--our-samples', type=int, default=900,
                        help='')  
    parser.add_argument('--our-max-patches', type=int, default=50000,
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

    
    # _, val_dataset= torch.utils.data.random_split(dataset1, [len(dataset1)-args.val_set_samples, args.val_set_samples])
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
    

    

    model.load_state_dict(torch.load( f"./pretrained_models/{args.dataset}_{args.arch}.pt") )
    print("Model Loaded")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    unlearn_model = our_unlearn(args,
            model,
            device,
            retain_loader,
            forget_loader,
            val_loader if args.val_set_mode else train_loader,
            test_loader,
            dataset1,
            val_index )    
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)     

    run = wandb.init(
            # Set the project where this run will be logged
            project=f"Class-{args.dataset}-{args.project_name}-{args.arch}",
            group= group_name,
            name=f"Class-{args.unlearn_class}" ,
            dir = os.environ["LOCAL_HOME"],
            # Track hyperparameters and run metadata
            config= vars(args)
            )
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
        
    
    
    
    wandb.log({"run_time":elapsed_time_ms})
    print("-"*40)
    print(f"Model Unlearnt with {args.unlearn_method}")
    print("-"*40)
    
    unlearn_model.eval()
    model.eval()
    # Saves time for ImageNet
    # train_retain_acc, train_forget_acc, train_metric = test(unlearn_model, device, train_loader, args.unlearn_class, args.class_label_names, args.num_classes,
    #     job_name = args.unlearn_method, set_name="Final Train Set")
    test(model, device, test_loader,args.unlearn_class, args.class_label_names, args.num_classes,
        job_name = args.unlearn_method, set_name="Final Original Result") 
    test(unlearn_model, device, test_loader,args.unlearn_class, args.class_label_names, args.num_classes,
        job_name = args.unlearn_method, set_name="Final Unlearn Result") 

    wandb.finish()

    if args.save_model:
        torch.save(unlearn_model.state_dict(), f"./pretrained_models/baseline/{args.dataset}_{args.arch}_{args.unlearn_method}_{','.join([str(val) for val in args.unlearn_class])}.pt")


if __name__ == '__main__':
    main()