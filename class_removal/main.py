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
from class_removal.utils import activation_projection_based_unlearning, get_dataset, get_model, get_retain_forget_partition
import random 
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--train-transform', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--arch', type=str, default='vgg11_bn', 
                        help='') 
    parser.add_argument('--data-path', type=str, default='../data/', 
                        help='')
    ### wandb parameters
    parser.add_argument('--project-name', type=str, default='unlearn-Project-Activation', 
                        help='')
    parser.add_argument('--group-name', type=str, default='final', 
                        help='')     
    ### GPM parameters
    parser.add_argument('--projection-type', type=str, default='baseline,I-(Mf-Mi)', 
                        help='')
    parser.add_argument('--mode', type=str, default="baseline,sap", metavar='EPS',
                        help='')
    parser.add_argument('--scale-coff', type=str, default= "1,3,10,30,100", metavar='SCF',
                        help='importance co-efficeint (default: 0)')
    parser.add_argument('--mode-forget', type=str, default=None, metavar='EPS',
                        help='')
    parser.add_argument('--scale-coff-forget', type=str, default= "0.5,0.75,0.9,1,3", metavar='SCF',
                        help='importance co-efficeint (default: 0)')
    parser.add_argument('--forget-samples', type=int, default=1350, metavar='EPS',
                        help='')
    parser.add_argument('--retain-samples', type=int, default=150, metavar='EPS',
                        help='')
    parser.add_argument('--max-samples', type=int, default=50000, metavar='EPS',
                        help='')
    parser.add_argument('--max-batch-size', type=int, default=150, metavar='EPS',
                        help='')
    parser.add_argument('--gpm-eps', type=float, default=0.99, metavar='EPS',
                        help='')
    parser.add_argument('--start-layer', type=str, default="0", metavar='EPS',
                        help='')
    parser.add_argument('--end-layer', type=str, default="0", metavar='EPS',
                        help='')    
    parser.add_argument('--projection-location', type=str, default="pre", metavar='EPS',
                        help='') 
    parser.add_argument('--val-set-mode', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--val-set-samples', type=int, default=100000, metavar='EPS',
                        help='')
    ### Unlearning parameters
    parser.add_argument('--unlearn-class', type=str, default="", 
                        help='') 
    parser.add_argument('--ignore-class', type=str, default="", 
                        help='') 
    parser.add_argument('--plot-cm', action='store_true', default=False,
                        help='For Saving the current Model')   
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')   
    parser.add_argument('--multiclass', action='store_true', default=False,
                        help='For Saving the current Model')  
    parser.add_argument('--load-loc', type=str, default=None,
                        help='')    
    parser.add_argument('--save-loc', type=str, default=None,
                        help='')    
    args = parser.parse_args()
    if args.seed == None:
        args.seed = random.randint(0, 65535)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.unlearn_class:
        args.unlearn_class=[int(val) for val in args.unlearn_class.split(",")]
        if args.multiclass and len(args.unlearn_class) == 2 and args.unlearn_class[0] < args.unlearn_class[1]:
            args.unlearn_class = [int(val) for val in range(args.unlearn_class[0], args.unlearn_class[1]+1)]
        elif args.multiclass and len(args.unlearn_class) == 1 :
            if args.dataset =="cifar10":
                args.unlearn_class = random.sample(range(10), args.unlearn_class[0]) 
            elif args.dataset =="cifar100":
                args.unlearn_class = random.sample(range(100), args.unlearn_class[0])
            elif args.dataset =="imagenet":
                args.unlearn_class = random.sample(range(1000), args.unlearn_class[0])
    else:
        args.unlearn_class = []    
        
    if args.ignore_class:
        args.ignore_class=[int(val) for val in args.ignore_class.split(",")]        
    else:
        args.ignore_class = []

    if "," in args.scale_coff:
        args.scale_coff=[float(val) for val in args.scale_coff.split(",")]        
    else:
        args.scale_coff = [float(args.scale_coff)]
    if "," in args.scale_coff_forget:
        args.scale_coff_forget=[float(val) for val in args.scale_coff_forget.split(",")]        
    else:
        args.scale_coff_forget = [float(args.scale_coff_forget)]
    if "," in args.projection_type:
        args.projection_type=[val for val in args.projection_type.split(",")]        
    else:
        args.projection_type = [args.projection_type]
    if "," in args.mode:
        args.mode=[val for val in args.mode.split(",")]        
    else:
        args.mode = [args.mode]
    if args.mode_forget is not None:
        if "," in args.mode_forget:
            args.mode_forget=[val for val in args.mode_forget.split(",")]            
        else:
            args.mode_forget = [args.mode_forget]
    else:
        args.mode_forget = [None]
    if "," in args.start_layer:
        args.start_layer=[int(val) for val in args.start_layer.split(",")]        
    else:
        args.start_layer = [int(args.start_layer)]
    if "," in args.end_layer:
        args.end_layer=[int(val) for val in args.end_layer.split(",")]        
    else:
        args.end_layer = [int(args.end_layer)]
    if "," in args.projection_location:
        args.projection_location=[val for val in args.projection_location.split(",")]        
    else:
        args.projection_location = [args.projection_location]
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
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
    # Load the dataset.
    dataset1, dataset2 = get_dataset(args)
    # Check trained model!
    model = get_model(args, device)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.load_loc is not None:
        model.load_state_dict(torch.load( f"./class_removal/pretrained_models/{args.load_loc}_{args.dataset}_{args.arch}.pt") )
    else:
        model.load_state_dict(torch.load( f"./class_removal/pretrained_models/{args.dataset}_{args.arch}.pt") )
    print("Model Loaded")
    
    # Fraction of trainset used as valset.
    if args.dataset == "cifar100":
        retain_dataset, forget_dataset = get_retain_forget_partition(args, dataset1, args.unlearn_class)
        if len(forget_dataset) < 0.1*args.val_set_samples:
            val_forget_dataset = forget_dataset 
        else:
            _, val_forget_dataset = torch.utils.data.random_split(forget_dataset, [len(forget_dataset)-(0.1*args.val_set_samples), 0.1*args.val_set_samples])
        _, val_retain_dataset = torch.utils.data.random_split(retain_dataset, [len(retain_dataset)-(args.val_set_samples - len(val_forget_dataset)), args.val_set_samples - len(val_forget_dataset)])
        val_dataset = torch.utils.data.ConcatDataset([val_forget_dataset, val_retain_dataset])
    else:
        _, val_dataset= torch.utils.data.random_split(dataset1, [len(dataset1)-args.val_set_samples, args.val_set_samples])


    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    unlearnt_model = activation_projection_based_unlearning(args, model, train_loader, val_loader, test_loader, device)
    if args.save_loc is not None:
        torch.save(unlearnt_model.state_dict(),  f"./class_removal/pretrained_models/{args.save_loc}_{args.dataset}_{args.arch}.pt")

if __name__ == '__main__':
    main()