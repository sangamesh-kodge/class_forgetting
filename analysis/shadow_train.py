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
from utils import get_dataset, get_model, get_retain_forget_partition, test
import random

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss= 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
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

# def test(model, device, test_loader, num_classes=10, ignore_class = None):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     cm = np.zeros((num_classes,num_classes))
#     dict_classwise_acc={}
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             cm+=confusion_matrix(target.cpu().numpy(),pred.squeeze(-1).cpu().numpy(), labels=[val for val in range(num_classes)])
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     classwise_acc = cm.diagonal()/cm.sum(axis=1)
#     for i in range(1,11):
#         dict_classwise_acc[str(i)] =  classwise_acc[i-1]

    

#     test_loss /= len(test_loader.dataset)
#     print('\Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

#     wandb.log({ "test_loss":test_loss, 
#                 "test_acc":100. * correct / len(test_loader.dataset),
#                 "classwise/test_acc":dict_classwise_acc
#                 }
#                 )

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
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
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-train-transform', action='store_false', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--arch', type=str, default='vgg11_bn', 
                        help='') 
    parser.add_argument('--data-path', type=str, default='../data/', 
                        help='')
    parser.add_argument('--ignore-class', type=int, default=None, 
                        help='')
    ### wandb parameters
    parser.add_argument('--project-name', type=str, default='train', 
                        help='')
    parser.add_argument('--group-name', type=str, default='final', 
                        help='')    
    args = parser.parse_args()   
    args.train_transform = not args.no_train_transform
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if args.seed is None:
        args.seed = random.randint(0, 65535)
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
    # Load the dataset.
    dataset1, dataset2 = get_dataset(args)
    if args.ignore_class is not None:
        dataset1, _ = get_retain_forget_partition(args, dataset1, [args.ignore_class])
        # args.num_classes = dataset1.num_classes
    # Check trained model!
    model = get_model(args, device)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if os.path.exists(f"./pretrained_models/mia_ulira_models/{args.dataset}_{args.arch}_{args.seed}_{args.ignore_class}.pt"):
        raise FileExistsError
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.gamma)
    run = wandb.init(
                    # Set the project where this run will be logged
                    project=f"Class-{args.dataset}-{args.project_name}",
                    group= f"{args.group_name}-{args.arch}-{args.ignore_class}", 
                    name=f"{args.seed}",
                    dir = os.environ["LOCAL_HOME"],
                    # Track hyperparameters and run metadata
                    config= vars(args))
                    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader, args.num_classes)
        if args.ignore_class is not None:
            unlearn_class_list = [args.ignore_class]
        else:
            unlearn_class_list = [] 
        test(model, device, test_loader,  unlearn_class_list, args.class_label_names, num_classes =args.num_classes, \
                    plot_cm=False, job_name = "baseline", verbose=True, set_name = "Val Set")
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss, epoch=(epoch+1))
            else:
                scheduler.step()
    if not os.path.exists(f"./pretrained_models/mia_ulira_models/"):
        os.makedirs(f"./pretrained_models/mia_ulira_models/")
    torch.save(model.state_dict(), f"./pretrained_models/mia_ulira_models/{args.dataset}_{args.arch}_{args.seed}_{args.ignore_class}.pt")
    wandb.finish()
    
if __name__ == '__main__':
    main()