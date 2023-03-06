import os
import torch.utils.data
from torch.nn import DataParallel
from model.backbone import CBAMResNet
from model.margin import ArcMarginProduct
from dataset.casia_webface import CASIAWebFace
from dataset.agedb import AgeDB30
from torch.optim import lr_scheduler
import torch.optim as optim
from evaluation.eval_agedb import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import torch.nn.functional as F


def run(args):
    ## GPU Settings
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ## Dataset
    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # validation dataset
    trainset = CASIAWebFace(args.train_root, args.train_file_list, down_size=args.down_size, single=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)

    # test dataset
    agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, down_size=args.down_size[0], transform=transform)
    agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=128,
                                            shuffle=False, num_workers=4, drop_last=False)


    ## Model
    HR_Net = CBAMResNet(num_layers=50, feature_dim=512)
    HR_Margin = ArcMarginProduct(in_feature=512, out_feature=trainset.class_nums, s=32.0)

    HR_Net.train()
    HR_Margin.train()

    # Optimizer and scheduler
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD([
            {'params': HR_Net.parameters(), 'weight_decay': 5e-4},
            {'params': HR_Margin.parameters(), 'weight_decay': 5e-4}
        ], lr=0.1, momentum=0.9, nesterov=True)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6, 11, 15, 17], gamma=0.1)

    # Load model to GPUs
    if multi_gpus:
        HR_Net = DataParallel(HR_Net).to(device)
        HR_Margin = DataParallel(HR_Margin).to(device)
        
    else:
        HR_Net = HR_Net.to(device)
        HR_Margin = HR_Margin.to(device)

    ## Train and Evaluation
    best_agedb30_acc = 0.0
    best_agedb30_iters = 0
    total_iters = 0

    GOING = True
    while GOING:
        scheduler.step()

        # Train model
        HR_Net.train()

        for data in tqdm(trainloader):
            img, label = data[0].to(device), data[1].to(device)

            # Forward HR network
            HR_logits = HR_Net(img)
            HR_out = HR_Margin(HR_logits, label)

            # Loss Calcullation
            loss_ce = criterion_ce(HR_out, label)

            # Backward
            optimizer.zero_grad()
            loss_ce.backward()
            optimizer.step()

            # Iters
            total_iters += 1
            if total_iters % 100 == 0:
                _, predict = torch.max(HR_out.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, learning rate: {}".format(total_iters, args.total_iters, loss_ce.item(), 100*correct/total, scheduler.get_lr()[0]))

            
            # Save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                print(msg)

                if multi_gpus:
                    net_state_dict = HR_Net.module.state_dict()
                    margin_state_dict = HR_Margin.module.state_dict()
                else:
                    net_state_dict = HR_Net.state_dict()
                    margin_state_dict = HR_Margin.state_dict()
                
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(args.save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(args.save_dir, 'Iter_%06d_margin.ckpt' % total_iters))
            
            if total_iters == args.total_iters:
                GOING = False
                break


    # Final Checkpoint
    if multi_gpus:
        net_state_dict = HR_Net.module.state_dict()
        margin_state_dict = HR_Margin.module.state_dict()
    else:
        net_state_dict = HR_Net.state_dict()
        margin_state_dict = HR_Margin.state_dict()
    
    torch.save({
        'iters': total_iters,
        'net_state_dict': net_state_dict},
        os.path.join(args.save_dir, 'last_net.ckpt'))
    torch.save({
        'iters': total_iters,
        'net_state_dict': margin_state_dict},
        os.path.join(args.save_dir, 'last_margin.ckpt'))

    
    # Final Eval
    HR_Net.eval()
    HR_Margin.eval()
    
    # Evaluation on AgeDB-30
    getFeatureFromTorch(args, os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'), HR_Net, device, agedbdataset, agedbloader)
    age_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'))
    print('AgeDB-30 Ave Accuracy: {:.4f}'.format(np.mean(age_accs) * 100))

    if best_agedb30_acc <= np.mean(age_accs) * 100:
        best_agedb30_acc = np.mean(age_accs) * 100
        best_agedb30_iters = total_iters

    print('Finally Best Accuracy: AgeDB-30: {:.4f} in iters: {}'.format(best_agedb30_acc, best_agedb30_iters))
    print('finishing training')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--save_dir', type=str, default='checkpoint/teacher', help='model save dir')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--down_size', nargs='+', default=[28], help='down-sampling ratio')
    parser.add_argument('--data_dir', type=str, default='/data/sung/dataset/Face')
    parser.add_argument('--total_iters', type=int, default=47000, help='total epochs')
    
    parser.add_argument('--save_freq', type=int, default=5000, help='save frequency')
    parser.add_argument('--gpus', type=str, default='1', help='model prefix')
    args = parser.parse_args()

    # Downsize
    args.down_size = [int(s) for s in args.down_size]
    print(args.down_size)

    # Path
    args.train_folder = os.path.join(args.data_dir, 'faces_webface_112x112')
    args.eval_folder = os.path.join(args.data_dir, 'evaluation')
    
    args.train_root = os.path.join(args.train_folder, 'image')
    args.train_file_list = os.path.join(args.train_folder, 'train.list')

    args.agedb_test_root = os.path.join(args.eval_folder, 'agedb_30')
    args.agedb_file_list = os.path.join(args.eval_folder, 'agedb_30.txt')

    # Result Folder
    os.makedirs(os.path.join(args.save_dir, 'result'), exist_ok=True)

    # Checkpoint Folder
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run
    run(args)