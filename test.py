import os
import torch.utils.data
from torch.nn import DataParallel
from model.backbone import CBAMResNet
from dataset.agedb import AgeDB30
from evaluation.eval_agedb import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse


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


    # test dataset
    agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, down_size=args.down_size[0], transform=transform)
    agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=128,
                                            shuffle=False, num_workers=4, drop_last=False)


    ## Model
    net = CBAMResNet(num_layers=50, feature_dim=512)
    net.load_state_dict(torch.load(args.checkpoint_path)['net_state_dict'])

    # Load model to GPUs
    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)


    # Evaluation on AGEDB-30
    net.eval()
    getFeatureFromTorch(args, os.path.join(args.result_dir, 'cur_agedb30_result.mat'), net, device, agedbdataset, agedbloader)
    age_accs = evaluation_10_fold(os.path.join(args.result_dir, 'cur_agedb30_result.mat'))
    print('AgeDB-30 Ave Accuracy: {:.4f}'.format(np.mean(age_accs) * 100))

    with open(os.path.join(args.result_dir, '%s_%.2f' %(args.name, np.mean(age_accs)*100)), 'w') as f:
        f.writelines('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--checkpoint_path', type=str, default='/home/sung/src/attention-transfer-LR-face/A-SKD_public_2/final_teacher/base_28/last_net.ckpt', help='model save dir')
    parser.add_argument('--down_size', nargs='+', default=[28])
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--class_num', type=int, default=10572, help='batch size')
    parser.add_argument('--data_dir', type=str, default='/home/sung/dataset/Face')
    parser.add_argument('--result_dir', type=str, default='./result_folder')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')
    parser.add_argument('--name', type=str, default='teacher', help='meta information')
    args = parser.parse_args()


    # Downsize
    args.down_size = [int(s) for s in args.down_size]
    print(args.down_size)
    
    # Path
    args.eval_folder = os.path.join(args.data_dir, 'evaluation')
    
    args.agedb_test_root = os.path.join(args.eval_folder, 'agedb_30')
    args.agedb_file_list = os.path.join(args.eval_folder, 'agedb_30.txt')
    
    args.cplfw_test_root = os.path.join(args.eval_folder, 'cplfw/aligned_images')
    args.cplfw_file_list = os.path.join(args.eval_folder, 'cplfw/pairs_CPLFW.txt')
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Run
    run(args)
    