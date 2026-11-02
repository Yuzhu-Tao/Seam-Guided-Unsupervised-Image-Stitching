import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset
import glob
from loss import cal_lp_loss,my_seam_loss, inter_grid_loss, intra_grid_loss



last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary_new_final_2_crop_edge_loss_bn')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
# path to save the model files
MODEL_DIR = os.path.join(last_path, 'model_new_final_2_crop_edge_loss_bn')
# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

import cv2
import numpy as np
import time
def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # define dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
       # net_dict=net.state_dict()#当前模型的参数

        #n_checkpoint={k:v for k,v in checkpoint.items() if k in net_dict}#过滤掉在预训练模型中不存在的key

       # net_dict.update(n_checkpoint)
       # prenet=checkpoint['model'].state_dict()#当前模型的参数
        net.load_state_dict(checkpoint['model'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')



    print("##################start training#######################")
    score_print_fre = 300
    #total_iter=(args.max_epoch-start_epoch)*len(train_loader)
    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.
        #seam_loss_sigma=0.
        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            iter_start = time.time()
            inpu1_tesnor = batch_value[0].float()
            inpu2_tesnor = batch_value[1].float()
           

            if torch.cuda.is_available():
                inpu1_tesnor = inpu1_tesnor.cuda()
               
                inpu2_tesnor = inpu2_tesnor.cuda()
               

            # forward, backward, update weights
            optimizer.zero_grad()

            batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)
            # result
            output_H = batch_out['output_H']
            output_H_inv = batch_out['output_H_inv']
            warp_mesh = batch_out['warp_mesh']
            warp_mesh_mask = batch_out['warp_mesh_mask']
            mesh1 = batch_out['mesh1']
            mesh2 = batch_out['mesh2']
            overlap = batch_out['overlap']
            mask1=batch_out['mask1']
            mask2=batch_out['mask2']
           
            #print(mask1.mean())
            # colored1= cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
            # cv2.imwrite('mk1.jpg',colored1)
            # calculate loss for overlapping regions
            #current_iter=epoch*len(train_loader)+i
            #mask_weight=max(0.01,0.1*(1-current_iter/total_iter))
            #print(mask_weight)
            overlap_loss = cal_lp_loss(inpu1_tesnor, inpu2_tesnor, output_H, output_H_inv, warp_mesh, warp_mesh_mask,mask1,mask2)
            # calculate loss for non-overlapping regions

            #seam_loss=my_seam_loss(inpu1_tesnor, inpu2_tesnor, output_H, output_H_inv, warp_mesh, warp_mesh_mask)
            nonoverlap_loss = 10*inter_grid_loss(overlap, mesh2) + 10*intra_grid_loss(mesh2)

            total_loss = overlap_loss + nonoverlap_loss
            total_loss.backward()

            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            #seam_loss_sigma+=seam_loss.item()
            loss_sigma += total_loss.item()

            #print(glob_iter)

            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                #average_seam_loss=seam_loss_sigma/score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.
                #seam_loss_sigma=0.
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f}  lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_overlap_loss, average_nonoverlap_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                colored1= cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
                colored1 = cv2.cvtColor(colored1, cv2.COLOR_BGR2RGB)
                #cv2.imwrite('mask2.jpg',colored2)
                colored1=torch.from_numpy(colored1).permute(2, 0, 1).cuda()
                colored2= cv2.applyColorMap((mask2[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
                colored2 = cv2.cvtColor(colored2, cv2.COLOR_BGR2RGB)
                #cv2.imwrite('mask2.jpg',colored2)
                colored2=torch.from_numpy(colored2).permute(2, 0, 1).cuda()
                writer.add_image('mask1', colored1, glob_iter)
                writer.add_image('mask2', colored2, glob_iter)
                writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)

                 
                torch.cuda.synchronize()
                iter_time = time.time() - iter_start
                cur_mem = torch.cuda.memory_allocated() / 1024 / 1024
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

                writer.add_scalar('time/iter_time', iter_time, glob_iter)
                writer.add_scalar('memory/current_MB', cur_mem, glob_iter)
                writer.add_scalar('memory/peak_MB', peak_mem, glob_iter)
                #writer.add_scalar('seam loss', average_seam_loss, glob_iter)
            # import torch, gc

            # gc.collect()
            # torch.cuda.empty_cache()
            glob_iter += 1


        scheduler.step()
        # save model
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
    print("##################end training#######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='/amax/home/taoyz/new_final_2/training/')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    
    # train
    train(args)


