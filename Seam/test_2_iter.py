import argparse
import torch
import time
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import cv2
#from torch_homography_model import build_model
from network import get_stitched_result, Network, build_new_ft_model,get_stitched_result_cut,get_stitched_result_build

import glob
from loss import cal_lp_loss2,cal_lp_loss2_cut_warp
import torchvision.transforms as T

#import PIL
resize_512 = T.Resize((512,512), antialias=True)


def loadSingleData(data_path, img1_name, img2_name):

    # load image1
    input1 = cv2.imread(data_path+img1_name)
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0
    input1 = np.transpose(input1, [2, 0, 1])

    # load image2
    input2 = cv2.imread(data_path+img2_name)
    input2 = input2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0
    input2 = np.transpose(input2, [2, 0, 1])

    # convert to tensor
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    return (input1_tensor, input2_tensor)



# path of project

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


MODEL_DIR = os.path.join(last_path, 'model')


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
   
       
        
    # define the network
    net = Network()
    
    if torch.cuda.is_available():
        net = net.cuda()
    
    
    optimizer = optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        #model_path = ckpt_list[-1]
        model_path='/amax/home/taoyz/Seam/Warp/model/epoch100_model.pth'
        checkpoint = torch.load(model_path)


        net.load_state_dict(checkpoint['model'])

        
        # optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        print('training from stratch!')

    # load dataset(only one pair of images)
    input1_tensor, input2_tensor = loadSingleData(data_path=args.path, img1_name = args.img1_name, img2_name = args.img2_name)

    if torch.cuda.is_available():
        input1_tensor = input1_tensor.cuda()
        input2_tensor = input2_tensor.cuda()

    input1_tensor_512 = resize_512(input1_tensor)
    input2_tensor_512 = resize_512(input2_tensor)

    loss_list = []

    print("##################start iteration#######################")
    for epoch in range(start_epoch, start_epoch + args.max_iter):
        net.train()
        
        optimizer.zero_grad()

        batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512)
        warp_mesh = batch_out['warp_mesh']
        warp_mesh2=batch_out['warp_mesh2']
        warp_mesh_mask = batch_out['warp_mesh_mask']
        rigid_mesh = batch_out['rigid_mesh']
        mesh = batch_out['mesh']
        mask1=batch_out['mask1']
        mask2=batch_out['mask2']
        output_H=batch_out['output_H']
        output_H_inv =batch_out['output_H_inv']
        
        
        out=get_stitched_result_build(input1_tensor_512,input2_tensor_512,rigid_mesh, mesh,mask1,mask2)
        seam=out['seam']
        
        total_loss = cal_lp_loss2_cut_warp(input1_tensor_512,output_H,output_H_inv,mask1,mask2,warp_mesh,warp_mesh2,warp_mesh_mask,seam)
        total_loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        current_iter = epoch-start_epoch+1
        print("Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(current_iter, args.max_iter, total_loss, optimizer.state_dict()['param_groups'][0]['lr']))


        loss_list.append(total_loss)


        if current_iter == 1:
            with torch.no_grad():
                output = get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh)
                
            cv2.imwrite( args.path+ 'edge_before_seam_cut.jpg', output['stitched_cut'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+ 'edge_before_optimization_cut.jpg', output['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_before_warp1.jpg', output['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_before_warp2.jpg', output['warp2'][0].cpu().detach().numpy().transpose(1,2,0))
            #cv2.imwrite( args.path+ 'before_optimization_mesh_cut.jpg', output['stitched_mesh'])
            cv2.imwrite( args.path+'edge_before_mk1.jpg', (batch_out['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
            colored = cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
            cv2.imwrite( args.path+'edge_before_mk1_color.jpg', colored)
            #exit()

        if current_iter >= 4:
            if torch.abs(loss_list[current_iter-4]-loss_list[current_iter-3]) <= 1e-4 and torch.abs(loss_list[current_iter-3]-loss_list[current_iter-2]) <= 1e-4 \
            and torch.abs(loss_list[current_iter-2]-loss_list[current_iter-1]) <= 1e-4:
                with torch.no_grad():
                    output = get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh)

                path = args.path + "edge_iter-" + str(epoch-start_epoch+1).zfill(3) + ".jpg"
                cv2.imwrite(path, output['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+ 'edge_seam_cut.jpg', output['stitched_cut'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+'edge_mk1.jpg', (batch_out['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
                colored = cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
                cv2.imwrite( args.path+'edge_mk1_color.jpg', colored)
                #cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
                cv2.imwrite( args.path+'edge_warp1.jpg', output['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+'edge_warp2.jpg', output['warp2'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+'edge_mask1.jpg', output['mask1'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+'edge_mask2.jpg', output['mask2'][0].cpu().detach().numpy().transpose(1,2,0))
                break

        if current_iter == args.max_iter:
            with torch.no_grad():
                output = get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh)

            path = args.path + "edge_iter-" + str(epoch-start_epoch+1).zfill(3) + ".jpg"
            
            cv2.imwrite(path, output['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
            #cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
            cv2.imwrite( args.path+ 'edge_seam_cut.jpg', output['stitched_cut'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_mk1.jpg', (batch_out['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
            colored = cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
            cv2.imwrite( args.path+'edge_mk1_color.jpg', colored)
            cv2.imwrite( args.path+'edge_warp1.jpg', output['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_warp2.jpg', output['warp2'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_mask1.jpg', output['mask1'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_mask2.jpg', output['mask2'][0].cpu().detach().numpy().transpose(1,2,0))

        scheduler.step()

    print("##################end iteration#######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    #nl: create the argument parser
    parser = argparse.ArgumentParser()
 
    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--path', type=str, default='/amax/home/taoyz/Seam/Warp/data/')
    parser.add_argument('--img1_name', type=str, default='01.jpg')
    parser.add_argument('--img2_name', type=str, default='02.jpg')


  
    args = parser.parse_args()
    print(args)
    start=time.time()
    #nl: rain
    train(args)
    end=time.time()
    print(end-start)

