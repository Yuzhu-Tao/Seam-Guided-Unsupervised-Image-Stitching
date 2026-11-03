import argparse
import torch
import time
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import cv2
#from torch_homography_model import build_model
from network import get_stitched_result, Network, build_new_ft_model,get_stitched_result_cut

import glob
from loss import cal_lp_loss2
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
#nl: os.path.dirname("__file__") ----- the current absolute path
#nl: os.path.pardir ---- the last path
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


#nl: path to save the model files
MODEL_DIR = os.path.join(last_path, 'model')

#nl: create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # define the network
    net = Network()
   
    if torch.cuda.is_available():
        net = net.cuda()
    
    
    

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        #model_path = ckpt_list[-1]
        model_path='/amax/home/taoyz/Seam/Warp/model/epoch100_model.pth'
        checkpoint = torch.load(model_path)
        #net_dict=net.state_dict()#当前模型的参数

        #n_checkpoint={k:v for k,v in checkpoint.items() if k in net_dict}#过滤掉在预训练模型中不存在的key

        #net_dict.update(n_checkpoint)


        net.load_state_dict(checkpoint['model'])

        
        # optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
      
        print('load model from {}!'.format(model_path))
    else:
       
        print('training from stratch!')

    # load dataset(only one pair of images)
    input1_tensor, input2_tensor = loadSingleData(data_path=args.path, img1_name = args.img1_name, img2_name = args.img2_name)
    #print(input1_tensor.shape)
    if torch.cuda.is_available():
        input1_tensor = input1_tensor.cuda()
        input2_tensor = input2_tensor.cuda()

    input1_tensor_512 = resize_512(input1_tensor)
    input2_tensor_512 = resize_512(input2_tensor)

    loss_list = []

    print("##################start test#######################")
    net.eval()


     
    with torch.no_grad():
        batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512)
        
    warp_mesh = batch_out['warp_mesh']
    warp_mesh_mask = batch_out['warp_mesh_mask']
    rigid_mesh = batch_out['rigid_mesh']
    mesh = batch_out['mesh']
    mask1=batch_out['mask1']
    mask2=batch_out['mask2']
    #warp_mesh_mask2=batch_out['warp_mesh_mask2']
    with torch.no_grad():
        output = get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh)
        # vgg_warp_input2_features_list=batch_out['vgg_warp_input2_features_list']
        # vgg_warp_mesh_features=batch_out['vgg_warp_mesh_features']
                
    cv2.imwrite( args.path+ 'before_seam_cut.jpg', output['stitched_cut'][0].cpu().detach().numpy().transpose(1,2,0))
    cv2.imwrite( args.path+ 'before_optimization_cut.jpg', output['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
            #cv2.imwrite( args.path+ 'before_optimization_mesh_cut.jpg', output['stitched_mesh'])
    cv2.imwrite( args.path+'mk1.jpg', (batch_out['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
    colored = cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
    cv2.imwrite( args.path+'mk1_color.jpg', colored)
    cv2.imwrite( args.path+'mk2.jpg', (batch_out['mask2'][0]*255).cpu().detach().numpy().transpose(1,2,0))
    colored = cv2.applyColorMap((mask2[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
    cv2.imwrite( args.path+'mk2_color.jpg', colored)


                #cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
    cv2.imwrite( args.path+'warp1.jpg', output['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
    cv2.imwrite( args.path+'warp2.jpg', output['warp2'][0].cpu().detach().numpy().transpose(1,2,0))
    cv2.imwrite( args.path+'mask1.jpg', output['mask1'][0].cpu().detach().numpy().transpose(1,2,0))
    cv2.imwrite( args.path+'mask2.jpg', output['mask2'][0].cpu().detach().numpy().transpose(1,2,0))
    torch.cuda.empty_cache()         
    print("##################end #######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    #nl: create the argument parser
    parser = argparse.ArgumentParser()
 
    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--path', type=str, default='/amax/home/taoyz/UDIS2-main/Warp2/data/DW_graph/final/1/2_final_data_20/')
    parser.add_argument('--img1_name', type=str, default='01.jpg')
    parser.add_argument('--img2_name', type=str, default='02.jpg')


    # import torch, gc
    # gc.collect()
    # torch.cuda.empty_cache()  
    # torch.cuda.reset_peak_memory_stats() 
    #nl: parse the arguments
    args = parser.parse_args()
    print(args)
    start=time.time()
    #nl: rain
    test(args)
    end=time.time()
    print(end-start)

