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


def loadSingleData(data_path1,data_path2, img1_name, img2_name):

    # load image1
    # print(data_path1)
    # print(img1_name)
    input1 = cv2.imread(data_path1+'/'+img1_name)
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0
    input1 = np.transpose(input1, [2, 0, 1])

    # load image2
    input2 = cv2.imread(data_path2+'/'+img2_name)
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
import torch
import torch.nn.functional as F

from skimage.metrics import structural_similarity as compare_ssim
def compute_patch_zncc(patch1, patch2):
    """
    patch1, patch2: tensors of shape [C, K, K]
    对每个通道分别计算ZNCC，然后平均通道结果
    """
    C = patch1.shape[0]
    zncc_channels = []
    for c in range(C):
        p1 = patch1[c]
        p2 = patch2[c]
        p1_mean = p1.mean()
        p2_mean = p2.mean()
        p1_zero_mean = p1 - p1_mean
        p2_zero_mean = p2 - p2_mean
        numerator = (p1_zero_mean * p2_zero_mean).sum()
        denominator = (p1_zero_mean.pow(2).sum().sqrt() * p2_zero_mean.pow(2).sum().sqrt() + 1e-8)
        zncc_channels.append(numerator / (denominator))
    zncc = torch.stack(zncc_channels).mean()
    return zncc
def compute_rmse(patch1, patch2):
    return torch.sqrt(F.mse_loss(patch1, patch2,  reduction='mean'))
def compute_psnr(patch1, patch2):
    mse = F.mse_loss(patch1, patch2, reduction='mean')
    # if mse == 0:
    #     return torch.tensor(float('inf')).cuda()
    return 20 * torch.log10(255.0 / torch.sqrt(mse + 1e-8)).cuda()  # 假设图像值归一化到[0,1]

def compute_ssim(patch1, patch2):
    # patch1, patch2: [C, H, W] -> convert to numpy [H, W, C]
    np1 = patch1.cpu().numpy().transpose(1, 2, 0)
    np2 = patch2.cpu().numpy().transpose(1, 2, 0)
    multichannel = np1.shape[2] > 1
    ssim, _ = compare_ssim(np1, np2, full=True, data_range=255.0,channel_axis=-1 if multichannel else None)
    return torch.tensor(ssim)
def evaluate_seam_quality(seam, ref_img, tgt_img, mask1, mask2, patch_size=15):
    """
    seam_indices: Tensor [N, 2]，每行是 (y, x)
    ref_img, tgt_img: Tensor [H, W] 或 [1, H, W]（灰度图）
    patch_size: patch 尺寸（必须为奇数）
    """
    ref_img=ref_img
    tgt_img=tgt_img
    
    seam_indices = (seam > 0).nonzero()
    mask=(mask1)*(mask2)
    #cv2.imwrite('mask.jpg', (mask*255).cpu().detach().numpy().transpose(1,2,0))
   
    #print(indices)
    _,H, W = ref_img.shape
    r = patch_size // 2  # half window radius
    zncc_scores = []
    rmse_scores =[]
    psnr_scores=[]
    ssim_scores=[]
    for y, x in seam_indices:
        y = y.item()
        x = x.item()
  
        if y < r or y >= H - r or x < r or x >= W - r:
            continue  # 边缘无法取完整 patch，跳过

        patch_ref = ref_img[:,y - r:y + r + 1, x - r:x + r + 1]
        patch_tgt = tgt_img[:,y - r:y + r + 1, x - r:x + r + 1]
        patch_mask = mask[:,y - r:y + r + 1, x - r:x + r + 1]
        patch_ref=patch_mask*patch_ref
        patch_tgt=patch_mask*patch_tgt
        # print(patch_ref.shape)
        # print(ref_img.shape)
        # print(patch_mask)
        # cv2.imwrite('patch_mask.jpg', (patch_mask*255).cpu().detach().numpy().transpose(1,2,0))
        # cv2.imwrite('ref.jpg', (patch_ref).cpu().detach().numpy().transpose(1,2,0))
        # cv2.imwrite('tgt.jpg', (patch_tgt).cpu().detach().numpy().transpose(1,2,0))
       
        #exit(0)
        zncc = compute_patch_zncc(patch_ref, patch_tgt)
        rmse=compute_rmse(patch_ref, patch_tgt)
        psnr=compute_psnr(patch_ref, patch_tgt)
        ssim=compute_ssim(patch_ref, patch_tgt)
        zncc_scores.append(zncc)
        rmse_scores.append(rmse)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    if len(zncc_scores) == 0:
        return None

    zncc_scores = torch.stack(zncc_scores)
    rmse_scores = torch.stack(rmse_scores)
    psnr_scores = torch.stack(psnr_scores)
    ssim_scores = torch.stack(ssim_scores)
    zncc_q_vals = 1.0 - (zncc_scores + 1.0) / 2.0  # ZNCC 转成误差
    zncc_val = zncc_q_vals.mean() if zncc_q_vals is not None else 0
    rmse_val = rmse_scores.mean() if rmse_scores is not None else 0
    psnr_val = psnr_scores.mean() if psnr_scores is not None else 0
    ssim_val = ssim_scores.mean() if ssim_scores is not None else 0

    return zncc_val,rmse_val,psnr_val,ssim_val
import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W
def draw_mesh_on_warp(warp, f_local):


    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp
def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
   
       
        
    # define the network
    net = Network()
    
    if torch.cuda.is_available():
        net = net.cuda()
    
    # for param in net.SAMNet.parameters():
    #     param.requires_grad=False
    
    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    
        #model_path = ckpt_list[-1]
    model_path='/amax/home/taoyz/Seam/Seam_composition/model/epoch100_model.pth'
    checkpoint = torch.load(model_path)
    #net_dict=net.state_dict()#当前模型的参数

    #n_checkpoint={k:v for k,v in checkpoint.items() if k in net_dict}#过滤掉在预训练模型中不存在的key

    #net_dict.update(n_checkpoint)


    net.load_state_dict(checkpoint['model'])

    
    # optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    scheduler.last_epoch = start_epoch
    print('load model from {}!'.format(model_path))
    

    # load dataset(only one pair of images)
    input1_tensor, input2_tensor = loadSingleData(data_path1=args.impath1,data_path2=args.impath2, img1_name = args.img1_name, img2_name = args.img2_name)
    #print(input1_tensor.shape)
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
        # vgg_warp_input2_features_list=batch_out['vgg_warp_input2_features_list']
        # vgg_warp_mesh_features=batch_out['vgg_warp_mesh_features']
        try:
            out=get_stitched_result_build(input1_tensor_512,input2_tensor_512,rigid_mesh, mesh,mask1,mask2)
        except (ValueError, torch.cuda.OutOfMemoryError) as e:
                print("  ⚠ evaluate_seam_quality 解包失败（NoneType），跳过此条")
                break
        seam=out['seam']
        
        total_loss = cal_lp_loss2_cut_warp(input1_tensor_512,output_H,output_H_inv,mask1,mask2,warp_mesh,warp_mesh2,warp_mesh_mask,seam)
        #total_loss = cal_lp_loss2_cut(output_H,output_H_inv,input1_tensor_512, input2_tensor_512,warp_mesh,warp_mesh2, warp_mesh_mask,mask1,mask2,vgg_warp_input2_features_list,vgg_warp_mesh_features)
        total_loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        current_iter = epoch-start_epoch+1
        if current_iter%10 == 0:
            print("Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(current_iter, args.max_iter, total_loss, optimizer.state_dict()['param_groups'][0]['lr']))


        loss_list.append(total_loss)


        if current_iter == 1:
            with torch.no_grad():
                try:
                    output = get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh)
                except ValueError:
                    print("  ⚠ evaluate_seam_quality 解包失败（NoneType），跳过此条")
                    break
        #         cv2.imwrite( args.path+ 's.jpg',(output['seam'][0]*255).cpu().detach().numpy().transpose(1,2,0))
                seam=output['seam'][0,0]
                warp1=output['warp1'][0]
                warp2=output['warp2'][0]
                mask1_w=output['mask1'][0]
                mask2_w=output['mask2'][0]
                try:
                    result = evaluate_seam_quality(seam, warp1, warp2, mask1_w,mask2_w,patch_size=35)
                    if result is not None:
                        zncc,rmse,psnr,ssim=result
                    else :
                        print('294错误')
                        break
                except TypeError:
                    print("  ⚠ evaluate_seam_quality 解包失败（NoneType），跳过此条")
                    break
       

        if current_iter >= 4:
            if torch.abs(loss_list[current_iter-4]-loss_list[current_iter-3]) <= 1e-4 and torch.abs(loss_list[current_iter-3]-loss_list[current_iter-2]) <= 1e-4 \
            and torch.abs(loss_list[current_iter-2]-loss_list[current_iter-1]) <= 1e-4:
                with torch.no_grad():
                    output = get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh)
                    seam=output['seam'][0,0]
                    warp1=output['warp1'][0]
                    warp2=output['warp2'][0]
                    mask1_w=output['mask1'][0]
                    mask2_w=output['mask2'][0]
                    mesh_trans=output['mesh_trans'][0]
                    result = evaluate_seam_quality(seam, warp1, warp2, mask1_w,mask2_w,patch_size=35)
                    if result is not None:
                        zncc,rmse,psnr,ssim=result
                    else :
                        print('330错误')
                        break
                    
                    #print(zncc,rmse,psnr,ssim )
                    values = [zncc.item(), rmse.item(), psnr.item(), ssim.item()]
                    line = "\t".join(f"{v:.4f}" for v in values)
                    print(line)

                    args.sum_zncc += zncc
                    args.sum_rmse += rmse
                    args.sum_psnr += psnr
                    args.sum_ssim += ssim
                    args.count    += 1
                    print(f'sum:',args.count,args.sum_zncc,args.sum_rmse,args.sum_psnr,args.sum_ssim)
                    with open(args.output_file, 'a') as file:  # 'a' 表示追加模式，如果文件不存在会创建文件
                        file.write(f'{args.img1_name}:\t{line}\n')      
                


                path = args.path + "edge_iter-" + str(epoch-start_epoch+1).zfill(3) + ".jpg"
                cv2.imwrite(path, output['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+ 'edge_seam_cut.jpg', output['stitched_cut'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+'edge_mk1.jpg', (batch_out['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
                colored = cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
                cv2.imwrite( args.path+'edge_mk1_color.jpg', colored)
                #cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
                cv2.imwrite( args.path+'edge_warp1.jpg', output['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+'edge_warp2.jpg', output['warp2'][0].cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+'edge_mask1.jpg', (output['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
                cv2.imwrite( args.path+'edge_mask2.jpg', (output['mask2'][0]*255).cpu().detach().numpy().transpose(1,2,0))
                
                warp2_d=output['warp2'][0].cpu().detach().numpy().transpose(1,2,0)
                final_warp2_mesh=draw_mesh_on_warp(warp2_d,mesh_trans)
                cv2.imwrite( args.path+'final_warp2_mesh.jpg', (final_warp2_mesh))
                
                break

        if current_iter == args.max_iter:
            with torch.no_grad():
                output = get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh)
                seam=output['seam'][0,0]
                warp1=output['warp1'][0]
                warp2=output['warp2'][0]
                mask1_w=output['mask1'][0]
                mask2_w=output['mask2'][0]
                try:
                    result = evaluate_seam_quality(seam, warp1, warp2, mask1_w,mask2_w,patch_size=35)
                    if result is not None:
                        zncc,rmse,psnr,ssim=result
                    else :
                        print('379错误')
                        break
                except (ValueError, torch.cuda.OutOfMemoryError) as e:
                    print(f"⚠ 错误: {e}. 跳过此条")
                    break  # 如果在循环中，使用 continue 跳过当前迭代    
                #score = evaluate_seam_quality(seam, warp1, warp2, patch_size=15)
                #print(zncc,rmse,psnr,ssim )
                values = [zncc.item(), rmse.item(), psnr.item(), ssim.item()]
                line = "\t".join(f"{v:.4f}" for v in values)
                print(line)
                mesh_trans=output['mesh_trans'][0]

                
                args.sum_zncc += zncc
                args.sum_rmse += rmse
                args.sum_psnr += psnr
                args.sum_ssim += ssim
                args.count    += 1
                print(f'sum:',args.count,args.sum_zncc,args.sum_rmse,args.sum_psnr,args.sum_ssim)
                with open(args.output_file, 'a') as file:  # 'a' 表示追加模式，如果文件不存在会创建文件
                    file.write(f'{args.img1_name}:\t{line}\n')      
                
                
            path = args.path + "edge_iter-" + str(epoch-start_epoch+1).zfill(3) + ".jpg"
            
            cv2.imwrite(path, output['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
            #cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
            cv2.imwrite( args.path+ 'edge_seam_cut.jpg', output['stitched_cut'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_mk1.jpg', (batch_out['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
            colored = cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET) 
            cv2.imwrite( args.path+'edge_mk1_color.jpg', colored)
            cv2.imwrite( args.path+'edge_warp1.jpg', output['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_warp2.jpg', output['warp2'][0].cpu().detach().numpy().transpose(1,2,0))
            warp2_d=output['warp2'][0].cpu().detach().numpy().transpose(1,2,0)
            final_warp2_mesh=draw_mesh_on_warp(warp2_d,mesh_trans)
            cv2.imwrite( args.path+'final_warp2_mesh.jpg', (final_warp2_mesh))
            cv2.imwrite( args.path+'edge_mask1.jpg', (output['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
            cv2.imwrite( args.path+'edge_mask2.jpg', (output['mask2'][0]*255).cpu().detach().numpy().transpose(1,2,0))

        scheduler.step()
    del net, optimizer
    torch.cuda.empty_cache()

    print("##################end iteration#######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    #nl: create the argument parser
    parser = argparse.ArgumentParser()
 
    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--zpath', type=str, default='/amax/home/taoyz/Seam_opt/parallax/UDIS/')
    parser.add_argument('--img1_name', type=str, default='01.jpg')
    parser.add_argument('--img2_name', type=str, default='02.jpg')
    # 设置文件夹路径
    args = parser.parse_args()
    print(args)

    input1_folder = os.path.join(args.zpath, 'input1')
    args.impath1=input1_folder
    input2_folder = os.path.join(args.zpath, 'input2')
    args.impath2=input2_folder

    # 获取input1和input2文件夹中的所有图像文件
    input1_files = sorted([f for f in os.listdir(input1_folder) if f.endswith('.jpg')])
    input2_files = sorted([f for f in os.listdir(input2_folder) if f.endswith('.jpg')])

    args.output_file='/amax/home/taoyz/Seam_opt/parallax/UDIS/output2.txt'
    args.sum_zncc=0.0
    args.sum_rmse=0.0
    args.sum_psnr=0.0
    args.sum_ssim=0.0
    args.count=0
    
    testpath='/amax/home/taoyz/Seam_opt/parallax/UDIS/out2/'
    # 循环读取input1和input2文件夹中的图像文件
    for i in range(len(input1_files)):
        img1_name = input1_files[i]
        img2_name = input2_files[i]

        # 更新args中的图像文件名称
        args.img1_name = img1_name
        print(img1_name)
        args.img2_name = img2_name

         # 创建新的文件夹，以图像对的名称命名
        folder_name = f"{img1_name.split('.')[0]}"  # 使用图像名创建文件夹名
        
        new_folder_path = os.path.join(testpath, folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created folder: {new_folder_path}")
        args.path = new_folder_path+ os.sep
        
        
        # 输出正在读取的图像对
      

        # 调用train函数并记录时间
        start = time.time()
        train(args)
        end = time.time()
    if args.count > 0:
        avg_zncc = args.sum_zncc / args.count
        avg_rmse = args.sum_rmse / args.count
        avg_psnr = args.sum_psnr / args.count
        avg_ssim = args.sum_ssim / args.count
        print(f"平均ZNCC: {avg_zncc:.4f}")
        print(f"平均RMSE: {avg_rmse:.4f}")
        print(f"平均PSNR: {avg_psnr:.4f}")
        print(f"平均SSIM: {avg_ssim:.4f}")
        values = [avg_zncc.item(), avg_rmse.item(), avg_psnr.item(), avg_ssim.item()]
        line = "\t".join(f"{v:.4f}" for v in values)
        with open(args.output_file, 'a') as file:  # 'a' 表示追加模式，如果文件不存在会创建文件
                file.write(f'avg\t{line}\n') 
    else:
        print("没有数据可计算。") 

        # 打印每一对图像处理的时间
       
    # import torch, gc
    # gc.collect()
    # torch.cuda.empty_cache()  
    # torch.cuda.reset_peak_memory_stats() 
    #nl: parse the arguments
    



