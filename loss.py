import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import grid_res
from network import optimized_find_seam
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W



   
def compute_mask_loss(img1,img2,warp_mask,l_num=1, mask_threshold=0.2, beta=10, lambda_reg=10):
    # 计算 L1 损失
    l1_loss = torch.abs((img1 - img2)**l_num)
    
    masked_loss = l1_loss[warp_mask > 0]
    
    k=1
    
    mask_threshold=torch.mean(masked_loss)+ k * masked_loss.std()
   
    mask = (l1_loss < mask_threshold).float()
    mask=mask*warp_mask 
    # 计算最终损失
    loss = torch.mean(mask*l1_loss) + lambda_reg *  torch.mean(torch.abs((mask - warp_mask)**l_num))
    
    return loss
#DP mask
def generate_mask(C, I,start,end):
    """
    根据接缝线 C 生成二值 mask
    C：接缝线的列坐标数组
    img_height：图像的高度（行数）
    img_width：图像的宽度（列数）
    """
    # 创建一个与输入图像大小相同的二值mask，初始为0
    batch_size,Ch=C.shape
    _,channel,height,width=I.shape
    mask =torch.zeros((batch_size,channel,height, width),  dtype=torch.float).to(I.device)

    for b in range(batch_size):
        for i in range(torch.floor(start[b]).to(torch.int),torch.floor(end[b]).to(torch.int)+1):
            # 确保 C2[b, i] - 25 和 C2[b, i] + 25 在合法范围内
            start_idx = max(C[b, i] - 25, 0)  
            end_idx = min(C[b, i] + 25, width)  
            mask[b, :, i, start_idx:end_idx] = 1 
    #cv2.imwrite('mas.jpg',(mask[0]*255).cpu().detach().numpy().transpose(1,2,0))
    #print(mask.shape)
    return mask


def l_num_loss(img1, img2, l_num=1):

    return torch.mean(torch.abs((img1 - img2)**l_num))
def masked_L1_loss(img1, img2, mask):
    
    masked_abs_diff = torch.abs(mask * img1 - mask * img2)  # 计算 masked 区域的 L1 误差
    valid_pixel_count = torch.sum(mask>0)  # 计算 mask 有效区域的像素数量（非零部分）
    #cv2.imwrite('ms.jpg',((masked_abs_diff+1)*127.5)[0].cpu().detach().numpy().transpose(1,2,0))
    # 避免除以零（当mask全为0时）
    # if valid_pixel_count == 0:
    #     return torch.tensor(0.0, device=mask.device)
    
    return torch.sum(masked_abs_diff) / valid_pixel_count

def masked_max_loss(img1, img2, mask):
    masked_abs_diff = torch.abs(mask * img1 - mask * img2)  # 计算 masked 区域的 L1 误差

    
    return torch.max(masked_abs_diff)+torch.min(masked_abs_diff)

# #DP
# def my_seam_loss(input1, input2, output_H, output_H_inv, warp_mesh, warp_mesh_mask):

#     batch_size, _, img_h, img_w = input1.size()

#     delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
#     input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)
    
#     mask1=torch.ones_like(input1_newbalance)
#     C1,start1,end1=optimized_find_seam(input1_newbalance, warp_mesh,mask1,warp_mesh_mask)
#     maskc1 = generate_mask(C1, input1_newbalance,start1,end1)  


#     seam_loss=masked_L1_loss(input1_newbalance,warp_mesh,maskc1)

#     return seam_loss
#triplet_loss = nn.TripletMarginLoss(margin=0.5, p=1, reduce=False,size_average=False)
def cal_lp_loss(input1, input2, output_H, output_H_inv, warp_mesh, warp_mesh_mask,mask1,mask2):
    batch_size, _, img_h, img_w = input1.size()

    # part one: sym homo loss with color balance
    delta1 = ( torch.sum(output_H[:,0:3,:,:], [2,3])  -   torch.sum(input1*mask1*output_H[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H[:,3:6,:,:], [2,3])
    input1_balance = input1*mask1 + delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2 = ( torch.sum(output_H_inv[:,0:3,:,:], [2,3])  -   torch.sum(input2*mask2*output_H_inv[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H_inv[:,3:6,:,:], [2,3])
    input2_balance = input2*mask2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    #lp_loss_1 = l_num_loss(input1_balance*output_H[:,3:6,:,:], output_H[:,0:3,:,:], 1) + l_num_loss(input2_balance*output_H_inv[:,3:6,:,:], output_H_inv[:,0:3,:,:], 1)

    lp_loss_1 = torch.mean(torch.abs(input1*mask1*output_H[:,3:6,:,:]-output_H[:,0:3,:,:]))+torch.mean(torch.abs(input2*mask2*output_H_inv[:,3:6,:,:]-output_H_inv[:,0:3,:,:]))
    #lp_loss_1 = torch.mean(torch.abs(input1_balance*output_H[:,3:6,:,:]-output_H[:,0:3,:,:]))/(torch.mean(mask1*mask2)+1e-6)+torch.mean(torch.abs(input2_balance*output_H_inv[:,3:6,:,:]-output_H_inv[:,0:3,:,:]))/(torch.mean(mask1*mask2)+1e-6)
    # part two: tps loss with color balance
    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*mask1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1*mask1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)
    
    #lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)
    lp_loss_2 = torch.mean(torch.abs(input1*mask1*warp_mesh_mask-warp_mesh))
    
    positive =torch.ones_like((mask1 * mask2))


    criterion = nn.BCELoss()
    lp_loss_3 = criterion(mask1*mask2 , positive)


    
    low_ratio1=(mask1<0.2).float().mean()
    low_penalty1=F.relu(low_ratio1-0.3)
    low_ratio2=(mask2<0.2).float().mean()
    low_penalty2=F.relu(low_ratio2-0.3)
    lp_loss =3. * lp_loss_1 + 1. * lp_loss_2+0.1*lp_loss_3.mean()+0.1*(low_penalty1+low_penalty2)

    #print(lp_loss)
    return lp_loss

#initial
def cal_lp_loss2(input1, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()

    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)
    lp_loss =  1. * lp_loss_2

    return lp_loss

def compute_sobel_edge(image,mask):
    gray=0.2989 *image[:,0,:,:]+0.5870 * image[:,1,:,:]+ 0.1140 * image[:,2,:,:]
    img=gray*mask
    C=image.shape[1]
    sobel_x=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32).view(1,1,3,3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3).cuda()
    
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    return grad_x,grad_y

#graph_mask
def dilate_mask(mask):
    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
     #define kernel
    in_channel = 1
    out_channel = 1
    kernel = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(out_channel,in_channel,3,3)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)

    #dilation
    x=mask
    for i in range(20):
        i=i+1
        x = F.conv2d(x,weight,stride=1,padding=1)
        x = torch.where(x < 1 , zeros, ones)
    return x



def cal_lp_loss2_cut(output_H,output_H_inv,input1, input2,warp_mesh, warp_mesh2,warp_mesh_mask,mask1,mask2,vgg_warp_input2_features_list,vgg_warp_mesh_features):

    
    batch_size, _, img_h, img_w = input1.size()

    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1+ delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)
   
    
    
    input1_mask=torch.ones_like(input1)
    m1,m2,seam=optimized_find_seam(input1, warp_mesh2,input1_mask,warp_mesh_mask,1)
    maskc1=(dilate_mask(seam).expand(-1,3,-1,-1))
        # compute_mask_loss_new(input1_newbalance*warp_mesh_mask,warp_mesh,warp_mesh_mask,l_num=1,lambda_reg=10)
        # m_f=getmask(warp_mesh_mask[:,0,:,:].unsqueeze(1))
    lp_loss_4 =5*masked_L1_loss(input1,warp_mesh2,maskc1)
    
   
    lp_loss=lp_loss_4
    return lp_loss
def cal_lp_loss2_cut_warp(input1,output_H,output_H_inv,mask1,mask2,warp_mesh,warp_mesh2,warp_mesh_mask,seam):


    
    

    maskc1=(dilate_mask(seam).expand(-1,3,-1,-1))
    #cv2.imwrite('maskc1.jpg',((maskc1)*255)[0].cpu().detach().numpy().transpose(1,2,0))

   # dil_seam=dilate_seam((seam).expand(-1,3,-1,-1))
    maskc1=maskc1*warp_mesh_mask
    

    #cv2.imwrite('ins.jpg',((ins+1)*127.5)[0].cpu().detach().numpy().transpose(1,2,0))

    #cv2.imwrite('was.jpg',((was+1)*127.5)[0].cpu().detach().numpy().transpose(1,2,0))
    
    lp_loss_4 =1*masked_L1_loss(input1,warp_mesh2,maskc1)
   
    # cv2.imwrite('i2.jpg',(warp_mesh2*maskc1*127.5)[0].cpu().detach().numpy().transpose(1,2,0))
    # 纹理对齐
    grad_x1,grad_y1=compute_sobel_edge(input1,maskc1[:,0,:,:])
    grad_x2,grad_y2=compute_sobel_edge(warp_mesh2,maskc1[:,0,:,:])
    grad1=torch.sqrt(grad_x1*grad_x1+grad_y1*grad_y1)
    grad2=torch.sqrt(grad_x2*grad_x2+grad_y2*grad_y2)
   
    # cv2.imwrite('img2_mask.jpg',(grad_x2*127)[0].cpu().detach().numpy().transpose(1,2,0))
    lp_loss_5 =0.1*(masked_L1_loss(grad_x1,grad_x2,maskc1)+masked_L1_loss(grad_y1,grad_y2,maskc1))
    
   

    return lp_loss_4+lp_loss_5
def inter_grid_loss(overlap, mesh):

    ##############################
    # compute horizontal edges
    w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:]
    # compute angles of two successive horizontal edges
    cos_w = torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (torch.sqrt(torch.sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*torch.sqrt(torch.sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:,0:grid_h,:] + delta_w_angle[:,1:grid_h+1,:]
    ##############################

    ##############################
    # compute vertical edges
    h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]
    # compute angles of two successive vertical edges
    cos_h = torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (torch.sqrt(torch.sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*torch.sqrt(torch.sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:,:,0:grid_w] + delta_h_angle[:,:,1:grid_w+1]
    ##############################

    # on overlapping regions
    depth_diff_w = (1-torch.abs(overlap[:,:,0:grid_w-1] - overlap[:,:,1:grid_w])) * overlap[:,:,0:grid_w-1]
    error_w = depth_diff_w * delta_w_angle
    # on overlapping regions
    depth_diff_h = (1-torch.abs(overlap[:,0:grid_h-1,:] - overlap[:,1:grid_h,:])) * overlap[:,0:grid_h-1,:]
    error_h = depth_diff_h * delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)



# intra-grid constraint
def intra_grid_loss(pts):

    max_w = 512/grid_w * 2
    max_h = 512/grid_h * 2

    delta_x = pts[:,:,1:grid_w+1,0] - pts[:,:,0:grid_w,0]
    delta_y = pts[:,1:grid_h+1,:,1] - pts[:,0:grid_h,:,1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)


    return loss



