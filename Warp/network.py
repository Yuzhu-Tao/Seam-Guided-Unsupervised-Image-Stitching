from matplotlib.pylab import norm
import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models

import torchvision.transforms as T
resize_512 = T.Resize((512,512), antialias=True)
import math
import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W
import maxflow

#find overlap's min&max coordinate
def boundary_stack(mask1,mask2): 
    mask=torch.round(mask1*mask2)
    batch_size,height,width=mask.shape
    device=mask.device
    indices=torch.where(mask>0)
    stacked_indices = torch.stack(indices, dim=1)
    min_y = torch.full((batch_size,), height, dtype=torch.long,device=device)
    min_x = torch.full((batch_size,), width, dtype=torch.long,device=device)
    max_y = torch.full((batch_size,), 0, dtype=torch.long,device=device)
    max_x = torch.full((batch_size,), 0, dtype=torch.long,device=device)
    for b in range(batch_size):
    # 获取当前批次的所有 y 和 x 坐标
        batch_indices = stacked_indices[stacked_indices[:, 0] == b]
        if batch_indices.size(0) > 0:
            min_y[b] = batch_indices[:, 1].min()
            min_x[b] = batch_indices[:, 2].min()
            max_y[b] = batch_indices[:, 1].max()
            max_x[b] = batch_indices[:, 2].max()
    return max_x,min_x,max_y,min_y
def boundary(mask1,mask2): 
    mask=torch.round(mask1*mask2)
    y,x=torch.where(mask>0)
    min_x, min_y = torch.min(x), torch.min(y)
    max_x,max_y=torch.max(x),torch.max(y)  
    return max_x,min_x,max_y,min_y

# find boundary
def find_boundary(mask):
    mask2=mask.clone()
    ones = torch.ones_like(mask2)
    zeros = torch.zeros_like(mask2)
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
   
    x = F.conv2d(1-mask2,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    # x = F.conv2d(x,weight,stride=1,padding=1)
    # x = torch.where(x < 1, zeros, ones)
    # x = F.conv2d(x,weight,stride=1,padding=1)
    # x = torch.where(x < 1, zeros, ones)
   # print(x.shape)

    #cv2.imwrite('x.jpg',(x*mask2*255).cpu().detach().numpy().transpose(1,2,0))
    #exit(0)
    return x*mask2



def compute_sobel_edge(image,flag=1):
    sobel_x=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32).view(1,1,3,3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3).cuda()
    if(flag==1):
        image=image.unsqueeze(0)
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_magnitude=gradient_magnitude.squeeze(0)
    else:
       
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
    return gradient_magnitude

def process_graphcut(args):
    b, bomask1, bomask2, mk1, mk2, overlap_y, overlap_x, norm_shift_x,norm_shift_y = args
    g = maxflow.GraphFloat()
    nodes = g.add_grid_nodes((overlap_y, overlap_x))

    # edges between overlap and source/sink
    source,sink = (math.inf, 0), (0, math.inf)
   
   
    if(np.sum(bomask1) != 0):
        g.add_grid_tedges(nodes[bomask1 == 1], *sink)
    if(np.sum(mk1) != 0):
        g.add_grid_tedges(nodes[mk1 == 1], *sink)
    if(np.sum(bomask2) != 0):
        g.add_grid_tedges(nodes[bomask2 == 1], *source)
    if(np.sum(mk2) != 0):
        g.add_grid_tedges(nodes[mk2 == 1], *source)
    
    # edges in overlap, right and down respectively

   
    structure_x = np.array([[0, 0, 1]], dtype=np.int32)  # 提前创建 NumPy 数组
    structure_y = structure_x.T  # 直接转置，不用重复创建
    g.add_grid_edges(nodes, weights=norm_shift_x, structure=structure_x , symmetric=True)
    g.add_grid_edges(nodes, weights=norm_shift_y, structure=structure_y, symmetric=True)

    flow = g.maxflow()
    # min-cut would divide the graph into two parts
    segments = g.get_grid_segments(nodes)
    #print(segments)
    segments = torch.from_numpy(segments.astype(np.bool_))
    


    return segments

import torch



def cpu_graphcut(batch_size,bomask1_list,bomask2_list,mk1_list,mk2_list,overlap_y_list,overlap_x_list,norm_x_list,norm_y_list):
    segment_list=[]

    g = maxflow.Graph[float]()

    return [process_graphcut((b, 
                                bomask1_list[b].cpu().detach().numpy(), 
                                bomask2_list[b].cpu().detach().numpy(), 
                                mk1_list[b].cpu().detach().numpy(), 
                                mk2_list[b].cpu().detach().numpy(), 
                                overlap_y_list[b].cpu().detach().numpy().item(), 
                                overlap_x_list[b].cpu().detach().numpy().item(), 
                                norm_x_list[b].cpu().detach().numpy(),norm_y_list[b].cpu().detach().numpy())) for b in range(batch_size)]



def optimized_find_seam2(image, patch, mask1,mask2,flag=0):
    device=image.device
    image=image.permute(0,2,3, 1)
    patch=patch.permute(0,2,3, 1)
    mask1_c=mask1[:,0,:,:].clone()
    mask2_c=mask2[:,0,:,:].clone()

    seam_mask = torch.zeros((image.shape[0],1,image.shape[1],image.shape[2]) ,device=device) # 拼接缝mask存储
    bomask1_list,bomask2_list,mk1_list,mk2_list,overlap_y_list,overlap_x_list,norm_x_list,norm_y_list=[],[],[],[],[],[],[],[]
    max_x_stack,min_x_stack,max_y_stack,min_y_stack= boundary_stack(mask1_c,mask2_c)
    for b in range(image.shape[0]):
        #find overlap's min&max coordinate
        
        # print(max_x_stack,min_x_stack,max_y_stack,min_y_stack) 
        # max_x,min_x,max_y,min_y= boundary(mask1_c[b],mask2_c[b]) 
        # print(max_x,min_x,max_y,min_y)
        overlap_y=max_y_stack[b]-min_y_stack[b]+1
        overlap_x=max_x_stack[b]-min_x_stack[b]+1
        image_overlap = image[b,min_y_stack[b]:max_y_stack[b]+1,min_x_stack[b]:max_x_stack[b]+1,0]
        #cv2.imwrite('image_overlap.jpg',image_overlap.cpu().numpy().detach())
        patch_overlap = patch[b,min_y_stack[b]:max_y_stack[b]+1,min_x_stack[b]:max_x_stack[b]+1,0]
        mask1_overlap = mask1_c[b,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]
        mask2_overlap = mask2_c[b,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]
        mask=mask1_overlap*mask2_overlap
      
        mk1=torch.round(mask1_overlap*(1-mask2_overlap))
        mk2=torch.round(mask2_overlap*(1-mask1_overlap))

        b_mask1=find_boundary((mask1_c[b]).unsqueeze(0))
        if torch.sum(b_mask1) == 0:
            #print(0)
            b_mask1[0,:,:2]=1
            b_mask1[0,:,-2:]=1
            b_mask1[0,-2:,:]=1
            b_mask1[0,:2,:]=1
       
        boundary_mask1=torch.round(mask1_c[b].unsqueeze(0)*find_boundary((mask2_c[b]).unsqueeze(0)))
        boundary_mask2=torch.round(mask2_c[b].unsqueeze(0)*b_mask1)
        
        bomask1=boundary_mask1[0,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]
        bomask2=boundary_mask2[0,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]

        

        #exit()
        norm = (torch.abs(image_overlap - patch_overlap).float()/256 )#color,基础权重
        dif=(torch.abs(compute_sobel_edge(image_overlap)-compute_sobel_edge(patch_overlap))/256)
        if(flag==1):
            norm=100*(norm+dif)+1
        else:
            norm=(norm+dif)+1
        #print(torch.mean(norm))
        #norm_shift_x = norm + torch.roll(norm, shifts=-1, dims=1)
        norm_shift_x = norm.clone()
        norm_shift_x[:, :-1] += norm[:, 1:]  # 只计算非溢出部分
        norm_shift_y = norm.clone()
        norm_shift_y[:-1, :] += norm[1:, :]  # 只计算非溢出部分
        #norm_shift_y = norm + torch.roll(norm, shifts=-1, dims=0)
        
        bomask1_list.append(bomask1)
        
        bomask2_list.append(bomask2)
       
        mk1_list.append(mk1)
        
        mk2_list.append(mk2)

     
        overlap_x_list.append(overlap_x) 
        overlap_y_list.append(overlap_y)
        norm_x_list.append(norm_shift_x)
        norm_y_list.append(norm_shift_y)
   
    segments_list=cpu_graphcut(image.shape[0],bomask1_list,bomask2_list,mk1_list,mk2_list,overlap_y_list,overlap_x_list,norm_x_list,norm_y_list)
   
    
    
    for b in range(image.shape[0]):
        sr_mask = segments_list[b]
        src_mask=(sr_mask).to(torch.uint8).to(device)
        dst_mask=(~sr_mask).to(torch.uint8).to(device)
        src = mask1_c[b,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]*src_mask
        dst = mask2_c[b,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]*dst_mask

        mask1_c[b,min_y_stack[b]:max_y_stack[b]+1, min_x_stack[b]:max_x_stack[b]+1]=src
        mask2_c[b,min_y_stack[b]:max_y_stack[b]+1, min_x_stack[b]:max_x_stack[b]+1]=dst

        seam_mask2=find_boundary(dst.float().unsqueeze(0))#(1,height,width)
        #seam_mask2=seam_mask2*src*dst
        #cv2.imwrite('x.jpg',(seam_mask2*255).cpu().detach().numpy().transpose(1,2,0))
        seam_mask[b,:,min_y_stack[b]:max_y_stack[b]+1, min_x_stack[b]:max_x_stack[b]+1]=seam_mask2#(batch,1,height,width)
        
    mask1_c =mask1_c.unsqueeze(1).expand(-1,3,-1,-1).to(device)
    mask2_c =mask2_c.unsqueeze(1).expand(-1,3,-1,-1).to(device)
    seam_mask=seam_mask.to(device)
    return mask1_c,mask2_c,seam_mask


def optimized_find_seam(image, patch, mask1,mask2,flag=0):
    device=image.device
    image=image.permute(0,2,3, 1)
    patch=patch.permute(0,2,3, 1)
    mask1_c=mask1[:,0,:,:].clone()
    mask2_c=mask2[:,0,:,:].clone()
    
    seam_mask = torch.zeros((image.shape[0],1,image.shape[1],image.shape[2]) ,device=device) # 拼接缝mask存储
    bomask1_list,bomask2_list,mk1_list,mk2_list,overlap_y_list,overlap_x_list,norm_x_list,norm_y_list=[],[],[],[],[],[],[],[]
    max_x_stack,min_x_stack,max_y_stack,min_y_stack= boundary_stack(mask1_c,mask2_c)
    for b in range(image.shape[0]):
        #find overlap's min&max coordinate
        
        # print(max_x_stack,min_x_stack,max_y_stack,min_y_stack) 
        # max_x,min_x,max_y,min_y= boundary(mask1_c[b],mask2_c[b]) 
        # print(max_x,min_x,max_y,min_y)
        overlap_y=max_y_stack[b]-min_y_stack[b]+1
        overlap_x=max_x_stack[b]-min_x_stack[b]+1
        image_overlap = image[b,min_y_stack[b]:max_y_stack[b]+1,min_x_stack[b]:max_x_stack[b]+1,0]#(h,w,c)
        
        patch_overlap = patch[b,min_y_stack[b]:max_y_stack[b]+1,min_x_stack[b]:max_x_stack[b]+1,0]#(h,w,c)
        mask1_overlap = mask1_c[b,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]
        mask2_overlap = mask2_c[b,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]
        mask=mask1_overlap*mask2_overlap
      
        mk1=torch.round(mask1_overlap*(1-mask2_overlap))
        mk2=torch.round(mask2_overlap*(1-mask1_overlap))
      
        b_mask1=find_boundary((mask1_c[b]).unsqueeze(0))
        
        if torch.sum(b_mask1) == 0:
            #print(0)
            b_mask1[0,:,:2]=1
            b_mask1[0,:,-2:]=1
            b_mask1[0,-2:,:]=1
            b_mask1[0,:2,:]=1
        b_mask2=find_boundary((mask2_c[b]).unsqueeze(0))
        boundary_mask1=torch.round(mask1_c[b].unsqueeze(0)*find_boundary((mask2_c[b]).unsqueeze(0)))
        boundary_mask2=torch.round(mask2_c[b].unsqueeze(0)*b_mask1)
   
        
        
        bomask1=boundary_mask1[0,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]
        bomask2=boundary_mask2[0,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]
     
       


        norm = (torch.abs(image_overlap - patch_overlap).float() )
        dif=(torch.abs(compute_sobel_edge(image_overlap)-compute_sobel_edge(patch_overlap)))
        #print(dif.shape)
        # cv2.imwrite('mask.jpg',(mask*256).cpu().detach().numpy())
        # cv2.imwrite('dif.jpg',(dif*256).cpu().detach().numpy())
        #exit(0)
        if(flag==1):
            norm=(norm+dif)+1
        else:
            norm=(norm+dif)+1+mask
        
        #print(torch.mean(norm))
        #norm_shift_x = norm + torch.roll(norm, shifts=-1, dims=1)
        norm_shift_x = norm.clone()
        norm_shift_x[:, :-1] += norm[:, 1:]  # 只计算非溢出部分
        norm_shift_y = norm.clone()
        norm_shift_y[:-1, :] += norm[1:, :]  # 只计算非溢出部分
        #norm_shift_y = norm + torch.roll(norm, shifts=-1, dims=0)
   
        bomask1_list.append(bomask1)
        
        bomask2_list.append(bomask2)
       
        mk1_list.append(mk1)
        
        mk2_list.append(mk2)

     
        overlap_x_list.append(overlap_x) 
        overlap_y_list.append(overlap_y)
        norm_x_list.append(norm_shift_x)
        norm_y_list.append(norm_shift_y)
  

      

    #segments_list=cugraph_graphcut(image.shape[0], bomask1_list, bomask2_list, mk1_list, mk2_list, overlap_y_list, overlap_x_list, norm_x_list,norm_y_list)
    segments_list=cpu_graphcut(image.shape[0],bomask1_list,bomask2_list,mk1_list,mk2_list,overlap_y_list,overlap_x_list,norm_x_list,norm_y_list)
   
    
    
    for b in range(image.shape[0]):
        sr_mask = segments_list[b]
        src_mask=(sr_mask).to(torch.uint8).to(device)
        dst_mask=(~sr_mask).to(torch.uint8).to(device)
        src = mask1_c[b,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]*src_mask
        dst = mask2_c[b,min_y_stack[b]:max_y_stack[b] + 1, min_x_stack[b]:max_x_stack[b] + 1]*dst_mask

        mask1_c[b,min_y_stack[b]:max_y_stack[b]+1, min_x_stack[b]:max_x_stack[b]+1]=src
        mask2_c[b,min_y_stack[b]:max_y_stack[b]+1, min_x_stack[b]:max_x_stack[b]+1]=dst

        seam_mask2=find_boundary(dst.float().unsqueeze(0))#(1,height,width)
        #seam_mask2=seam_mask2*src*dst
        #cv2.imwrite('x.jpg',(seam_mask2*255).cpu().detach().numpy().transpose(1,2,0))
        seam_mask[b,:,min_y_stack[b]:max_y_stack[b]+1, min_x_stack[b]:max_x_stack[b]+1]=seam_mask2#(batch,1,height,width)
        
    mask1_c =mask1_c.unsqueeze(1).expand(-1,3,-1,-1).to(device)
    mask2_c =mask2_c.unsqueeze(1).expand(-1,3,-1,-1).to(device)
    seam_mask=seam_mask.to(device)
    return mask1_c,mask2_c,seam_mask



def draw_seam_on_images(img1, C2,k,j,path,start,end):
    batch_size, channels, height, width = img1.shape
    for b in range(img1.shape[0]):
        
        img_np = (img1[b]).cpu().detach().numpy().transpose(1,2,0)
        img_np = img_np.copy()
        #print(img_np.shape)
        #print(img_np.dtype)
        seam_points = [(C2[b, i].item(), i) for i in range(height)]
        # print(seam_points)
        # print(torch.floor(start[b]).to(torch.int),torch.floor(end[b]).to(torch.int))
        # print(seam_points[0])
        # exit()
        for i in range(torch.floor(start[b]).to(torch.int),torch.floor(end[b]).to(torch.int)-1):
            #print(seam_points[i],i)
            cv2.line(img_np, seam_points[i], seam_points[i + 1], (0, 255, 0), 3)
        path_warp1=path
        path = path_warp1 + str(k)+str(j+1).zfill(6) + ".jpg"
        cv2.imwrite(path, img_np)
    return img_np

# draw mesh on image
# warp: h*w*3
# f_local: grid_h*grid_w*2
def draw_mesh_on_warp(warp, f_local):

    warp = np.ascontiguousarray(warp)

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


#Covert global homo into mesh
def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh

# get rigid mesh
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2
# import kornia.augmentation as K
# import kornia.geometry.transform as KT
# import Augmentor


# random augmentation
# it seems to do nothing to the performance
def data_aug(img1, img2):
    # Randomly shift brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img1_aug = img1 * random_brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img2_aug = img2 * random_brightness

    # Randomly shift color
    white = torch.ones([img1.size()[0], img1.size()[2], img1.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img1_aug  *= color_image

    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img2_aug  *= color_image

    # clip
    img1_aug = torch.clamp(img1_aug, -1, 1)
    img2_aug = torch.clamp(img2_aug, -1, 1)

    return img1_aug, img2_aug


def get_vgg19_feature(vgg_model,input,layer_index,flag):
    if(flag==1):
        vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))#由于ImageNet训练时经过均值归一化的
        if torch.cuda.is_available():
            vgg_mean = vgg_mean.cuda()
        vgg_input = input-vgg_mean
    else:
        vgg_input=input
    #x = vgg_model.features[0](vgg_input)
    #FeatureMap_list.append(x)

    x_list = []

    for i in range(0,layer_index+1):
        if i == 0:
            x = vgg_model.features[0](vgg_input)
        else:
            x = vgg_model.features[i](x)
            if i == 6 or i == 13 or i ==24:
            #if i ==24:
                x_list.append(x)#保留低级中级和高级语义特征

    return x_list

# for train.py / test.py
def build_model(net, input1_tensor, input2_tensor,is_training = True):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # network
    if is_training == True:
        aug_input1_tensor, aug_input2_tensor = data_aug(input1_tensor, input2_tensor)
        H_motion, mesh_motion,mask1,mask2 = net(aug_input1_tensor, aug_input2_tensor)
    else:
        H_motion, mesh_motion,mask1,mask2 = net(input1_tensor, input2_tensor)
    
    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])

    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()

    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
    H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

    mask1=mask1.repeat(1,3,1,1)
    mask2=mask2.repeat(1,3,1,1)
    input2_tensor2=input2_tensor*mask2
    input1_tensor2=input1_tensor*mask1
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_H = torch_homo_transform.transformer(torch.cat((input2_tensor2, mask,mask*mask2), 1), H_mat, (img_h, img_w))

    H_inv_mat = torch.matmul(torch.matmul(M_tile_inv, torch.inverse(H)), M_tile)
    output_H_inv = torch_homo_transform.transformer(torch.cat((input1_tensor2, mask,mask*mask1), 1), H_inv_mat, (img_h, img_w))

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion


    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor2, mask,mask*mask2), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]
    
    
    # H2_mask = torch_homo_transform.transformer(mask*mask2, H_mat, (img_h, img_w))
    # H1_mask = torch_homo_transform.transformer(mask*mask1, H_mat, (img_h, img_w))
    # #output_H_inv = torch_homo_transform.transformer(torch.cat((input1_tensor*mask1, mask), 1), H_inv_mat, (img_h, img_w))
    # tps_mask = torch_tps_transform.transformer(mask*mask2, norm_mesh, norm_rigid_mesh, (img_h, img_w))
    

    # calculate the overlapping regions to apply shape-preserving constraints
    overlap = torch_tps_transform.transformer(warp_mesh_mask, norm_rigid_mesh, norm_mesh, (img_h, img_w))
    overlap = overlap.permute(0, 2, 3, 1).unfold(1, int(img_h/grid_h), int(img_h/grid_h)).unfold(2, int(img_w/grid_w), int(img_w/grid_w))
    overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
    overlap_one = torch.ones_like(overlap)
    overlap_zero = torch.zeros_like(overlap)
    overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)


    out_dict = {}
    out_dict.update(output_H=output_H, output_H_inv = output_H_inv, warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, mesh1 = rigid_mesh, mesh2 = mesh, overlap = overlap,mask1=mask1,mask2=mask2)


    return out_dict

# for train_ft.py
def build_new_ft_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()
    #input1_tensor,input2_tensor=augment(input1_tensor,input2_tensor)
    H_motion, mesh_motion,mask1,mask2 = net(input1_tensor, input2_tensor)
    
    H_motion = H_motion.reshape(-1, 4, 2)
    #print(H_motion)
    #H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)

    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    #mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    #print(dst_p)
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)
    #print(H)
    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])

    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()

    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
    H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

    mask1=mask1.repeat(1,3,1,1)
    mask2=mask2.repeat(1,3,1,1)
    input2_tensor2=input2_tensor*mask2
    input1_tensor2=input1_tensor*mask1
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_H = torch_homo_transform.transformer(torch.cat((input2_tensor2, mask), 1), H_mat, (img_h, img_w))

    H_inv_mat = torch.matmul(torch.matmul(M_tile_inv, torch.inverse(H)), M_tile)
    output_H_inv = torch_homo_transform.transformer(torch.cat((input1_tensor2, mask), 1), H_inv_mat, (img_h, img_w))

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion
    #print(mesh)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)
    
    
    
   
   
  
    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor2, mask,input2_tensor), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]
    warp_mesh2 = output_tps[:,6:9,...]
    #cv2.imwrite('warp_mesh.jpg',((warp_mesh+1)*127.5)[0].cpu().detach().numpy().transpose(1,2,0))
    
    
    # #feature
    # vgg_warp_input2_features_list=[]
    
    # with torch.no_grad():
    #     vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    #     vgg_model.eval()
    #     if torch.cuda.is_available():
    #         vgg_model = vgg_model.cuda()
    #     warp_mesh_f=(warp_mesh+1)*127.5
    #     vgg_input2_features=get_vgg19_feature(vgg_model,(input2_tensor2+1)*127.5,24,flag=1)#conv4_3
    #     vgg_warp_mesh_features=get_vgg19_feature(vgg_model,warp_mesh_f,24,flag=1)
    #     vgg_warp_input2_features_list=[]
    #     for i in range(3):
    #         batch_size,channel,img_h_vgg,img_w_vgg=vgg_input2_features[i].size()

        
            
    #         rigid_mesh_vgg = torch.stack([rigid_mesh[...,0]*img_w/img_w_vgg, rigid_mesh[...,1]*img_h/img_h_vgg], 3)
    #         mesh_vgg = torch.stack([mesh[...,0]*img_w/img_w_vgg, mesh[...,1]*img_h/img_h_vgg], 3)
            
    #         norm_rigid_mesh_vgg = get_norm_mesh(rigid_mesh_vgg, img_h, img_w)
    #         norm_mesh_vgg = get_norm_mesh(mesh_vgg, img_h_vgg, img_w_vgg)

    #         vgg_warp_input2_out = torch_tps_transform.transformer((vgg_input2_features[i]), norm_mesh_vgg,norm_rigid_mesh_vgg,  (img_h_vgg, img_w_vgg))
    #         vgg_warp_input2_features_list.append(vgg_warp_input2_out)


  
       
    out_dict = {}
    #out_dict.update(warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, rigid_mesh = rigid_mesh, mesh = mesh,mask1=mask1,mask2=mask2)

    out_dict.update(output_H=output_H,output_H_inv=output_H_inv,warp_mesh = warp_mesh,warp_mesh2=warp_mesh2, warp_mesh_mask = warp_mesh_mask, rigid_mesh = rigid_mesh, mesh = mesh,mask1=mask1,mask2=mask2)


    return out_dict

# for train_ft.py
def get_stitched_result_build(input1_tensor, input2_tensor, rigid_mesh, mesh,mask1,mask2):
    batch_size, _, img_h, img_w = input1_tensor.size()
    
    #rigid_mesh = torch.stack([rigid_mesh[...,0]*img_w/512, rigid_mesh[...,1]*img_h/512], 3)
    #mesh = torch.stack([mesh[...,0]*img_w/512, mesh[...,1]*img_h/512], 3)

    ######################################
    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    # print(out_width)
    # print(out_height)

    warp1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    warp1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = input1_tensor

    mask1_warp = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1_warp[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = 1

    mask1_m = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1_m[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = mask1
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()

    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)

    stitch_tps_out = torch_tps_transform.transformer(torch.cat([input2_tensor, mask,input2_tensor*mask2], 1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))
    
    warp2 = stitch_tps_out[:,0:3,:,:]
    mask2_warp = stitch_tps_out[:,3:6,:,:]
    input2_mask2_m = stitch_tps_out[:,6:9,:,:]
    
   
    m1,m2,seam_init=optimized_find_seam(warp1, warp2,mask1_warp,mask2_warp,1)

    ws12=warp2+seam_init
    ws11=warp1+seam_init
    # cv2.imwrite('ws12.jpg',((ws12+1)*127.5)[0].cpu().detach().numpy().transpose(1,2,0))
    # cv2.imwrite('ws11.jpg',((ws11+1)*127.5)[0].cpu().detach().numpy().transpose(1,2,0))
    seam=seam_init[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w]
    #ws2=output_tps[:,0:3,:,:]+seam
    #cv2.imwrite('ws2.jpg',((ws2+1)*127.5)[0].cpu().detach().numpy().transpose(1,2,0))

    out_dict = {}
    out_dict.update(warp1 = warp1, mask1_warp = mask1_warp, warp2 = warp2, mask2_warp = mask2_warp,mask1_m=mask1_m,input2_mask2_m=input2_mask2_m,seam=seam)

    return out_dict
# for train_ft.py
def get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh):
    batch_size, _, img_h, img_w = input1_tensor.size()

    rigid_mesh = torch.stack([rigid_mesh[...,0]*img_w/512, rigid_mesh[...,1]*img_h/512], 3)
    mesh = torch.stack([mesh[...,0]*img_w/512, mesh[...,1]*img_h/512], 3)

    ######################################
    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    print(out_width)
    print(out_height)

    warp1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    warp1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = (input1_tensor+1)*127.5

    mask1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = 255

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()

    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)

    stitch_tps_out = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask], 1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))
    warp2 = stitch_tps_out[:,0:3,:,:]*127.5
    mask2 = stitch_tps_out[:,3:6,:,:]*255

    stitched = warp1*(warp1/(warp1+warp2+1e-6)) + warp2*(warp2/(warp1+warp2+1e-6))

    stitched_mesh = draw_mesh_on_warp(stitched[0].cpu().detach().numpy().transpose(1,2,0), mesh_trans[0].cpu().detach().numpy())

    out_dict = {}
    out_dict.update(warp1 = warp1, mask1 = mask1, warp2 = warp2, mask2 = mask2, stitched = stitched, stitched_mesh = stitched_mesh)

    return out_dict
# def find_seam(img1,img2,corners,masks):
#     seam_finder = cv2.detail_GraphCutSeamFinder('COLOR')
# # 转换为 float32
#     imgs_float = [img1.astype(np.float32), img2.astype(np.float32)]

#     # 查找接缝
#     masks = seam_finder.find(imgs_float, corners, masks)
#     dilated_mask = cv2.dilate(masks, None)
#     seam_mask = cv2.resize(dilated_mask, (masks.shape[1], masks.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
#     masks = cv2.bitwise_and(seam_mask, masks)
#     cv2.imwrite("seam_mask.jpg",masks)
#     exit()
#     print(masks)
#     seam_mask = cv2.UMat.get(masks[0])
#     overlaid_img = np.copy(img1)
#     overlaid_img[seam_mask == 255] = (0,0,255)
        
#     # resized_mask1 = cv2.resize(seam_masks[0], (img1.shape[1], img1.shape[0]),
#     #                       interpolation=cv2.INTER_LINEAR_EXACT)
#     overlay1 = img1.copy()
#     overlay1[masks == 255] = [0, 0, 255]
#     cv2.imwrite("seam_mask.jpg",overlaid_img)
#     exit()
#     return overlaid_img

def get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh):
    batch_size, _, img_h, img_w = input1_tensor.size()

 
    rigid_mesh = torch.stack([rigid_mesh[...,0]*img_w/512, rigid_mesh[...,1]*img_h/512], 3)
    mesh = torch.stack([mesh[...,0]*img_w/512, mesh[...,1]*img_h/512], 3)

    ######################################
    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)
    #print(width_min,width_max)
    out_width = width_max - width_min
    out_height = height_max - height_min
    # print(out_width)
    # print(out_height)

    warp1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    warp1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = (input1_tensor+1)*127.5

    mask1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = 1

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()

    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    stitch_tps_out = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask], 1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))
    warp2 = stitch_tps_out[:,0:3,:,:]*127.5
    mask2 = stitch_tps_out[:,3:6,:,:]


 


    input1_tensor = (input1_tensor+1)*127.5

    # find_seam(input1_tensor,warp2,[0,0],mask2)
    # maskinput1=torch.ones_like(input1_tensor)
    m1,m2,seam=optimized_find_seam((warp1/127.5-1), (warp2/127.5-1),mask1,mask2,1)

    
    #print(m1.dtype)

 


    seam=seam.expand(-1,3,-1,-1).cuda()
    stitched_cut2=warp1+warp2-mask1*mask2*255+(seam[:,0,:,:])*255
    #draw_seam_on_images(stitched_cut2, C,8,8,path,start,end)

    r=torch.ones_like(warp2).cuda()
    r[:,0,:,:]=((m1[:,0,:,:])*warp1[:,0,:,:])
    r[:,1,:,:]=((m2[:,1,:,:])*warp2[:,1,:,:])
    re = m1 * warp1+ m2 * warp2
  
    out_dict = {}
    out_dict.update(warp1 = warp1, mask1 = m1*255, warp2 = warp2, mask2 = m2*255, stitched = re,stitched_cut=stitched_cut2,seam=seam)

    return out_dict

# for test_output.py
def build_output_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)
    H_motion, mesh_motion = net(resized_input1, resized_input2)

    H_motion = H_motion.reshape(-1, 4, 2)
    H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)


    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    #print(out_width)
    #print(out_height)

    # get warped img1
    M_tensor = torch.tensor([[out_width / 2.0, 0., out_width / 2.0],
                      [0., out_height / 2.0, out_height / 2.0],
                      [0., 0., 1.]])
    N_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
        N_tensor = N_tensor.cuda()
    N_tensor_inv = torch.inverse(N_tensor)

    I_ = torch.tensor([[1., 0., width_min],
                      [0., 1., height_min],
                      [0., 0., 1.]])#.unsqueeze(0)
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        I_ = I_.cuda()
        mask = mask.cuda()
    I_mat = torch.matmul(torch.matmul(N_tensor_inv, I_), M_tensor).unsqueeze(0)

    homo_output = torch_homo_transform.transformer(torch.cat((input1_tensor+1, mask), 1), I_mat, (out_height.int(), out_width.int()))

    torch.cuda.empty_cache()
    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    tps_output = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask],1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))


    out_dict = {}
    out_dict.update(final_warp1=homo_output[:, 0:3, ...]-1, final_warp1_mask = homo_output[:, 3:6, ...], final_warp2=tps_output[:, 0:3, ...]-1, final_warp2_mask = tps_output[:, 3:6, ...], mesh1=rigid_mesh, mesh2=mesh_trans)

    return out_dict

def build_output_model_cut(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)
    H_motion, mesh_motion,mask1,mask2 = net(resized_input1, resized_input2)

    H_motion = H_motion.reshape(-1, 4, 2)
    H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)


    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    #print(out_width)
    #print(out_height)

    # get warped img1
    M_tensor = torch.tensor([[out_width / 2.0, 0., out_width / 2.0],
                      [0., out_height / 2.0, out_height / 2.0],
                      [0., 0., 1.]])
    N_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
        N_tensor = N_tensor.cuda()
    N_tensor_inv = torch.inverse(N_tensor)

    I_ = torch.tensor([[1., 0., width_min],
                      [0., 1., height_min],
                      [0., 0., 1.]])#.unsqueeze(0)
    
   
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        I_ = I_.cuda()
        mask = mask.cuda()
    I_mat = torch.matmul(torch.matmul(N_tensor_inv, I_), M_tensor).unsqueeze(0)

    homo_output = torch_homo_transform.transformer(torch.cat((input1_tensor+1, mask), 1), I_mat, (out_height.int(), out_width.int()))

    torch.cuda.empty_cache()
    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    tps_output = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask],1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))
    
    
    # calculate the overlapping regions to apply shape-preserving constraints
    overlap = torch_tps_transform.transformer(tps_output[:,3:6,...], norm_rigid_mesh, norm_mesh, (out_height.int(), out_width.int()))
    overlap = overlap.permute(0, 2, 3, 1).unfold(1, int(out_height.int()/grid_h), int(out_height.int()/grid_h)).unfold(2, int(out_width.int()/grid_w), int(out_width.int()/grid_w))
    overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
    overlap_one = torch.ones_like(overlap)
    overlap_zero = torch.zeros_like(overlap)
    overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)


    out_dict = {}
    out_dict.update(final_warp1=homo_output[:, 0:3, ...]-1, final_warp1_mask = homo_output[:,3:6,...], final_warp2=tps_output[:, 0:3, ...]-1, final_warp2_mask = tps_output[:,3:6,...], mesh1=rigid_mesh, mesh2=mesh_trans,mask1=mask1,mask2=mask2,overlap=overlap)

    return out_dict



class ResNetBlock(nn.Module):
    """ResNet Basic Block"""
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.shortcut=nn.Identity()
    def forward(self, x):
        residual = self.shortcut(x)  # 跳跃连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差连接
        return F.relu(out)
    
class SAMNet(nn.Module):
    def __init__(self, n_classes,num_res_blocks=2):
        super(SAMNet, self).__init__()

        # 加载预训练的ResNet模型
        # base_model = models.resnet.resnet50(pretrained=True)
        # base_layers = list(base_model.children())

        # # 编码器部分：使用ResNet的前几个层
        # self.encoder0 = nn.Sequential(*base_layers[:4])  # 包括conv1和bn1和relu和maxpool
        # self.encoder1 = nn.Sequential(*base_layers[4:5])  # 包括layer1
        # self.encoder2 = base_layers[5]  # layer2
        # self.encoder3 = base_layers[6]  # layer3
        #self.encoder4 = base_layers[7]  # layer4

        #attention
            # self.con=nn.Conv2d(1,1,kernel_size=3,padding=1)
        # self.att_conv=nn.Sequential(
        #     nn.Conv2d(1,1,kernel_size=3,padding=1),
        #     #torch.sigmoid()
        # )
      
        #encoder
        self.encoder_conv1_1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #ResNetBlock(64),

        )
  
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #ResNetBlock(128),
        )
       
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #ResNetBlock(256),
             #nn.BatchNorm2d(256)
        )
      
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv4_1 = nn.Sequential(
            #nn.Sequential(*[ResNetBlock(256) for _ in range(num_res_blocks)] ),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #ResNetBlock(512),
             #nn.BatchNorm2d(256)
        )
       
       

        self.decoder_up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.decoder_conv3_1 =  nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        self.decoder_conv3_2 =  nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
       

        self.decoder_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
   
        self.decoder_conv2_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder_conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.decoder_up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv1_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
           
            nn.ReLU(True)
        )
        self.decoder_conv1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
     
            nn.ReLU(True)
        )



      
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        # self.final_conv.bias.data.fill_(5.0) 
       
    def forward(self, x):
      
        batch,_,H,W=x.shape#(1,3,512,512)
        
        x_gray=x[:,0,:,:]*0.299+x[:,1,:,:]*0.587+x[:,2,:,:]*0.114
        x_gray=x_gray.unsqueeze(1)
        x_sobel=compute_sobel_edge(x_gray,flag=0)
        
           
        x_sobel_cat=torch.cat((x,x_sobel),dim=1)
        x1 = self.encoder_conv1_1(x_sobel_cat)#(1,64,512,512)
        x1_pool = self.pool1(x1)#(1,64,256,256)
      
        x2 = self.encoder_conv2_1(x1_pool)
        x2_pool = self.pool2(x2)
        
        x3 = self.encoder_conv3_1(x2_pool)
        x3_pool = self.pool3(x3)
  
        x4 = self.encoder_conv4_1(x3_pool)
  
        
        #y=self.res_blocks(x3_pool)
    
 


        y = self.decoder_up3(x4)
        y = torch.cat((y, x3), dim=1)
        y = self.decoder_conv3_1(y)
        y = self.decoder_conv3_2(y)


        y = self.decoder_up2(y)
        y = torch.cat((y, x2), dim=1)
        y = self.decoder_conv2_1(y)
        y = self.decoder_conv2_2(y)


        y = self.decoder_up1(y)
        y = torch.cat((y, x1), dim=1)
        y = self.decoder_conv1_1(y)
        y = self.decoder_conv1_2(y)
        

        
        return self.final_conv(y)
def normMask(mask, strenth = 0.5):
    """
    :return: to attention more region

    """
    batch_size, c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(batch_size, -1).max(1)[0]
    max_value = max_value.reshape(batch_size, 1, 1, 1)
    mask = mask/(max_value*strenth)
    mask = torch.clamp(mask, 0, 1)

    return mask

# define and forward
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SAMNet=SAMNet(n_classes=1).to(device)
        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet1_part2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )


        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)

        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        ssl._create_default_https_context = ssl._create_unverified_context
        resnet50_model = models.resnet.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        #resnet50_model = models.resnet.resnet50(pretrained=True)
        if torch.cuda.is_available():
            resnet50_model = resnet50_model.cuda()
        self.feature_extractor_stage1, self.feature_extractor_stage2 = self.get_res50_FeatureMap(resnet50_model)
        #-----------------------------------------

    def get_res50_FeatureMap(self, resnet50_model):

        layers_list = []

        layers_list.append(resnet50_model.conv1)
        layers_list.append(resnet50_model.bn1)
        layers_list.append(resnet50_model.relu)
        layers_list.append(resnet50_model.maxpool)
        layers_list.append(resnet50_model.layer1)
        layers_list.append(resnet50_model.layer2)

        feature_extractor_stage1 = nn.Sequential(*layers_list)

        feature_extractor_stage2 = nn.Sequential(resnet50_model.layer3)

        #layers_list.append(resnet50_model.layer3)

        return feature_extractor_stage1, feature_extractor_stage2

    # forward
    def forward(self, input1_tesnor, input2_tesnor):
        batch_size, _, img_h, img_w = input1_tesnor.size()
        
       
        #cv2.imwrite('input1_tesnor.jpg',((input1_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0))
        mask1=torch.sigmoid(self.SAMNet(input1_tesnor))
       
        mask2=torch.sigmoid(self.SAMNet(input2_tesnor))
       

        mask1_64 = F.interpolate(mask1, size=(img_h//8, img_w//8), mode='bilinear', align_corners=True)
        mask1_32 = F.interpolate(mask1, size=(img_h//16, img_w//16), mode='bilinear', align_corners=True)
        mask2_64 = F.interpolate(mask2, size=(img_h//8, img_w//8), mode='bilinear', align_corners=True)
        mask2_32 = F.interpolate(mask2, size=(img_h//16, img_w//16), mode='bilinear', align_corners=True)
        

        feature_1_64 = self.feature_extractor_stage1(input1_tesnor)
        feature_1_32 = self.feature_extractor_stage2(feature_1_64)
        feature_2_64 = self.feature_extractor_stage1(input2_tesnor)
        feature_2_32 = self.feature_extractor_stage2(feature_2_64)
      
        feature_1_64=mask1_64*feature_1_64
        feature_1_32=mask1_32*feature_1_32
        feature_2_64=mask2_64*feature_2_64
        feature_2_32=mask2_32*feature_2_32

        ######### stage 1
        correlation_32 = self.CCL(feature_1_32, feature_2_32)
        temp_1 = self.regressNet1_part1(correlation_32)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        offset_1 = self.regressNet1_part2(temp_1)
        H_motion_1 = offset_1.reshape(-1, 4, 2)


        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        warp_feature_2_64 = torch_homo_transform.transformer(feature_2_64, H_mat, (int(img_h/8), int(img_w/8)))

        ######### stage 2
        correlation_64 = self.CCL(feature_1_64, warp_feature_2_64)
        temp_2 = self.regressNet2_part1(correlation_64)
        temp_2 = temp_2.view(temp_2.size()[0], -1)
        offset_2 = self.regressNet2_part2(temp_2)


        return offset_1, offset_2,mask1,mask2


    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches


    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        #print(norm_feature_2.size())

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)

        match_vol = torch.cat(match_vol, 0)
        #print(match_vol .size())

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)
        #print(flow.size())

        return feature_flow
