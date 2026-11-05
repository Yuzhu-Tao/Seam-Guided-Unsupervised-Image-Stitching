import argparse
import torch
import time
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
#from torch_homography_model import build_model
from network import  Network,build_output_model_cut,build_new_ft_model
from dataset import *
import glob
from torch.utils.data import DataLoader

import torchvision.transforms as T

#import PIL
resize_512 = T.Resize((512,512), antialias=True)


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
    return zncc_q_vals.mean(),rmse_scores.mean(),psnr_scores.mean(),ssim_scores.mean()

def loadSingleData(data_path, img1_name, img2_name):

    # load image1
    input1_path=os.path.join(data_path,img1_name )
    input1 = cv2.imread(input1_path)
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0
    input1 = np.transpose(input1, [2, 0, 1])

    # load image2
    input2_path=os.path.join(data_path,img2_name )
    input2 = cv2.imread(input2_path)
    input2 = input2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0
    input2 = np.transpose(input2, [2, 0, 1])

    # convert to tensor
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    return (input1_tensor, input2_tensor)




last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


MODEL_DIR = os.path.join(last_path, 'model')


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

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
def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)
    # define the network
    net = Network()

    if torch.cuda.is_available():
        net = net.cuda()




    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()

        #model_path = ckpt_list[-1]
    model_path='/amax/home/taoyz/Seam/Seam_composition/model/epoch100_model.pth'
    checkpoint = torch.load(model_path)



    net.load_state_dict(checkpoint['model'])


    print('load model from {}!'.format(model_path))



    path_seam = args.result_path + 'before_seam_cut/'
    if not os.path.exists(path_seam):
        os.makedirs(path_seam)
    path_result = args.result_path + 'before_optimization_cut/'
    if not os.path.exists(path_result):
        os.makedirs(path_result)
    path_mk1 = args.result_path + 'mk1/'
    if not os.path.exists(path_mk1):
        os.makedirs(path_mk1)
    path_mk1color = args.result_path + 'mk1_color/'
    if not os.path.exists(path_mk1color):
        os.makedirs(path_mk1color)
    path_mk2 = args.result_path + 'mk2/'
    if not os.path.exists(path_mk2):
        os.makedirs(path_mk2)
    path_mk2color = args.result_path + 'mk2_color/'
    if not os.path.exists(path_mk2color):
        os.makedirs(path_mk2color)
    path_warp1 = args.result_path + 'warp1/'
    if not os.path.exists(path_warp1):
        os.makedirs(path_warp1)
    path_warp2 = args.result_path + 'warp2/'
    if not os.path.exists(path_warp2):
        os.makedirs(path_warp2)
    path_mask1 = args.result_path + 'mask1/'
    if not os.path.exists(path_mask1):
        os.makedirs(path_mask1)
    path_mask2 = args.result_path + 'mask2/'
    if not os.path.exists(path_mask2):
        os.makedirs(path_mask2)
    path_mesh = args.result_path + 'final_warp2_mesh/'
    if not os.path.exists(path_mesh):
        os.makedirs(path_mesh)


    print("##################start test#######################")


    net.eval()
    sum_zncc = 0.0
    sum_rmse = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    count    = 0
    num=0
    for i, batch_value in enumerate(test_loader):

        input1_tensor = batch_value[0].float()
        input2_tensor = batch_value[1].float()
        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()

        with torch.no_grad():


        #     input1_tensor_512 = resize_512(input1_tensor)
        #     input2_tensor_512 = resize_512(input2_tensor)
        #     batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512)

        #     rigid_mesh = batch_out['rigid_mesh']
        #     mesh = batch_out['mesh']
        #     mask1=batch_out['mask1']
        #     mask2=batch_out['mask2']

        #     output=get_stitched_result_cut(input1_tensor,input2_tensor,rigid_mesh, mesh,mask1,mask2)
        #     warp1=output['warp1'][0]
        #     warp2=output['warp2'][0]
        #     mask1_w=output['mask1'][0]
        #     mask2_w=output['mask2'][0]
        #     mesh_trans=output['mesh_trans'][0]

        #     try:
        #         zncc,rmse,psnr,ssim = evaluate_seam_quality(seam, warp1, warp2, mask1_w,mask2_w,patch_size=35)
        #     except TypeError:
        #         print("  ⚠ evaluate_seam_quality 解包失败（NoneType），跳过此条")
        #         continue
        #     sum_zncc += zncc
        #     sum_rmse += rmse
        #     sum_psnr += psnr
        #     sum_ssim += ssim
        #     count    += 1
        #     print('before seam ZNCC\'s score',zncc,rmse,psnr,ssim )
        #     values = [zncc.item(), rmse.item(), psnr.item(), ssim.item()]
        #     line = "\t".join(f"{v:.4f}" for v in values)
        #     print(line)
        #     print(f'sum:',count,sum_zncc,sum_rmse,sum_psnr,sum_ssim)
        #     #print(f'{folder_name}:',line)
        #     with open(args.output_file, 'a') as file:  # 'a' 表示追加模式，如果文件不存在会创建文件
        #         file.write(f'{i+727}:\t{line}\n')

        # path = path_seam + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, (batch_out['stitched_cut'][0]).cpu().detach().numpy().transpose(1,2,0))
        # path = path_result + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, batch_out['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
        #         #cv2.imwrite( args.path+ 'before_optimization_mesh_cut.jpg', output['stitched_mesh'])
        # path = path_mk1 + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, (batch_out['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))
        # colored = cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)
        # path = path_mk1color + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, colored)

        # path = path_mk2 + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, (batch_out['mask2'][0]*255).cpu().detach().numpy().transpose(1,2,0))
        # colored = cv2.applyColorMap((mask2[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)
        # path = path_mask2 + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, colored)


        #             #cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
        # path = path_warp1 + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite(path, batch_out['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
        # path = path_warp2 + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, batch_out['warp2'][0].cpu().detach().numpy().transpose(1,2,0))

        # warp2_d=batch_out['warp2'][0].cpu().detach().numpy().transpose(1,2,0)
        # final_warp2_mesh=draw_mesh_on_warp(warp2_d,mesh_trans)

        # path = path_mesh + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, (final_warp2_mesh))

        # path = path_mask1 + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite(path, (batch_out['mask1_w'][0]*255).cpu().detach().numpy().transpose(1,2,0))

        # path = path_mask2 + str(i+1+727).zfill(6) + ".jpg"
        # cv2.imwrite( path, (batch_out['mask2_w'][0]*255).cpu().detach().numpy().transpose(1,2,0))
        # torch.cuda.empty_cache()


            batch_out = build_output_model_cut(net, input1_tensor, input2_tensor)

            mask1=batch_out['mask1']
            mask2=batch_out['mask2']
            seam=batch_out['seam'][0,0]
            warp1=batch_out['warp1'][0]
            warp2=batch_out['warp2'][0]
            mask1_w=batch_out['mask1_w'][0]
            mask2_w=batch_out['mask2_w'][0]
            mesh_trans=batch_out['mesh_trans'][0]

            try:
                zncc,rmse,psnr,ssim = evaluate_seam_quality(seam, warp1, warp2, mask1_w,mask2_w,patch_size=35)
            except TypeError:
                print("  ⚠ evaluate_seam_quality 解包失败（NoneType），跳过此条")
                continue
            sum_zncc += zncc
            sum_rmse += rmse
            sum_psnr += psnr
            sum_ssim += ssim
            count    += 1
            print('before seam ZNCC\'s score',zncc,rmse,psnr,ssim )
            values = [zncc.item(), rmse.item(), psnr.item(), ssim.item()]
            line = "\t".join(f"{v:.4f}" for v in values)
            print(line)
            print(f'sum:',count,sum_zncc,sum_rmse,sum_psnr,sum_ssim)
            #print(f'{folder_name}:',line)
            with open(args.output_file, 'a') as file:  # 'a' 表示追加模式，如果文件不存在会创建文件
                file.write(f'{i+num}:\t{line}\n')

            path = path_seam + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, (batch_out['stitched_cut'][0]).cpu().detach().numpy().transpose(1,2,0))
            path = path_result + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, batch_out['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
                    #cv2.imwrite( args.path+ 'before_optimization_mesh_cut.jpg', output['stitched_mesh'])
            path = path_mk1 + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, (batch_out['mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0))



            path = path_mk2 + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, (batch_out['mask2'][0]*255).cpu().detach().numpy().transpose(1,2,0))
            H, W = input1_tensor.shape[2], input1_tensor.shape[3]
            mask1 = F.interpolate(mask1, size=(H, W), mode='nearest')
            colored = cv2.applyColorMap((mask1[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)
            path = path_mk1color + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, colored)


            mask2 = F.interpolate(mask2, size=(H, W), mode='nearest')
            colored = cv2.applyColorMap((mask2[0,0]*255).cpu().detach().numpy().clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)
            path = path_mk2color + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, colored)


                        #cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
            path = path_warp1 + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite(path, batch_out['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
            path = path_warp2 + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, batch_out['warp2'][0].cpu().detach().numpy().transpose(1,2,0))

            warp2_d=batch_out['warp2'][0].cpu().detach().numpy().transpose(1,2,0)
            final_warp2_mesh=draw_mesh_on_warp(warp2_d,mesh_trans)

            path = path_mesh + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, (final_warp2_mesh))

            path = path_mask1 + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite(path, (batch_out['mask1_w'][0]*255).cpu().detach().numpy().transpose(1,2,0))

            path = path_mask2 + str(i+num).zfill(6) + ".jpg"
            cv2.imwrite( path, (batch_out['mask2_w'][0]*255).cpu().detach().numpy().transpose(1,2,0))
            torch.cuda.empty_cache()
    if count > 0:
        avg_zncc = sum_zncc / count
        avg_rmse = sum_rmse / count
        avg_psnr = sum_psnr / count
        avg_ssim = sum_ssim / count
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
    print("##################end #######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    #nl: create the argument parser
    parser = argparse.ArgumentParser()

    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/amax/home/taoyz/Seam/Seam_composition/data/')
    parser.add_argument('--result_path', type=str, default='/amax/home/taoyz/Seam/Seam_composition/data/')
    # parser.add_argument('--img1_name', type=str, default='01.jpg')
    # parser.add_argument('--img2_name', type=str, default='02.jpg')
    parser.add_argument('--output_file',type=str,default='taoyz/Seam/Seam_composition/data/output.txt')
    print('<==================== Loading data ===================>\n')

    #nl: parse the arguments
    args = parser.parse_args()
    print(args)
    start=time.time()
    #nl: rain
    test(args)
    end=time.time()
    print(end-start)
