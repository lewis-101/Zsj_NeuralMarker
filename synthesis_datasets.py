import random

import numpy as np
import cv2
from tqdm import tqdm
import torch
import pandas as pd
from core.utils.augmentor import Augmentor
from easydict import EasyDict
import os
from core.utils.utils import coords_grid
from core.utils.frame_utils import writeFlow
from core.utils.forward_warp import ForwardWarp
import argparse

def create_ranbinbow_stripes(srcImg, stripes_path):
    # 读入图像
    # img = cv.imread('G:\Image_Decomposition\RainbowNet\data_prepare\picture\Places365_test_00000003.jpg')
    # 获取图像的高度和宽度
    height, width = srcImg.shape[:2]

    # 设置图片大小
    img_size = (height, width)

    # 设置条纹的宽度 相当于像素之间的角度的变化速率
    stripe_width = 20
    # 设置色轮的角速度 色轮一秒60圈
    angular_velocity = 2 * np.pi / 60

    # 计算每个像素所对应的时间间隔 相机的一秒30帧
    time_per_pixel = 1 / 30

    # 生成一个空白的图片
    img = np.zeros((img_size[0], img_size[1], 3))

    # 计算每个像素对应的角度
    x, y = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
    angle = y * stripe_width * angular_velocity * time_per_pixel

    # 生成RGB颜色，通过调节振幅可以实现淡化效果
    amplitude = 1
    red = (amplitude * np.sin(angle + 0 * np.pi / 3) + 1) / 2
    green = (amplitude * np.sin(angle + 2 * np.pi / 3) + 1) / 2
    blue = (amplitude * np.sin(angle + 4 * np.pi / 3) + 1) / 2

    # 将RGB颜色加入到图片中
    # img[:, :, 0] = red
    # img[:, :, 1] = green
    # img[:, :, 2] = blue
    img[:, :, 2] = red * 255
    img[:, :, 1] = green * 255
    img[:, :, 0] = blue * 255

    # 显示生成的图片
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    # stripes_path = 'G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg'
    stripes_path = stripes_path
    # plt.imsave('G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg', img)
    cv2.imwrite(stripes_path, img)

    return img, stripes_path

def synthesis_data(args):
    data = pd.read_csv(args.csv)
    augmentor = Augmentor(args)
    forward_warp = ForwardWarp()

    H, W = args.image_size

    first = args.first
    stripes_path = args.stripes_path
    raw_seg_out_path = args.raw_seg_out_path
    # image_free = args.image_free_path

    coords1 = coords_grid(1, H, W, 'cuda:0').contiguous().cuda()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(os.path.join(args.save_dir, 'images'))
        os.mkdir(os.path.join(args.save_dir, 'flows'))
        os.mkdir(os.path.join(args.save_dir, 'masks'))
        os.mkdir(os.path.join(args.save_dir, 'mask_for_blend'))
        os.mkdir(os.path.join(args.save_dir, 'images_mask'))
        os.mkdir(os.path.join(args.save_dir, 'images_free_mask'))
    for idx in tqdm(range(len(data))):
        item = data.iloc[idx]

        im = cv2.imread(os.path.join(args.root, item['fg_path']))
        bg = cv2.imread(os.path.join(args.root, item['bg_path']))
        bg = bg ** 0.85 #背景图变暗
        tnf_type = item['tnf_type']
        theta = None
        if tnf_type == 'affine':
            theta = item[3:9].tolist()
            theta[0] *= 0.85
            theta[4] *= 0.85
        elif tnf_type == 'hom':
            theta = item[3:11].tolist()
            theta[0] *= 0.85
            theta[4] *= 0.85
        elif tnf_type == 'tps':
            theta = item[3:].tolist()
            theta[0] *= 0.85
            theta[4] *= 0.85
        # ===============================================================================
        # 在这里加彩虹条纹 ims = im + ranbow
        # 灰度变RGB
        if len(im.shape) == 2:
            im = np.stack((im,) * 3, axis=-1)

        if first == 1:
            # 创建条纹
            final_stripes, stripes_path = create_ranbinbow_stripes(im, stripes_path)
        # 读取条纹
        final_stripes = cv2.imread(stripes_path)
        # 融合图像
        # ims = cv2.addWeighted(im, 0.85, final_stripes, 0.15, 0)
        # 用完创建条纹方法之后
        first = first + 1
        # ===============================================================================
        # ims 是融合过的 把他当成单单扭曲后的GT

        # im => ims
        fg = torch.from_numpy(im).permute([2, 0, 1])
        theta = torch.Tensor(theta)
        
        im_warp, grid_map, _, _ = augmentor(fg, tnf_type, theta)
        im_warps = im_warp # 伽马变换备份

        grid = coords_grid(1, H, W, grid_map.device)[0].permute(1, 2, 0)
        flow_gt = grid_map - grid
        # 加一个判断 使得图片偏移出太多背景图 则不要
        if flow_gt[:,:,0].max() < W and flow_gt[:,:,1].max() < H and flow_gt[:,:,0].min() > -W and flow_gt[:,:,1].min() >-H:
            out = forward_warp(coords1.cuda(), flow_gt[None].permute([0,3,1,2]).cuda())
            coords2 = out[0][0] / out[1][0]
            intensity = torch.norm(out[0][0], dim=0, p=0).cuda()
            # True -> 1 -> 255 -> 白(表示强度不为0，存在光流变换) False -> 0 -> 黑
            mask = intensity != 0
            forward_flow = (coords2 - coords1) * mask

            warp_mask = (grid_map[:, :, 0] >= 0) & (grid_map[:, :, 0] < W) & \
                        (grid_map[:, :, 1] >= 0) & (grid_map[:, :, 1] < H)

            mask_for_blend = np.expand_dims(warp_mask, 2).repeat(3, 2)
            im_warp = im_warp.permute([1, 2, 0]) * mask_for_blend + bg * ~mask_for_blend # ~ 按位取反
            im_warp = torch.clip(im_warp, 0, 255).numpy().astype(np.uint8)
            # 没加上条纹的gt,仅仅扭曲
            cv2.imwrite(os.path.join(args.save_dir, 'images_free_mask/image_{:06d}_1.png'.format(idx)), im_warp)

            # 融合图像 扭曲完以后再加条纹条纹
            im_warp = cv2.addWeighted(im_warp, 0.85, final_stripes, 0.15, 0)

            # cool = random.uniform(0.6, 1.3) # 伽马变换 (可以不加的)
            cool = 1.5
            im_warpsa = (im_warps.permute([1, 2, 0]) * mask_for_blend) ** cool + bg * ~mask_for_blend # 伽马变暗
            im_warpsa = torch.clip(im_warpsa, 0, 255).numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(args.save_dir, 'images/image_{:06d}_0.png'.format(idx)), im)
            # gt
            cv2.imwrite(os.path.join(raw_seg_out_path, 'image_{:06d}_0.png'.format(idx)), im)
            cv2.imwrite(os.path.join(args.save_dir, 'images/image_{:06d}_1.png'.format(idx)), im_warp)
            cv2.imwrite(os.path.join(args.save_dir, 'images_mask/image_{:06d}_1.png'.format(idx)), im_warp)
            cv2.imwrite(os.path.join(args.save_dir, 'images/image_{:06d}_2.png'.format(idx)), im_warpsa)
            cv2.imwrite(os.path.join(args.save_dir, 'mask_for_blend/image_{:06d}.png'.format(idx)), mask_for_blend.astype(np.uint8)*255)
            cv2.imwrite(os.path.join(args.save_dir, 'images_mask/image_{:06d}.png'.format(idx)),mask_for_blend.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(args.save_dir, 'images_free_mask/image_{:06d}.png'.format(idx)),mask_for_blend.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(args.save_dir, 'masks/mask_{:06d}.png'.format(idx)), mask.cpu().numpy().astype(np.uint8)*255)

            writeFlow(os.path.join(args.save_dir, 'flows/flow_{:06d}.flo'.format(idx)), forward_flow[0].permute([1,2,0]).cpu())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='./data/MegaDepth_CAPS', type=str)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument('--image_size', type=int, nargs='+', default=[480, 640])
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--stripes_path",
                        default='G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg', type=str)
    parser.add_argument("--raw_seg_out_path",
                        default='G:/Image_Decomposition/Grounded-Segment-Anything-outputs/raw_seg_out_free',
                        type=str)
    # parser.add_argument("--image_free_path",
    #                     default='G:/Image_Decomposition/Grounded-Segment-Anything-outputs/image_free',
    #                     type=str)
    args = parser.parse_args()
    # first = 1
    synthesis_data(args)
