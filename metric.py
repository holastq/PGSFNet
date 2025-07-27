import os
import cv2
import numpy as np
import argparse
import torch
import lpips
from torchvision.transforms import ToTensor
from torchvision import transforms
from scipy.linalg import sqrtm
import torch.nn.functional as F
# from torchvision.models import inception_v3
from scipy import linalg
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)

def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

def _ssim(img, img2):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    # print("ssim:{}".format(np.array(ssims).mean()))
    return np.array(ssims).mean()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, ('Two covariances have different dimensions')

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid

def extract_inception_features_single_image(image, inception, device='cuda'):

    # Ensure the image is in the specified device
    image = image.to(device)
    feature = inception(image)[0].view(1, -1)
    
    return feature

def calculate_fid(img, img2, **kwargs):
#     inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
#     inception = nn.DataParallel(inception).eval().to(device)
    
    # inception_model = Inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model = InceptionV3().eval().cuda()
    inception_model.eval()
    # print(img.shape)
    
    # img_tensor = preprocess_image(img).cuda()
    # img2_tensor = preprocess_image(img2).cuda()
    img_tensor = transform(Image.fromarray(img)).unsqueeze(0).cuda()
    img2_tensor = transform(Image.fromarray(img2)).unsqueeze(0).cuda()
    # print(img_tensor.shape)
    
    # Inception V3
    with torch.no_grad():
        # features1 = inception_model(img_tensor)
        features1 = extract_inception_features_single_image(img_tensor, inception_model)
        features2 = extract_inception_features_single_image(img2_tensor, inception_model)
        # features2 = inception_model(img2_tensor)
    # print(features1.shape)
    features1 = features1.cpu().numpy()
    features2 = features2.cpu().numpy()
    # print(features1.shape)
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, features1, rowvar=False)
    # print(mu1.shape)
    # print(sigma1.shape)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, features2, rowvar=False)

    #FID
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    # print("fid:{}".format(fid))
    return fid

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

lpips_model = lpips.LPIPS(net='vgg').cuda()
transform = transforms.Compose([transforms.ToTensor()])

def calculate_lpips(img, img2, **kwargs):
    img = transform(img)
    img2 = transform(img2)

    img1_tensor = torch.unsqueeze(img, 0).cuda()
    img2_tensor = torch.unsqueeze(img2, 0).cuda()

    lpips = lpips_model(img1_tensor, img2_tensor)

    return lpips.item()


def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def evaluate_images(folder1, folder2):
    # Get list of image filenames in both folders
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    # common_files = set(files1).intersection(files2)
    # print(common_files)
    psnr_values = []
    ssim_values = []
    lpips_values = []
    fid_values = []
    for filename in files1:
        
        img1 = cv2.imread(os.path.join(folder1, filename))#sr
        img2 = cv2.imread(os.path.join(folder2, filename.replace("_realce_pixel106000.png",".JPG")))#gt

        if img1 is None:
            pass
        elif img1.shape[0]<100 or img1.shape[1]<100:
            pass
        else:
            H, W, C = img1.shape
            img2 = cv2.resize(img2, (W, H))
            # H, W, C = img2.shape
            # img1 = cv2.resize(img1, (W, H))
            print(img2.shape)
        
            assert img1.shape == img2.shape
        
            # Calculate PSNR between images
            psnr = calculate_psnr(img1, img2, crop_border=2, test_y_channel=True)
            fid = calculate_fid(img1, img2)
            ssim = calculate_ssim(img1, img2, crop_border=2, test_y_channel=True)
            lpips = calculate_lpips(img1, img2)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips)
            fid_values.append(fid)
    
    # Average PSNR over all images
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    avg_fid = np.mean(fid_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    return avg_psnr, avg_ssim, avg_lpips, avg_fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-sr', type=str, default='./sr')
    parser.add_argument('-gt', type=str, default='./gt')
    args = parser.parse_args()
    folder1 = args.sr
    folder2 = args.gt
    avg_psnr, avg_ssim, avg_lpips, avg_fid = evaluate_images(folder1, folder2)
    print("psnr:{}".format(avg_psnr))
    print("ssim:{}".format(avg_ssim))
    print("lpips:{}".format(avg_lpips))
    print("fid:{}".format(avg_fid))