import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor
from basicsr.utils.registry import METRIC_REGISTRY
from scipy import linalg
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

# https://github.com/clovaai/stargan-v2/blob/master/metrics/fid.py
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

@METRIC_REGISTRY.register()
def calculate_fid(img, img2, **kwargs):
    inception_model = InceptionV3().eval().cuda()
    inception_model.eval()

    img_tensor = transform(Image.fromarray(img)).unsqueeze(0).cuda()
    img2_tensor = transform(Image.fromarray(img2)).unsqueeze(0).cuda()

    with torch.no_grad():
        features1 = extract_inception_features_single_image(img_tensor, inception_model)
        features2 = extract_inception_features_single_image(img2_tensor, inception_model)

    features1 = features1.cpu().numpy()
    features2 = features2.cpu().numpy()
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, features1, rowvar=False)

    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, features2, rowvar=False)

    # 计算 FID
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.

#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is:
#     d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
#     Stable version by Dougal J. Sutherland.

#     Args:
#         mu1 (np.array): The sample mean over activations.
#         sigma1 (np.array): The covariance matrix over activations for generated samples.
#         mu2 (np.array): The sample mean over activations, precalculated on an representative data set.
#         sigma2 (np.array): The covariance matrix over activations, precalculated on an representative data set.

#     Returns:
#         float: The Frechet Distance.
#     """
#     assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
#     assert sigma1.shape == sigma2.shape, ('Two covariances have different dimensions')

#     cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

#     # Product might be almost singular
#     if not np.isfinite(cov_sqrt).all():
#         print('Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
#         offset = np.eye(sigma1.shape[0]) * eps
#         cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

#     # Numerical error might give slight imaginary component
#     if np.iscomplexobj(cov_sqrt):
#         if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
#             m = np.max(np.abs(cov_sqrt.imag))
#             raise ValueError(f'Imaginary component {m}')
#         cov_sqrt = cov_sqrt.real

#     mean_diff = mu1 - mu2
#     mean_norm = mean_diff @ mean_diff
#     trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
#     fid = mean_norm + trace

#     return fid

# def extract_inception_features_single_image(image, inception, device='cuda'):

#     # Ensure the image is in the specified device
#     image = image.to(device)
#     feature = inception(image)[0].view(1, -1)
    
#     return feature

# @METRIC_REGISTRY.register()
# def calculate_fid(img, img2, **kwargs):
# #     inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
# #     inception = nn.DataParallel(inception).eval().to(device)
    
#     inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
#     inception_model.eval()
#     # print(img.shape)
    
#     # img_tensor = preprocess_image(img).cuda()
#     # img2_tensor = preprocess_image(img2).cuda()
#     img_tensor = transform(Image.fromarray(img)).unsqueeze(0).cuda()
#     img2_tensor = transform(Image.fromarray(img2)).unsqueeze(0).cuda()
#     # print(img_tensor.shape)
    
#     with torch.no_grad():
#         # features1 = inception_model(img_tensor)
#         features1 = extract_inception_features_single_image(img_tensor, inception_model)
#         features2 = extract_inception_features_single_image(img2_tensor, inception_model)
#         # features2 = inception_model(img2_tensor)
#     # print(features1.shape)
#     features1 = features1.cpu().numpy()
#     features2 = features2.cpu().numpy()
#     # print(features1.shape)
#     mu1 = np.mean(features1, axis=0)
#     sigma1 = np.cov(features1, features1, rowvar=False)
#     # print(mu1.shape)
#     # print(sigma1.shape)
#     mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, features2, rowvar=False)

    
#     fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
#     # print("fid:{}".format(fid))
#     return fid

# transform = transforms.Compose([
#     # transforms.Resize(299),
#     # transforms.CenterCrop(299),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

