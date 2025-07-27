import torch
import lpips
from basicsr.utils.registry import METRIC_REGISTRY
from torchvision.transforms import ToTensor
from torchvision import transforms


lpips_model = lpips.LPIPS(net='vgg').cuda()
transform = transforms.Compose([transforms.ToTensor()])

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, **kwargs):
    # print(img.shape)
    img = transform(img)
    img2 = transform(img2)

    img1_tensor = torch.unsqueeze(img, 0).cuda()
    img2_tensor = torch.unsqueeze(img2, 0).cuda()

    lpips = lpips_model(img1_tensor, img2_tensor)
    # print(lpips)

    return lpips.item()