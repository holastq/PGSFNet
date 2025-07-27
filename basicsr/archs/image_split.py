import torch
import numpy as np
from PIL import Image

def image_segmentation(image, block_size):

    width, height = image.size
    blocks = []
    for y in range(0, height, block_size[1]):
        for x in range(0, width, block_size[0]):
            box = (x, y, x + block_size[0], y + block_size[1])
            block = image.crop(box)
            blocks.append(block)
    return blocks

def image_to_tensor(image):
    return torch.tensor(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

def tensor_to_image(tensor):
    return Image.fromarray((tensor.squeeze(0) * 255).permute(1, 2, 0).byte().cpu().numpy())

def process_large_image(image_path, block_size, model):
    image = Image.open(image_path)
    image_blocks = image_segmentation(image, block_size)
    processed_blocks = []
    for block in image_blocks:
        tensor_block = image_to_tensor(block)
        processed_block = model(tensor_block)
        processed_blocks.append(processed_block)
    output_image = combine_image_blocks(processed_blocks, image.size, block_size)
    return output_image

def combine_image_blocks(blocks, image_size, block_size):
    width, height = image_size
    output_image = Image.new('RGB', (width, height))
    x, y = 0, 0
    for block in blocks:
        output_image.paste(tensor_to_image(block), (x, y))
        x += block_size[0]
        if x >= width:
            x = 0
            y += block_size[1]
    return output_image
