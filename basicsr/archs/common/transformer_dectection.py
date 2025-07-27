import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import copy
from typing import Optional, Tuple
from basicsr.archs.common.multiheadattention import MultiHeadAttention_window, MultiHeadAttention_query

class Transformer(nn.Module):
    def __init__(
        self,
        in_dim=512,
        embed_dim=1024,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        window_size=4,
        normalize_before=True,
        return_intermediate=False
        ):
        super(Transformer, self).__init__()
        encoder_layer = Transform_Encoder_Layer(in_dim, embed_dim, num_heads, dropout, window_size, normalize_before)
        encoder_norm = nn.LayerNorm(in_dim)
        self.encoder = Transform_Encoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = Transform_Decoder_Layer(in_dim, embed_dim, num_heads, dropout, window_size, normalize_before)
        decoder_norm = nn.LayerNorm(in_dim)
        self.decoder = Transform_Decoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate)

        self._reset_parameters()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)#HW B C
        
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # mask = mask.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
        # print(query_embed.shape)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        # print("x.shape:{}".format(x.shape))
        # print("mask:{}".format(mask.shape))
        # print("pos_embed:{}".format(pos_embed.shape))
        memory = self.encoder(x, H=H, W=W, key_padding_mask=mask, pos=pos_embed)
        # print('memory.shape:{}'.format(memory.shape))
        # print("encoder is ready")
        tf = self.decoder(x=tgt, H=H, W=W, memory=memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        # print("decoder is ready")
        # print("tf.shape:{}".format(tf.transpose(1, 2).shape))
        return tf.transpose(1, 2), memory.permute(1, 2, 0).view(B, C, H, W)


def generate_tensor_mask(tensor):#generate mask for the input tensor
    B, C, H, W = tensor.shape
    dtype = tensor.dtype
    device = tensor.device
    tensor_ = torch.zeros((B, H, W), dtype=dtype, device=device)
    mask = torch.ones((B, H, W), dtype=torch.bool, device=device)
    for i, (img, pad_img, m) in enumerate(zip(tensor, tensor_, mask)):
        # print(img)
        # pad_img[:C, :H, :W].copy_(img)
        m[:H, :W] = False
    return tensor, mask


class PositionEmbedding_sin(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor, mask):
        x = tensor
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Transform_Encoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm = True
        ):
        super(Transform_Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x = None, H = None, W = None, mask = None, key_padding_mask = None, pos = None):
        # res = x
        x_ = x
        # print("H:{}".format(H))
        for layer in self.layers:
            x_ = layer(x_, H, W, mask, key_padding_mask, pos) 
        if self.norm is not None:
            x_ = self.norm(x_)
        # x = res + x
        return x_


class Transform_Decoder(nn.Module):
    def __init__(
        self,
        decoder_layer, 
        num_layers, 
        norm=True, 
        return_intermediate=False
        ):
        super(Transform_Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, x = None, H = None, W = None, memory = None, mask = None, key_padding_mask = None, memory_mask = None, memory_key_padding_mask = None, pos = None, query_pos = None):
        x_ = x
        intermediate = []
        # print(memory.shape)
        for layer in self.layers:
            x_ = layer(x=x_, H=H, W=W, memory=memory, mask=mask, x_key_padding_mask=key_padding_mask, memory_mask=memory_mask, 
                        memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(x_))
        if self.norm is not None:

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(x_)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return x_.unsqueeze(0)


class Transform_Encoder_Layer(nn.Module):
    def __init__(
        self,
        in_dim=512,
        embed_dim=2048,
        num_heads=8,
        dropout=0.1,
        window_size=4,
        normalize_before=False
        ):
        super(Transform_Encoder_Layer, self).__init__()
        self.attn = MultiHeadAttention_window(in_dim, window_size, num_heads, attn_drop=dropout)
        self.linear1 = nn.Linear(in_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim, in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before

    def pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, x, H, W, mask = None, key_padding_mask = None, pos = None):
        q = k = self.pos_embed(x, pos)
        x2 = self.attn(q, k, v=x, H=H, W=W, mask=mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

    def forward_pre(self, x, H, W, mask = None, key_padding_mask = None, pos = None):
        x2 = self.norm1(x)
        q = k = self.pos_embed(x, pos)
        x2 = self.attn(q, k, v=x2, H=H, W=W, mask=mask)[0]
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x
    
    def forward(self, x, H, W, mask = None, key_padding_mask = None, pos = None):
        if self.normalize_before:
            x = self.forward_pre(x, H, W, mask, key_padding_mask, pos)
        else:
            x = self.forward_post(x, H, W, mask, key_padding_mask, pos)
        return x


class Transform_Decoder_Layer(nn.Module):
    def __init__(
        self,
        in_dim=512,
        embed_dim=2048,
        num_heads=8,
        dropout=0.1,
        window_size=4,
        normalize_before=False
        ):
        super(Transform_Decoder_Layer, self).__init__()
        self.attn = nn.MultiheadAttention(in_dim, num_heads, dropout=dropout)
        self.multihead_attn = MultiHeadAttention_query(in_dim, window_size, num_heads, attn_drop=dropout)
        self.linear1 = nn.Linear(in_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim, in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace = True)
        self.normalize_before = normalize_before

    def pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, x, H, W, memory, mask = None, x_key_padding_mask = None, 
                    memory_mask = None, memory_key_padding_mask = None,
                    pos = None, query_pos = None):
        q = k = self.pos_embed(x, query_pos)
        x2 = self.attn(q, k, value=x, attn_mask=mask, key_padding_mask=x_key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.multihead_attn(q=self.pos_embed(x, query_pos), k=self.pos_embed(memory, pos), v=memory, H=H, W=W, mask=memory_mask)[0]
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x

    def forward_pre(self, x, H, W, memory, mask = None, x_key_padding_mask = None, 
                    memory_mask = None, memory_key_padding_mask = None,
                    pos = None, query_pos = None):
        x2 = self.norm1(x)
        q = k = self.pos_embed(x2, query_pos)
        x2 = self.attn(q, k, value=x2, attn_mask=mask, key_padding_mask=x_key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.multihead_attn(q=self.pos_embed(x2, query_pos), k=self.pos_embed(memory, pos), v=memory, H=H, W=W, mask=memory_mask)
        x = x + self.dropout2(x2)
        x2 = self.norm3(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout3(x2)
        return x

    def forward(self, x, H, W, memory, mask = None, x_key_padding_mask = None, 
                memory_mask = None, memory_key_padding_mask = None,
                pos = None, query_pos = None):
        if self.normalize_before:
            x = self.forward_pre(x, H, W, memory, mask, x_key_padding_mask, memory_mask, memory_key_padding_mask, pos, query_pos)
        else:
            x = self.forward_post(x, H, W, memory, mask, x_key_padding_mask, memory_mask, memory_key_padding_mask, pos, query_pos)
        return x


class Transfromer_Detection(nn.Module):
    def __init__(
        self,
        in_channel,
        transformer,
        num_classes,
        num_queries,
        aux_loss = False
        ):
        super(Transfromer_Detection, self).__init__()
        self.num_classes = num_classes
        self.transformer = transformer
        hidden_dim = transformer.in_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) 
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 8, 7)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.conv_first = nn.Conv2d(in_channel, hidden_dim, 1)
        self.postion_embed = PositionEmbedding_sin(num_pos_feats=hidden_dim // 2, normalize = True)
        self.aux_loss = aux_loss

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    
    def forward(self, x, mask):
        # x_, mask = generate_tensor_mask(x)
        assert mask is not None
        pos = self.postion_embed(x, mask)
        # print('pos.shape:{}'.format(pos.shape))
        # print(mask.shape)
        # print(x.shape)
        # print(self.query_embed.weight.shape)
        tf, memory = self.transformer(self.conv_first(x), mask, self.query_embed.weight, pos)#text feature
        # print(memory.shape)############################### memory can be used to enhance the imformation of the text position
        outputs_class = self.class_embed(tf)
        outputs_coord = self.bbox_embed(tf).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out, memory


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == "__main__":
    x = torch.randn(1,3,64,64)
    x_, mask = generate_tensor_mask(x)
    dim = 96
    conv = nn.Conv2d(3, dim, 1)
    x = conv(x)
    trans = Transformer(in_dim=dim, embed_dim=256, num_heads=4, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1, normalize_before=True)
    TD = Transfromer_Detection(in_channel=dim, transformer=trans, num_classes=1, num_queries=200)
    out, memory = TD(x, mask)

