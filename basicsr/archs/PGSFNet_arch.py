import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper
from omegaconf import OmegaConf
from basicsr.archs.common.upsample import Upsample, UpsampleOneStep
from basicsr.archs.common.mixed_attn_block_efficient import (
    _get_stripe_info,
    EfficientMixAttnTransformerBlock,
)
from basicsr.archs.common.ops import (
    bchw_to_blc,
    blc_to_bchw,
    calculate_mask,
    calculate_mask_all,
    get_relative_coords_table_all,
    get_relative_position_index_simple,
)
from basicsr.archs.common.swin_v1_block import (
    build_last_conv,
)
from timm.models.layers import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.common.fourierup import freup_AreadinterpolationV2, freup_Periodicpadding
from basicsr.archs.common.adaptive_frequency_modulation import AFM
from basicsr.archs.common.transformer_dectection import generate_tensor_mask, Transformer, Transfromer_Detection
from basicsr.archs.common.text_information_enhance import TIE, STN, Transform_Text_Layer

class TransformerStage(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads_window,
        num_heads_stripe,
        window_size,
        stripe_size,
        stripe_groups,
        stripe_shift,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        init_method="",
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        args=None,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.init_method = init_method

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = EfficientMixAttnTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads_w=num_heads_window,
                num_heads_s=num_heads_stripe,
                window_size=window_size,
                window_shift=i % 2 == 0,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] if stripe_shift else False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                res_scale=0.1 if init_method == "r" else 1.0,
                args=args,
            )
            if fairscale_checkpoint:
                block = checkpoint_wrapper(block, offload_to_cpu=offload_to_cpu)
            self.blocks.append(block)
        self.conv = build_last_conv(conv_type, dim)

    def _init_weights(self):
        for n, m in self.named_modules():
            if self.init_method == "w":
                if isinstance(m, (nn.Linear, nn.Conv2d)) and n.find("cpb_mlp") < 0:
                    print("nn.Linear and nn.Conv2d weight initilization")
                    m.weight.data *= 0.1
            elif self.init_method == "l":
                if isinstance(m, nn.LayerNorm):
                    print("nn.LayerNorm initialization")
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
            elif self.init_method.find("t") >= 0:
                scale = 0.1 ** (len(self.init_method) - 1) * int(self.init_method[-1])
                if isinstance(m, nn.Linear) and n.find("cpb_mlp") < 0:
                    trunc_normal_(m.weight, std=scale)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data *= 0.1
                print(
                    "Initialization nn.Linear - trunc_normal; nn.Conv2d - weight rescale."
                )
            else:
                raise NotImplementedError(
                    f"Parameter initialization method {self.init_method} not implemented in TransformerStage."
                )

    def forward(self, x, x_size, table_index_mask):
        res = x
        for blk in self.blocks:
            res = blk(res, x_size, table_index_mask)
        res = bchw_to_blc(self.conv(blc_to_bchw(res, x_size)))#blc_to_bchw Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)
        # print((res+x).shape) # 1 4096 96
        return res + x

    def flops(self):
        pass

class AFMStage(nn.Module):
    def __init__(
        self,
        in_channels=3,#embed dim
        out_channels=3,
        # afm_window_size,
        num_filter=8,
        filter_size="(3,3)",
        use_gaussian=True,
        gaussian_kernel_size="(3,3)",
        gaussian_sigma=0.5,# 0.5-2
        range=4,
        fairscale_checkpoint=False,
        offload_to_cpu=False
    ):
        super(AFMStage, self).__init__()
        self.AFM_block = AFM(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filter=num_filter,
            filter_size=filter_size,
            use_gaussian=use_gaussian,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,# 0.5-2
            range=range,
            fairscale_checkpoint=fairscale_checkpoint,
            offload_to_cpu=offload_to_cpu,
            args=None
        )
    
    def forward(self, x, x_size):
        res = x
        res = blc_to_bchw(res, x_size)
        res = self.AFM_block(res)
        res = bchw_to_blc(res)
        return res

class TD(nn.Module):
    def __init__(
        self,
        depth_1=6,
        depth_2=4,
        in_channels=3,
        out_channels=3,
        embed_dim=96,
        num_heads_1=4,
        num_heads_2=4,
        anchor_use="light_kind",
        anchor_size=2,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
        init_method="n",
        res_scale=1.0,
        conv_type="1conv",
        fairscale_checkpoint=False, 
        offload_to_cpu=False,
        transformer_dim=96,
        transformer_embed_dim=256,
        transformer_num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        transformer_dropout=0.1,
        window_size=4,
        normalize_before=True,
        num_classes=1,
        num_queries=200
    ):
        super(TD, self).__init__()
        self.model_TIE = TIE(
            in_channels=in_channels,
            out_channels=out_channels,
            fairscale_checkpoint=fairscale_checkpoint,
            offload_to_cpu=offload_to_cpu,
        )
        self.model_STN = STN(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim
        )
        self.transformer = Transformer(
            in_dim=transformer_dim,
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=transformer_dropout,
            window_size=window_size,
            normalize_before=normalize_before
        )
        self.conv_TDP = nn.Conv2d(in_channels, transformer_dim, 1)
        self.model_TDP = Transfromer_Detection(#text dectection position
            in_channel=transformer_dim,
            transformer=self.transformer,
            num_classes=num_classes,
            num_queries=num_queries
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0) 
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x):
        x_ = self.model_TIE(x)
        x_, mask = generate_tensor_mask(x)
        x = self.conv_TDP(x)
        out, memory = self.model_TDP(x, mask)
        return memory


@ARCH_REGISTRY.register()
class PGSFNet(nn.Module):
    def __init__(
        self,
        img_size=64,
        in_channels=3,
        out_channels=3,
        embed_dim=96,
        upscale=2,
        img_range=1.0,
        upsampler="",
        depths=[6, 6, 6, 6, 6, 6],
        num_heads_window=[3, 3, 3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3, 3, 3],
        window_size=8,
        stripe_size=[8, 8],  # used for stripe window attention
        stripe_groups=[None, None],
        stripe_shift=False,
        mlp_ratio=4.0,        
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        out_proj_type="linear",
        local_connection=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        # afm_window_size=10,
        num_filter=8,
        filter_size="(3,3)",
        use_gaussian=True,
        gaussian_kernel_size="(3,3)",
        gaussian_sigma=0.5,# 0.5-2
        laplace_range=4,
        res_td=0.1,
        td_TTL_embed_dim=96,
        td_TTL_depth1=6,       
        td_TTL_depth2=4,       
        td_TTL_num_heads_1=4,
        td_TTL_num_heads_2=4,
        td_anchor_use="light_kind",
        td_anchor_size=2,
        td_drop=0.0,
        td_attn_drop=0.0,
        td_res_scale=1.0,
        td_transformer_dim=96,
        td_transformer_embed_dim=256,
        td_transformer_num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        td_transformer_dropout=0.1,    
        td_window_size=4,
        normalize_before=True,
        td_num_classes=1,
        td_num_queries=200,
        init_method="n",  # initialization method of the weight parameters used to train large scale models. normal swin v1
        fairscale_checkpoint=False,  # fairscale activation checkpointing
        offload_to_cpu=False,
        euclidean_dist=False,
        **kwargs,
    ):
        super(PGSFNet, self).__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_out_feats = 64
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.img_range = img_range
        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)


        self.norm_start = norm_layer(embed_dim)
        self.res_td = res_td

        # stochastic depth
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        max_stripe_size = max([0 if s is None else s for s in stripe_size])
        max_stripe_groups = max([0 if s is None else s for s in stripe_groups])
        max_stripe_groups *= anchor_window_down_factor

        self.pad_size = max(window_size, max_stripe_size, max_stripe_groups)
        # if max_stripe_size >= window_size:
        #     self.pad_size *= anchor_window_down_factor
        # if stripe_groups[0] is None and stripe_groups[1] is None:
        #     self.pad_size = max(stripe_size)
        # else:
        #     self.pad_size = window_size
        self.input_resolution = to_2tuple(img_size)
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_window_size = pretrained_window_size
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor
        # Head of the network. First convolution.
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        args = OmegaConf.create(
            {
                "out_proj_type": out_proj_type,
                "local_connection": local_connection,
                "euclidean_dist": euclidean_dist,
            })

        for k, v in self.set_table_index_mask(self.input_resolution).items():
            self.register_buffer(k, v)

        self.layers = nn.ModuleList()
        self.layers_afm = nn.ModuleList()

        for i in range(len(depths)-1):
            layer = TransformerStage(
                dim=embed_dim,
                input_resolution=self.input_resolution,
                depth=depths[i],
                num_heads_window=num_heads_window[i],
                num_heads_stripe=num_heads_stripe[i],
                window_size=self.window_size,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_shift=stripe_shift,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                conv_type=conv_type,
                init_method=init_method,
                fairscale_checkpoint=fairscale_checkpoint,
                offload_to_cpu=offload_to_cpu,
                args=args,
            )
            layer_afm = AFMStage(
                in_channels=embed_dim,
                out_channels=embed_dim,
                num_filter=num_filter,
                filter_size=filter_size,
                use_gaussian=use_gaussian,
                gaussian_kernel_size=gaussian_kernel_size,
                gaussian_sigma=gaussian_sigma,
                range=laplace_range,
                fairscale_checkpoint=fairscale_checkpoint,
                offload_to_cpu=offload_to_cpu
            )
            self.layers.append(layer)
            self.layers_afm.append(layer_afm)

        self.layer_last = TransformerStage(
            dim=embed_dim,
            input_resolution=self.input_resolution,
            depth=depths[len(depths)-1],
            num_heads_window=num_heads_window[len(depths)-1],
            num_heads_stripe=num_heads_stripe[len(depths)-1],
            window_size=self.window_size,
            stripe_size=stripe_size,
            stripe_groups=stripe_groups,
            stripe_shift=stripe_shift,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qkv_proj_type=qkv_proj_type,
            anchor_proj_type=anchor_proj_type,
            anchor_one_stage=anchor_one_stage,
            anchor_window_down_factor=anchor_window_down_factor,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:len(depths)-1]) : sum(depths[: len(depths)])],
            norm_layer=norm_layer,
            pretrained_window_size=pretrained_window_size,
            pretrained_stripe_size=pretrained_stripe_size,
            conv_type=conv_type,
            init_method=init_method,
            fairscale_checkpoint=fairscale_checkpoint,
            offload_to_cpu=offload_to_cpu,
            args=args
        )

        self.TD = TD(
            depth_1=td_TTL_depth1,
            depth_2=td_TTL_depth2,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=td_TTL_embed_dim,
            num_heads_1=td_TTL_num_heads_1,
            num_heads_2=td_TTL_num_heads_2,
            anchor_use=td_anchor_use,
            anchor_size=td_anchor_size,
            mlp_ratio=mlp_ratio,
            drop=td_drop,
            attn_drop=td_attn_drop,
            norm_layer=norm_layer,
            init_method=init_method,
            res_scale=td_res_scale,
            conv_type=conv_type,
            fairscale_checkpoint=fairscale_checkpoint, 
            offload_to_cpu=offload_to_cpu,
            transformer_dim=embed_dim,
            transformer_embed_dim=td_transformer_embed_dim,
            transformer_num_heads=td_transformer_num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            transformer_dropout=td_transformer_dropout,
            window_size=td_window_size,
            normalize_before=normalize_before,
            num_classes=td_num_classes,
            num_queries=td_num_queries
        )
        self.norm_end = norm_layer(embed_dim)

        # Tail of the network
        self.conv_after_body = build_last_conv(conv_type, embed_dim)

        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_feats, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_out_feats)
            self.conv_last = nn.Conv2d(num_out_feats, out_channels, 3, 1, 1)

        elif self.upsampler == "freup_area":
            self.Fup = freup_AreadinterpolationV2(out_channels, upscale)
            self.fuse = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, out_channels, 3, 1, 1), nn.LeakyReLU(inplace=True))

        elif self.upsampler == "freup_period": 
            self.Fup = freup_Periodicpadding(out_channels, upscale)
            self.fuse = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, out_channels, 3, 1, 1), nn.LeakyReLU(inplace=True))
        else:

            self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        self.apply(self._init_weights)
        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()

    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        # ss - stripe_size, sss - stripe_shift_size
        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df = self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )#rpe
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(ss[::-1], self.pretrained_stripe_size, df)


        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)#stripe_size achor to windo index
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)#stripe_size window to achor index
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)#windows mask
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == self.input_resolution:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        mod_pad_w = (self.pad_size - w % self.pad_size) % self.pad_size
        try:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        except BaseException:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
        return x

    def forward_features(self, x):
        B, C, H, W = x.shape
        x_size = (H, W)#hw
        memory = self.TD(x) # TIE
        x_ = self.conv_first(x)
        x = bchw_to_blc(x_)
        x = self.norm_start(x)
        x = self.pos_drop(x)

        table_index_mask = self.get_table_index_mask(x.device, x_size)
        for layer, layer_afm in zip(self.layers, self.layers_afm):
            x = layer(x, x_size, table_index_mask)
            # print("layer is ok")
            x_afm = layer_afm(x, x_size)
            x = x + x_afm * 0.5
        B, L, N = x.shape
        x = x + self.res_td * bchw_to_blc(memory)
        x = self.layer_last(x, x_size, table_index_mask)
        x = self.norm_end(x) 
        x = blc_to_bchw(x, x_size)
        x = self.conv_after_body(x) + x_
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        # print(H)
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        # print(x.shape)

        x = self.forward_features(x)

        if self.upsampler == "pixelshuffle":
            # for classical SR
            # x = self.conv_first(x)
            # x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "freup_area":
            #fourier area + spatial area
            # print(x.shape)
            # x = self.conv_first(x)
            # x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x1 = self.Fup(x)
            x2 = F.interpolate(x, scale_factor=self.upscale, mode='bilinear')
            x = x1 + x2
            x = self.fuse(x)
        elif self.upsampler == "freup_period":
            # x = self.conv_first(x)
            # x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x1 = self.Fup(x)
            x2 = F.interpolate(x, scale_factor=self.upscale, mode='bilinear')
            x = x1 + x2
            x = self.fuse(x)
        else:
            # for image denoising and JPEG compression artifact reduction
            # x_first = self.conv_first(x)
            # res = self.conv_after_body(self.forward_features(x_first)) + x_first
            if self.in_channels == self.out_channels:
                x = x + self.conv_last(res)
            else:
                x = self.conv_last(res)

        x = x / self.img_range + self.mean
        # print("over")

        return x[:, :, : H * self.upscale, : W * self.upscale]

    def flops(self):
        pass

    def convert_checkpoint(self, state_dict):
        for k in list(state_dict.keys()):
            if (
                k.find("relative_coords_table") >= 0
                or k.find("relative_position_index") >= 0
                or k.find("attn_mask") >= 0
                or k.find("model.table_") >= 0
                or k.find("model.index_") >= 0
                or k.find("model.mask_") >= 0
                # or k.find(".upsample.") >= 0
            ):
                state_dict.pop(k)
                print(k)
        return state_dict

if __name__ == "__main__":
    window_size = 8

    model = PGSFNet(
        upscale=4,
        img_size=64,
        window_size=window_size,
        depths=[4, 4, 8, 8, 8, 4, 4],
        embed_dim=180,
        num_heads_window=[3, 3, 3, 3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
        mlp_ratio=2,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_window_down_factor=2,
        out_proj_type="linear",
        conv_type="1conv",
        upsampler="pixelshuffle",
        local_connection=True,
    )

    print(model)

    x = torch.randn((1, 3, 64, 64))
    x = model(x)
    print(x.shape)
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")