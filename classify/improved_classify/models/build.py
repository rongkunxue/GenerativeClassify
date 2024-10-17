import torch.nn as nn


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.type
    # if model_type == "swinTransformer":
    #     layernorm = nn.LayerNorm
    #     from .swinTransformer.swin_transformer import SwinTransformer

    #     model = SwinTransformer(
    #         img_size=config.DATA.IMG_SIZE,
    #         patch_size=config.MODEL.SWIN.PATCH_SIZE,
    #         in_chans=config.MODEL.SWIN.IN_CHANS,
    #         num_classes=config.MODEL.NUM_CLASSES,
    #         embed_dim=config.MODEL.SWIN.EMBED_DIM,
    #         depths=config.MODEL.SWIN.DEPTHS,
    #         num_heads=config.MODEL.SWIN.NUM_HEADS,
    #         window_size=config.MODEL.SWIN.WINDOW_SIZE,
    #         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
    #         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
    #         qk_scale=config.MODEL.SWIN.QK_SCALE,
    #         drop_rate=config.MODEL.DROP_RATE,
    #         drop_path_rate=config.MODEL.DROP_PATH_RATE,
    #         ape=config.MODEL.SWIN.APE,
    #         norm_layer=layernorm,
    #         patch_norm=config.MODEL.SWIN.PATCH_NORM,
    #         use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    #         fused_window_process=False,
    #     )
    if model_type == "GenerativeClassifyUNet_Diff":
        from .generativeClassify.unet_diff import generativeClassify

        model = generativeClassify(config.MODEL)
    elif model_type == "GenerativeClassifyUNet_ICFM":
        from .generativeClassify.unet_icfm import generativeClassify
        
        model = generativeClassify(config.MODEL)
    elif model_type == "GenerativeClassifyUNet_OT":
        from .generativeClassify.unet_ot import generativeClassify

        model = generativeClassify(config.MODEL)
        
    elif model_type == "GenerativeClassifyDiT_Diff":
        from .generativeClassify.dit_diff import generativeClassify

        model = generativeClassify(config.MODEL)
    elif model_type == "GenerativeClassifyDiT_ICFM":
        from .generativeClassify.dit_icfm import generativeClassify

        model = generativeClassify(config.MODEL)
    elif model_type == "GenerativeClassifyDiT_OT":
        from .generativeClassify.dit_ot import generativeClassify

        model = generativeClassify(config.MODEL)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model
