# --------------------------------------------------------
# Swin Transformer
# Code Link: https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'

from .pretrained import pre_trained_model
from .NRCVQA import NRCVQA
from .resnet import ResNet_50

def build_model(model_type,text_tokens_T,text_tokens_A,text_tokens_C,embedding_T,embedding_A,embedding_C,text_encoder,ctx_number):
    # model_type = config.MODEL.TYPE

    if model_type == 'resnet-50':
        model = ResNet_50()
    elif model_type == "NRCVQA":
        model = NRCVQA(text_tokens_T,
                        text_tokens_A,
                        text_tokens_C,
                        embedding_T,
                        embedding_A,
                        embedding_C,
                        text_encoder,
                        ctx_number,
                        img_size=224,
                        patch_size=(2,4,4),
                        in_chans=3,
                        num_classes=1,
                        embed_dim=96,
                        depths=[2, 2, 2, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=(8,7,7),
                        qkv_bias=True,
                        qk_scale=None,
                        ape=False,
                        drop_rate=0.2,
                        drop_path_rate=0.2,
                        attn_drop_rate=0.2,
                        patch_norm=True,
                        use_checkpoint=False
                        )
    elif model_type == "pre_train":
        model = pre_trained_model(text_tokens_T,
                                  text_tokens_A,
                                  text_tokens_C,
                                  embedding_T,
                                  embedding_A,
                                  embedding_C,
                                  text_encoder,
                                  ctx_number,
                                  img_size=224,
                                  patch_size=(2,4,4),
                                  in_chans=3,
                                  num_classes=1,
                                  embed_dim=96,
                                  depths=[2, 2, 2, 2],
                                  num_heads=[3, 6, 12, 24],
                                  window_size=(8,7,7),
                                  qkv_bias=True,
                                  qk_scale=None,
                                  ape=False,
                                  drop_rate=0.2,
                                  drop_path_rate=0.2,
                                  attn_drop_rate=0.2,
                                  patch_norm=True,
                                  use_checkpoint=False
                                  )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
