# model.py
import torch.nn as nn
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model
import configs

def build_lora_clip():
    """載入並注入 LoRA 的 CLIP 模型骨幹"""
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(configs.DEVICE)
    for p in clip.parameters(): p.requires_grad = False
    peft_cfg = LoraConfig(
        r=configs.LORA_RANK,
        lora_alpha=configs.LORA_ALPHA,
        target_modules=["q_proj","k_proj","v_proj"],
        lora_dropout=0.05, bias="none"
    )
    clip.vision_model = get_peft_model(clip.vision_model, peft_cfg)
    return clip

def build_classifier():
    """建立線性分類頭 (LoRA 或 Baseline 共用)"""
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proj_dim = clip.config.projection_dim
    return nn.Linear(proj_dim, 2)

def save_checkpoint(clip, clf, path):
    """儲存 model 與分類頭到指定路徑"""
    import torch
    torch.save({
        'vision_state': clip.vision_model.state_dict(),
        'clf_state':    clf.state_dict()
    }, path)