# train.py
import torch
import torch.nn as nn
from tqdm import tqdm
import configs
from model import build_lora_clip, build_classifier, save_checkpoint
from dataset import make_loaders
from transformers import CLIPProcessor


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable/total


def train_lora():
    clip = build_lora_clip().to(configs.DEVICE)
    clf  = build_classifier().to(configs.DEVICE)
    print(f"LoRA trainable ratio: {count_trainable_params(clf)*100:.2f}% <5% requirement")
    opt = torch.optim.AdamW(list(clip.vision_model.parameters())+list(clf.parameters()), lr=configs.LR)
    criterion = nn.CrossEntropyLoss()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    train_loader,_,_,_ = make_loaders(proc)
    for ep in range(configs.EPOCHS):
        clip.train(); clf.train()
        loss_sum = 0
        for imgs, labels in tqdm(train_loader, desc=f"LoRA Ep{ep+1}"):
            imgs, labels = imgs.to(configs.DEVICE), labels.to(configs.DEVICE)
            feats = clip.get_image_features(pixel_values=imgs)
            logits = clf(feats); loss = criterion(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step(); loss_sum += loss.item()
        print(f"LoRA Ep{ep+1} Avg Loss: {loss_sum/len(train_loader):.4f}")
    save_checkpoint(clip, clf, configs.CHECKPOINT_DIR/"lora_clip.pth")


def train_baseline():
    clip = build_lora_clip().to(configs.DEVICE)
    clf  = build_classifier().to(configs.DEVICE)
    print(f"Baseline trainable ratio: {count_trainable_params(clf)*100:.2f}%")
    opt = torch.optim.AdamW(clf.parameters(), lr=configs.LR)
    criterion = nn.CrossEntropyLoss()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    train_loader,_,_,_ = make_loaders(proc)
    for ep in range(configs.EPOCHS):
        clf.train(); loss_sum=0
        for imgs, labels in tqdm(train_loader, desc=f"Baseline Ep{ep+1}"):
            imgs, labels = imgs.to(configs.DEVICE), labels.to(configs.DEVICE)
            with torch.no_grad(): feats=clip.get_image_features(pixel_values=imgs)
            logits=clf(feats); loss=criterion(logits,labels)
            opt.zero_grad(); loss.backward(); opt.step(); loss_sum+=loss.item()
        print(f"Baseline Ep{ep+1} Avg Loss: {loss_sum/len(train_loader):.4f}")
    save_checkpoint(clip, clf, configs.CHECKPOINT_DIR/"baseline_probe.pth")
