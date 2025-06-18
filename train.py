# train.py
#import torch
from tqdm import tqdm
import configs
from model import build_lora_clip, build_classifier, save_checkpoint
from dataset import make_loaders
import torch, torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, logging

def train_lora():
    clip = build_lora_clip().to(configs.DEVICE)
    clf  = build_classifier().to(configs.DEVICE)
    opt  = torch.optim.AdamW(
        list(clip.vision_model.parameters()) + list(clf.parameters()), lr=configs.LR
    )
    criterion = nn.CrossEntropyLoss()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    train_loader, _, _, _ = make_loaders(proc)

    for ep in range(configs.EPOCHS):
        clip.train(); clf.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"LoRA 訓練 Ep{ep+1}"):
            imgs, labels = imgs.to(configs.DEVICE), labels.to(configs.DEVICE)
            feats  = clip.get_image_features(pixel_values=imgs)
            logits = clf(feats)
            loss   = criterion(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"LoRA Ep{ep+1} Loss: {total_loss/len(train_loader):.4f}")
    save_checkpoint(clip, clf, configs.CHECKPOINT_DIR/"lora_clip.pth")


def train_baseline():
    clip = build_lora_clip().to(configs.DEVICE)
    clf  = build_classifier().to(configs.DEVICE)
    opt  = torch.optim.AdamW(clf.parameters(), lr=configs.LR)
    criterion = nn.CrossEntropyLoss()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    train_loader, _, _, _ = make_loaders(proc)

    for ep in range(configs.EPOCHS):
        clf.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"基線訓練 Ep{ep+1}"):
            imgs, labels = imgs.to(configs.DEVICE), labels.to(configs.DEVICE)
            with torch.no_grad(): feats = clip.get_image_features(pixel_values=imgs)
            logits = clf(feats)
            loss   = criterion(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"基線 Ep{ep+1} Loss: {total_loss/len(train_loader):.4f}")
    save_checkpoint(clip, clf, configs.CHECKPOINT_DIR/"baseline_probe.pth")
