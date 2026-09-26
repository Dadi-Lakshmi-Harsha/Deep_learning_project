import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True



import kagglehub
import os

# Download latest version
path=kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

# Adjust paths based on the actual downloaded dataset structure
# The dataset typically has a nested structure like:
# <download_path>/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/
train_base=os.path.join(path, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)")
train_dir=os.path.join(train_base, "train")
valid_dir=os.path.join(train_base, "valid")
# The test directory might be at a different level, often directly under the first 'New Plant Diseases Dataset(Augmented)'
# or directly under the downloaded path. Assuming it's under the first 'New Plant Diseases Dataset(Augmented)'
test_dir=os.path.join(path,"New Plant Diseases Dataset(Augmented)", "test")
Diseases_classes=os.listdir(train_dir)
print(Diseases_classes)
print("\nTotal number of classes are: ", len(Diseases_classes))
plt.figure(figsize=(60,60), dpi=200)
cnt=0
plant_names=[]
tot_images=0
for i in Diseases_classes:
    cnt+=1
    plant_names.append(i)
    plt.subplot(7,7,cnt)
    image_path =os.listdir(train_dir + "/" + i)
    print("The Number of Images in " +i+ ":", len(image_path), end= " ")
    tot_images +=len(image_path)
    img_show = plt.imread(train_dir + "/" + i + "/" + image_path[0])
    plt.imshow(img_show)
    plt.xlabel(i,fontsize=30)
    plt.xticks([])
    plt.yticks([])
    print("\nTotal Number of Images in Directory: ", tot_images)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train=ImageFolder(train_dir, transform=transform)
valid=ImageFolder(valid_dir, transform=transform)
val_size=len(valid)//2
test_size=len(valid)-val_size
val_dataset,test_dataset=random_split(valid, [val_size, test_size])
img,label=train[0]
print(img.shape, label)
train_loader=DataLoader(train, batch_size=16, shuffle=True)
val_loader =DataLoader(val_dataset,batch_size=16, shuffle=False)
test_loader=DataLoader(test_dataset, batch_size=16, shuffle=False)
class ViTConfig:
    def __init__(self):
        self.image_size=128
        self.patch_size=16
        self.in_channels=3
        self.embed_dim=384
        self.num_heads=6
        self.num_layers=4
        self.mlp_ratio=4.0
        self.dropout=0.1
        self.attn_dropout=0.0
        self.num_classes=38
        self.num_patches=(self.image_size//self.patch_size)**2
        self.seq_len=self.num_patches+1    
class PatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_size=cfg.patch_size
        # using conv to extract patches + project them
        self.proj=nn.Conv2d(
            cfg.in_channels,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size)
        self.norm=nn.LayerNorm(cfg.embed_dim)
    def forward(self, x):
        # (B, 3, 256, 256) → (B, embed_dim, 16, 16)
        x =self.proj(x)
        # flatten patches
        x =x.flatten(2)      # (B, embed_dim, 256)
        x =x.transpose(1, 2) # (B, 256, embed_dim)
        x =self.norm(x)
        return x
import math
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads=cfg.num_heads
        self.head_dim=cfg.embed_dim // cfg.num_heads
        self.scale=1/math.sqrt(self.head_dim)
        # one layer for q, k, v together
        self.qkv=nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=False)
        self.out=nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.attn_drop=nn.Dropout(cfg.attn_dropout)
        self.proj_drop=nn.Dropout(cfg.dropout)
    def forward(self, x):
        B,N,C=x.shape
        qkv=self.qkv(x)   # (B, N, 3*C)
        qkv=qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv=qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q,k,v=qkv[0], qkv[1], qkv[2]
        # attention scores
        attn=(q @ k.transpose(-2, -1))*self.scale
        attn=torch.softmax(attn, dim=-1)
        attn=self.attn_drop(attn)
        # apply attention to values
        x=attn@v   # (B, heads, N, head_dim)
        x=x.transpose(1, 2).reshape(B, N, C)
        x=self.out(x)
        x=self.proj_drop(x)
        return x
class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation and dropout.
    hidden_dim = embed_dim × mlp_ratio  (default 4×)
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        hidden=int(cfg.embed_dim * cfg.mlp_ratio)
        self.net=nn.Sequential(
            nn.Linear(cfg.embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.norm1=nn.LayerNorm(cfg.embed_dim)
        self.attn=MultiHeadSelfAttention(cfg)
        self.norm2=nn.LayerNorm(cfg.embed_dim)
        self.ffn=FeedForward(cfg)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x =x + self.attn(self.norm1(x))   # residual around MHSA
        x =x + self.ffn(self.norm2(x))    # residual around FFN
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # patch embedding
        self.patch_embed=PatchEmbedding(cfg)
        # cls token
        self.cls_token=nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        # positional embedding
        self.pos_embed=nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.embed_dim))
        self.dropout=nn.Dropout(cfg.dropout)
        # transformer blocks
        self.blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        # final norm
        self.norm=nn.LayerNorm(cfg.embed_dim)
        # classifier head
        self.head=nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 3),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim // 3, cfg.num_classes))
        self.init_weights()
    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        B =x.shape[0]
        # patch embeddings
        x =self.patch_embed(x)   # (B, 256, embed_dim)
        # add cls token
        cls=self.cls_token.expand(B, -1, -1)
        x=torch.cat([cls, x], dim=1)   # (B, 257, embed_dim)
        # add position info
        x=x + self.pos_embed
        x=self.dropout(x)
        # transformer
        x=self.blocks(x)
        x=self.norm(x)
        # take cls token
        x=x[:, 0]
        return self.head(x)


DEVICE =torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg  =ViTConfig()                   # all defaults above
model =VisionTransformer(cfg).to(DEVICE)
total =sum(p.numel() for p in model.parameters())
train =sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params   : {total:,}")   # ~86M
print(f"Trainable      : {train:,}")
dummy =torch.randn(4, 3, 128, 128).to(DEVICE)
out   =model(dummy)
print(f"Output shape   : {out.shape}")
def run_epoch(model, loader, criterion, optimizer=None, device="cpu", grad_clip=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss =0
    correct =0
    total =0
    for i,(imgs, labels) in enumerate(loader):
        imgs, labels =imgs.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()
        outputs =model(imgs)
        loss =criterion(outputs, labels)
        if is_train:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss+=loss.item()
        _, preds=torch.max(outputs, 1)
        correct+=(preds == labels).sum().item()
        total+=labels.size(0)
        if i%50==0:
            print(f"Batch: {i}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / len(loader), correct / total
def train_model(model, train_loader, val_loader, config, device):
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"])
    scheduler=config["scheduler"](optimizer)
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc=0
    best_weights=None
    for epoch in range(config["epochs"]):
        train_loss,train_acc = run_epoch(
            model,train_loader,criterion,
            optimizer,device,config["grad_clip"])
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion,
            optimizer=None, device=device)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc>best_acc:
            best_acc=val_acc
            best_weights=model.state_dict()
    return best_acc, best_weights
def evaluate_test(model,test_loader,device):
    model.eval()
    correct,total = 0,0
    with torch.no_grad():
        for imgs,labels in test_loader:
            imgs,labels=imgs.to(device), labels.to(device)
            outputs=model(imgs)
            preds=outputs.argmax(dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
    return correct/total
def get_scheduler(name, epochs):
    if name =="cosine":
        return lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif name== "step":
        return lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
def save_result(config, val_acc, test_acc):
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            config["lr"],
            config["weight_decay"],
            config["grad_clip"],
            config["epochs"],
            val_acc,
            test_acc
        ])
csv_file = "results_grid.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lr", "weight_decay", "grad_clip",
            "epochs", "val_acc", "test_acc"])
        

search_space = {
    "lr": [ 1e-4,2e-4],          # safe range for ViT
    "weight_decay": [0.01,0.1], # regularization
    "grad_clip": [0.5,1.0],           # stabilize training
    "epochs": 5
}
best_overall =0
best_config =None
for lr in search_space["lr"]:
    for wd in search_space["weight_decay"]:
        for clip in search_space["grad_clip"]:
            print("\nRunning config:",
                  f"lr={lr}, wd={wd}, clip={clip}")
            config = {
                "lr": lr,
                "weight_decay": wd,
                "grad_clip": clip,
                "epochs": search_space["epochs"],
                "scheduler": get_scheduler("cosine", 5),
            }
            model = VisionTransformer(cfg).to(DEVICE)
            val_acc, best_weights = train_model(model, train_loader, val_loader, config, DEVICE)
            model.load_state_dict(best_weights)
            test_acc =evaluate_test(model, test_loader, DEVICE)
            print(f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            save_result(config, val_acc, test_acc)
            if test_acc >best_overall:
                best_overall= test_acc
                best_config= config
                best_weights_overall =best_weights
print("\nBest Accuracy:", best_overall)
print("Best Config:", best_config)
