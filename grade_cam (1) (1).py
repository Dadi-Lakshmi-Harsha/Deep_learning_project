import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import cv2
torch.backends.cudnn.benchmark = True
import kagglehub
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
train_base=os.path.join(
    path,
    "New Plant Diseases Dataset(Augmented)",
    "New Plant Diseases Dataset(Augmented)",)
train_dir=os.path.join(train_base, "train")
valid_dir=os.path.join(train_base, "valid")
test_dir=os.path.join(path, "New Plant Diseases Dataset(Augmented)", "test")
Diseases_classes=sorted(os.listdir(train_dir))
print("Total number of classes:", len(Diseases_classes))
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
train_dataset=ImageFolder(train_dir, transform=transform)
valid_dataset=ImageFolder(valid_dir, transform=transform)
val_size =len(valid_dataset) //2
test_size=len(valid_dataset) - val_size
val_dataset,test_dataset=random_split(valid_dataset, [val_size, test_size])
img,label=train_dataset[0]
print(img.shape,label)
train_loader=DataLoader(train_dataset, batch_size=16, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader=DataLoader(val_dataset,   batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader=DataLoader(test_dataset,  batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=True)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)
class GradCAMCNN(nn.Module):
    def __init__(self, num_classes: int = 38, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.layer1=ConvBlock(3,   64,  pool=True)
        self.layer2=ConvBlock(64,  128, pool=True)
        self.layer3=ConvBlock(128, 256, pool=True)
        self.layer4=ConvBlock(256, 512, pool=True)   # GradCAM target
        self.gap =nn.AdaptiveAvgPool2d(1)
        self.head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.gradients:   torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module,input,output):
            self.activations=output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients=grad_out[0].detach()
        target_conv=self.layer4.block[0]
        target_conv.register_forward_hook(forward_hook)
        target_conv.register_full_backward_hook(backward_hook)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return self.head(x)
    def generate_gradcam(self, input_tensor, target_class=None,
                         device=torch.device("cpu")):
        self.eval()
        input_tensor=input_tensor.to(device)
        output = self.forward(input_tensor)
        if target_class is None:
            target_class=output.argmax(dim=1).item()
        self.zero_grad()
        one_hot=torch.zeros_like(output)
        one_hot[0, target_class]=1.0
        output.backward(gradient=one_hot)
        weights=self.gradients.mean(dim=(2, 3),keepdim=True)
        cam=(weights * self.activations).sum(dim=1,keepdim=True)
        cam=F.relu(cam)
        cam=F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam=cam.squeeze().cpu().numpy()
        cam_min,cam_max=cam.min(),cam.max()
        if cam_max - cam_min > 1e-8:
            cam=(cam - cam_min) / (cam_max - cam_min)
        else:
            cam=np.zeros_like(cam)
        return cam.astype(np.float32)
def overlay_gradcam(original_img, heatmap, alpha=0.5,
                    colormap=cv2.COLORMAP_JET):
    heatmap_uint8=np.uint8(255 * heatmap)
    heatmap_color=cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color=cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay=np.clip(
        alpha*heatmap_color.astype(np.float32) +
        (1-alpha)* original_img.astype(np.float32),
        0, 255,
    ).astype(np.uint8)
    return overlay
def visualize_gradcam_batch(model,dataset,class_names,device,
                            num_images=8, save_path="gradcam_results.png"):
    model.eval()
    indices = np.random.choice(len(dataset), num_images, replace=False)
    inv_normalize=transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    fig,axes=plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    fig.suptitle("GradCAM Visualizations", fontsize=16)
    for row,idx in enumerate(indices):
        img_tensor,true_label=dataset[idx]
        inp=img_tensor.unsqueeze(0)
        heatmap=model.generate_gradcam(inp, device=device)
        with torch.no_grad():
            logits=model(inp.to(device))
        pred_label=logits.argmax(dim=1).item()
        orig=inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        orig=np.clip(orig, 0, 1)
        orig_uint8=(orig * 255).astype(np.uint8)
        heatmap_uint8=np.uint8(255 * heatmap)
        heatmap_color=cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color=cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay=overlay_gradcam(orig_uint8, heatmap)
        axes[row, 0].imshow(orig_uint8)
        axes[row, 0].set_title(
            f"Original\nTrue: {class_names[true_label]}", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(heatmap_color)
        axes[row, 1].set_title("GradCAM heatmap", fontsize=8)
        axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"Overlay\nPred: {class_names[pred_label]}", fontsize=8)
        axes[row, 2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"GradCAM visualisation saved → {save_path}")
def run_epoch(model, loader, criterion, optimizer=None,
              device="cpu", grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss,correct,total = 0.0,0,0
    context=torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs=model(imgs)
            loss =criterion(outputs, labels)
            if is_train:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss +=loss.item()
            _, preds=outputs.max(1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
            if i % 50==0:
                print(f"Batch: {i}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss/len(loader), correct/total
def train_model(model,train_loader,val_loader,config,device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler=config["scheduler"](optimizer)
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc=0.0
    best_weights=None
    for epoch in range(config["epochs"]):
        train_loss, train_acc = run_epoch(
            model,train_loader,criterion,optimizer,
            device, config["grad_clip"],
)
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
        )
        scheduler.step()
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            best_weights={k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
    return best_val_acc,best_weights
def evaluate_test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs,labels in test_loader:
            imgs,labels = imgs.to(device), labels.to(device)
            preds=model(imgs).argmax(dim=1)
            correct+=(preds == labels).sum().item()
            total +=labels.size(0)
    return correct/total
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_scheduler(name, epochs):
    if name == "cosine":
        return lambda opt: CosineAnnealingLR(opt, T_max=epochs)
    elif name == "step":
        return lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=5, gamma=0.5)
csv_file = "results_gradcam_grid.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "weight_decay", "dropout", "grad_clip",
                         "epochs", "val_acc", "test_acc"])
def save_result(config, val_acc, test_acc):
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([config["lr"], config["weight_decay"],
                         config["dropout"], config["grad_clip"],
                         config["epochs"],
                         round(val_acc, 5), round(test_acc, 5)])
search_space = {
    "lr":           [5e-4, 1e-3],
    "weight_decay": [1e-4, 1e-3],
    "dropout":      [0.3, 0.5],
    "grad_clip":    [1.0, 2.0],
    "epochs":       5,
}
if __name__ == "__main__":
    best_overall=0.0
    best_config=None
    best_weights_overall=None
    best_model_overall=None
    for lr in search_space["lr"]:
        for wd in search_space["weight_decay"]:
            for dropout in search_space["dropout"]:
                for clip in search_space["grad_clip"]:
                    print(f"\nRunning config: lr={lr}, wd={wd}, "
                        f"dropout={dropout}, clip={clip}")
                    config = {
                        "lr":           lr,
                        "weight_decay": wd,
                        "dropout":      dropout,
                        "grad_clip":    clip,
                        "epochs":       search_space["epochs"],
                        "scheduler":    get_scheduler("cosine", search_space["epochs"]),
                    }
                    model = GradCAMCNN(
                        num_classes=len(Diseases_classes),
                        dropout=dropout,
                    ).to(DEVICE)
                    val_acc, best_weights = train_model(
                        model, train_loader, val_loader, config, DEVICE)
                    model.load_state_dict(best_weights)
                    model.to(DEVICE)
                    test_acc = evaluate_test(model, test_loader, DEVICE)
                    print(f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
                    save_result(config, val_acc, test_acc)
                    if test_acc>best_overall:
                        best_overall=test_acc
                        best_config =config.copy()
                        best_weights_overall =best_weights
                        best_model_overall=model
    print("\nBest Accuracy:", best_overall)
    print("Best Config:", best_config)
    best_model_overall.load_state_dict(best_weights_overall)
    best_model_overall.to(DEVICE)
    best_model_overall.eval()
    torch.save(
        {
            "model_state_dict": best_weights_overall,
            "config":           best_config,
            "test_acc":         best_overall,
            "classes":          Diseases_classes,
        },
        "best_gradcam_model.pth",
    )
    print("Best model saved → best_gradcam_model.pth")
    visualize_gradcam_batch(
        model       =best_model_overall,
        dataset     =valid_dataset,
        class_names =Diseases_classes,
        device      =DEVICE,
        num_images  =8,
        save_path   ="gradcam_results.png",
    )
    results_df =pd.read_csv(csv_file)
    results_df =results_df.sort_values("test_acc", ascending=False)
    print("\nGrid search results (sorted by test_acc):")
    print(results_df.to_string(index=False))
    results_df.to_csv("results_gradcam_grid_sorted.csv", index=False)
    print("Sorted results saved → results_gradcam_grid_sorted.csv")