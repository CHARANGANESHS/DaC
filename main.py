import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from collections import defaultdict
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from tqdm import tqdm
from medmnist import OrganSMNIST, OrganAMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")


batch_size = 32
source_epochs = 5
adapt_epochs = 5
learning_rate = 0.01
momentum = 0.9
num_classes = 11

source_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

weak_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

strong_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class OrganTargetDataset(Dataset):
    def __init__(self, dataset, transform_weak, transform_strong):
        self.dataset = dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_w = self.transform_weak(img)
        img_s = self.transform_strong(img)
        return img_w, img_s, int(label)


class SFUDAModel(nn.Module):
    def __init__(self, num_classes):
        super(SFUDAModel, self).__init__()
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        return features, logits

class MemoryBank:
    def __init__(self, feature_dim, dataset_size, momentum=0.2):
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.bank = torch.zeros(dataset_size, feature_dim).to(device)
    
    def update(self, indices, features):
        with torch.no_grad():
            self.bank[indices] = self.momentum * self.bank[indices] + (1 - self.momentum) * features
            self.bank[indices] = F.normalize(self.bank[indices], dim=1)
    
    def get_features(self):
        return self.bank
    
    
def self_training_loss(p_w, p_s, pseudo_label, num_classes, lambda_diversity=0.1, omega_entropy=0.1):
    ce_loss = F.cross_entropy(p_w, pseudo_label.long()) + F.cross_entropy(p_s, pseudo_label.long())
    mean_pred = p_w.mean(dim=0)
    diversity_loss = F.kl_div(mean_pred.log(), torch.full_like(mean_pred, 1/num_classes), reduction='batchmean')
    entropy_loss = - (p_w * p_w.log()).sum(dim=1).mean()
    return ce_loss + lambda_diversity * diversity_loss + omega_entropy * entropy_loss

def contrastive_loss(anchor, positive, negatives, temperature=0.05):
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=1)
    
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = torch.matmul(anchor, negatives.t()) / temperature
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(device)
    loss = F.cross_entropy(logits, labels)
    return loss

def mmd_loss(source_features, target_features):
    mean_source = source_features.mean(dim=0)
    mean_target = target_features.mean(dim=0)
    loss = torch.norm(mean_source - mean_target, p=2)**2
    return loss

def compute_centroids(features, pseudo_labels, num_classes):
    centroids = []
    for c in range(num_classes):
        mask = (pseudo_labels == c)
        if mask.sum() == 0:
            centroid = torch.zeros(features.size(1)).to(device)
        else:
            centroid = features[mask].mean(dim=0)
            centroid = F.normalize(centroid.unsqueeze(0), dim=1).squeeze(0)
        centroids.append(centroid)
    centroids = torch.stack(centroids, dim=0)
    return centroids

def assign_pseudo_labels(features, centroids):
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, centroids.t())
    pseudo_labels = sim.argmax(dim=1)
    confidence, _ = sim.max(dim=1)
    return pseudo_labels, confidence

def divide_samples(confidence, threshold=0.8):
    source_like_idx = (confidence >= threshold).nonzero(as_tuple=False).squeeze()
    target_specific_idx = (confidence < threshold).nonzero(as_tuple=False).squeeze()
    return source_like_idx, target_specific_idx

def update_pseudo_labels(model, dataloader, tau=0.8):
    model.eval()
    all_features = []
    all_indices = []
    with torch.no_grad():
        for batch_idx, (img_w, _, _) in enumerate(dataloader):
            img_w = img_w.to(device)
            features, _ = model(img_w)
            features = F.normalize(features, dim=1)
            all_features.append(features)
            indices = torch.arange(batch_idx * batch_size, min((batch_idx+1)*batch_size, len(dataloader.dataset))).to(device)
            all_indices.append(indices)
    all_features = torch.cat(all_features, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    pseudo_labels = all_features.argmax(dim=1)  # initial pseudo-labels
    centroids = compute_centroids(all_features, pseudo_labels, num_classes)
    pseudo_labels, confidence = assign_pseudo_labels(all_features, centroids)
    source_like_idx, target_specific_idx = divide_samples(confidence, threshold=tau)
    model.train()
    return all_indices, pseudo_labels, confidence, source_like_idx, target_specific_idx, centroids



print("Starting source training on OrganSMNIST train split...")
source_dataset_full = OrganSMNIST(split='train', transform=source_transform, download=True)
source_loader = DataLoader(source_dataset_full, batch_size=batch_size, shuffle=True, num_workers=0)

source_model = SFUDAModel(num_classes=num_classes).to(device)
optimizer_source = optim.SGD(source_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)

for epoch in range(source_epochs):
    running_loss = 0.0
    for img, labels in tqdm(source_loader, desc=f"Source Epoch {epoch+1}/{source_epochs}", ncols=80):
        # print(torch.unique(labels))
        # break
        img = img.to(device)
        labels = labels.squeeze().to(device)
        features, logits = source_model(img)
        loss = F.cross_entropy(logits, labels)
        optimizer_source.zero_grad()
        loss.backward()
        optimizer_source.step()
        running_loss += loss.item()
    print(f"Source Epoch [{epoch+1}/{source_epochs}], Loss: {running_loss/len(source_loader):.4f}")

torch.save(source_model.state_dict(), "../models/source_model_organsmnist.pth")
print("Source training finished and model saved.")


adapted_model = SFUDAModel(num_classes=num_classes).to(device)
adapted_model.load_state_dict(torch.load("../models/source_model_organsmnist.pth"))

for param in adapted_model.fc.parameters():
    param.requires_grad = False

target_dataset_full = OrganAMNIST(split='train', transform=None, download=True)
target_dataset = OrganTargetDataset(target_dataset_full, transform_weak=weak_transform, transform_strong=strong_transform)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

dataset_size = len(target_dataset)
feature_dim = 2048
memory_bank = MemoryBank(feature_dim, dataset_size, momentum=0.2)

optimizer_adapt = optim.SGD(adapted_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
alpha = 1.0
beta = 0.5
gamma = 1.0
dataset_indices = np.arange(dataset_size)

print("Starting adaptation on OrganAMNIST train split...")
for epoch in range(adapt_epochs):
    all_indices, pseudo_labels, confidence, src_like_idx, tgt_spec_idx, centroids = update_pseudo_labels(adapted_model, target_loader, tau=0.8)
    running_loss = 0.0
    for batch_idx, (img_w, img_s, _) in enumerate(tqdm(target_loader, desc=f"Adapt Epoch {epoch+1}/{adapt_epochs}", ncols=80)):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + img_w.size(0)
        batch_indices = torch.tensor(dataset_indices[batch_start:batch_end]).to(device)
        
        img_w = img_w.to(device)
        img_s = img_s.to(device)
        
        features_w, logits_w = adapted_model(img_w)
        features_s, logits_s = adapted_model(img_s)
        p_w = F.softmax(logits_w, dim=1)
        p_s = F.softmax(logits_s, dim=1)
        batch_pseudo = pseudo_labels[batch_indices]
        
        loss_self = self_training_loss(p_w, p_s, batch_pseudo, num_classes)
        batch_features = F.normalize(features_w, dim=1)
        positive_proto = centroids[batch_pseudo]
        bank_features = memory_bank.get_features()
        
        loss_contrastive = 0.0
        for i in range(batch_features.size(0)):
            neg_mask = (pseudo_labels != batch_pseudo[i])
            negatives = bank_features[neg_mask]
            if negatives.size(0) > 0:
                loss_contrastive += contrastive_loss(batch_features[i].unsqueeze(0),
                                                        positive_proto[i].unsqueeze(0),
                                                        negatives)
        loss_contrastive = loss_contrastive / batch_features.size(0)
        
        src_mask = torch.tensor([idx.item() in src_like_idx.cpu().numpy() for idx in batch_indices]).bool()
        tgt_mask = torch.tensor([idx.item() in tgt_spec_idx.cpu().numpy() for idx in batch_indices]).bool()
        if src_mask.sum() > 0 and tgt_mask.sum() > 0:
            src_feats = batch_features[src_mask]
            tgt_feats = batch_features[tgt_mask]
            loss_mmd = mmd_loss(src_feats, tgt_feats)
        else:
            loss_mmd = torch.tensor(0.0).to(device)
        
        loss = alpha * loss_self + gamma * loss_contrastive + beta * loss_mmd
        
        optimizer_adapt.zero_grad()
        loss.backward()
        optimizer_adapt.step()
        
        memory_bank.update(batch_indices, batch_features.detach())
        running_loss += loss.item()
    print(f"Adapt Epoch [{epoch+1}/{adapt_epochs}], Loss: {running_loss/len(target_loader):.4f}")


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img_w, _, labels in dataloader:
            img_w = img_w.to(device)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            labels = labels.to(device)
            _, logits = model(img_w)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


accuracy = evaluate(adapted_model, target_loader)
print(f"Adapted Target Domain Accuracy on OrganAMNIST Train Subset: {accuracy:.2f}%")