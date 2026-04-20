import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR
import math
import json


from contrast_model import FFNNContrastiveModel


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """
    Linear warmup then cosine decay.
    min_lr_ratio: floor for LR as a fraction of peak LR (e.g. 0.1 = decay to 10% of peak)
    """
    def lr_lambda(current_step):

        # Linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)

        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def average_by_index(embeddings, indices):
    """ If multiple rows in embeddings correspond to the same index in indices, average those rows together. """
    unique_indices = indices.unique()
    averaged = torch.zeros(len(unique_indices), embeddings.shape[-1], device=embeddings.device)
    for i, idx in enumerate(unique_indices):
        mask = indices == idx
        averaged[i] = embeddings[mask].mean(dim=0)
    return averaged, unique_indices

def siglip_loss(model, emb_a, emb_b, idx_a, idx_b, dataset, type = 'drug-protein'):
    """
    Calculates sigmoid contrastive loss for a batch of drug and gene embeddings. Uses labels from dataset to upweight positive pairs and downweight negative pairs.

    Args:
        model: contrastive model being trained, used to access temperature and bias parameters
        emb_a: [batch, embed_dim] projected embeddings for entity A (e.g., drugs)
        emb_b: [batch, embed_dim] projected embeddings for entity B (e.g., genes)
        idx_a: [batch] indices of entity A in the similarity matrix (e.g., drug indices)
        idx_b: [batch] indices of entity B in the similarity matrix (e.g., gene indices)
        dataset: dataset object containing label matrices
        type: type of pairs to get labels for - "drug-protein", "drug", or "protein"
    """
    # Normalize embeddings so dotproduct is cosine similarity
    emb_a_norm = F.normalize(emb_a, dim=-1)
    emb_b_norm = F.normalize(emb_b, dim=-1)

    # Get rid of duplicate pairs
    emb_a_unique, idx_a_unique = average_by_index(emb_a_norm, idx_a)
    emb_b_unique, idx_b_unique = average_by_index(emb_b_norm, idx_b)

    # Get labels for unique pairs from dataset
    weight_matrix = dataset.get_label_matrix(idx_a_unique, idx_b_unique, type = type)

    # Cosine similarity scaled by temperature and shifted by bias
    if type == 'drug-protein':
        logits = (emb_a_unique @ emb_b_unique.T) * model.temps.exp()[0] + model.biases[0]  # [batch]
    elif type == 'drug':
        logits = (emb_a_unique @ emb_a_unique.T) * model.temps.exp()[1] + model.biases[1]  # [batch]
    elif type == 'protein':
        logits = (emb_b_unique @ emb_b_unique.T) * model.temps.exp()[2] + model.biases[2]  # [batch]

    # sigmoid contrastive loss with upweighting of positive pairs and downweighting of negative pairs based on labels in weight_matrix
    loss = -F.logsigmoid(weight_matrix * logits).mean()

    return loss

base_parameters = {'max_learning_rate': 1e-4, 'epochs': 100, 'batch_size': 512, 'warmup_fraction': 0.05, 'drug_lambda': 0.1, 'protein_lambda': 0.1}

def train_contrastive_limited(model, dataset, output_directory, params = base_parameters, 
                      device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float32):
    """ Trains model on training dataset using contrastive loss. Saves model every 10 epochs. Logs loss information in output_directory/training_log.txt. """
    
    model.to(device)

    # building dataloaders for training and validation datasets
    dataset.drug_weights = dataset.drug_weights.to(device)
    dataset.protein_weights = dataset.protein_weights.to(device)
    dataset.drug_protein_weights = dataset.drug_protein_weights.to(device)

    train_dataset = Subset(dataset, dataset.train_indices)
    valid_dataset = Subset(dataset, dataset.val_indices)
    train_loader = DataLoader(dataset = train_dataset, batch_size = params["batch_size"], shuffle = True)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = params["batch_size"], shuffle = False)

    # defining loss and optimizer
    total_steps = params["epochs"] * len(train_loader)
    warmup_steps = int(params["warmup_fraction"] * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr = params["max_learning_rate"], weight_decay = 1e-4)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # begin training
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_loss_dp = 0.0
        total_loss_dd = 0.0
        total_loss_pp = 0.0

        for step, batch in enumerate(train_loader):

            # moving to device
            drug_index, drug_embedding, gene_index, gene_embedding = batch
            drug_embedding, gene_embedding = drug_embedding.to(device), gene_embedding.to(device)
            drug_index, gene_index = drug_index.to(device), gene_index.to(device)

            # getting projected embeddings
            optimizer.zero_grad()
            drug_emb, gene_emb = model(drug_embedding, gene_embedding)

            # calculating multi-task loss
            contrastive_loss = siglip_loss(model, drug_emb, gene_emb, drug_index, gene_index, dataset, type = 'drug-protein')
            drug_loss = siglip_loss(model, drug_emb, drug_emb.detach(), drug_index, drug_index, dataset, type = 'drug')
            protein_loss = siglip_loss(model, gene_emb, gene_emb.detach(), gene_index, gene_index, dataset, type = 'protein')
            loss = contrastive_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_loss_dp += contrastive_loss.item()
            total_loss_dd += drug_loss.item()
            total_loss_pp += protein_loss.item()

        n = len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        entry = {
        "epoch": epoch,
        "total_loss": total_loss / n,
        "drug-protein_loss": total_loss_dp / n,
        "drug-drug_loss": total_loss_dd / n,
        "protein-protein_loss": total_loss_pp / n,
        "lr": current_lr,
        "temps": model.temps.exp().detach().cpu().numpy().tolist(),
        "biases": model.biases.detach().cpu().numpy().tolist()}

        with open(f"{output_directory}/training_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

        if epoch % 10 == 0 or epoch == params["epochs"]:
            torch.save(model.state_dict(), f"{output_directory}/model_epoch_{epoch}.pt")
        if epoch % 5 == 0:
            valid_contrastive(model, valid_loader, dataset, output_directory, epoch, params, device, dtype)
    
    return model

def train_contrastive(model, dataset, output_directory, params = base_parameters, 
                      device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float32):
    """ Trains model on training dataset using contrastive loss. Saves model every 10 epochs. Logs loss information in output_directory/training_log.txt. """
    
    model.to(device)

    # building dataloaders for training and validation datasets
    dataset.drug_weights = dataset.drug_weights.to(device)
    dataset.protein_weights = dataset.protein_weights.to(device)
    dataset.drug_protein_weights = dataset.drug_protein_weights.to(device)

    train_dataset = Subset(dataset, dataset.train_indices)
    valid_dataset = Subset(dataset, dataset.val_indices)
    train_loader = DataLoader(dataset = train_dataset, batch_size = params["batch_size"], shuffle = True)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = params["batch_size"], shuffle = False)

    # defining loss and optimizer
    total_steps = params["epochs"] * len(train_loader)
    warmup_steps = int(params["warmup_fraction"] * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr = params["max_learning_rate"], weight_decay = 1e-4)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # begin training
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_loss_dp = 0.0
        total_loss_dd = 0.0
        total_loss_pp = 0.0

        for step, batch in enumerate(train_loader):

            # moving to device
            drug_index, drug_embedding, gene_index, gene_embedding = batch
            drug_embedding, gene_embedding = drug_embedding.to(device), gene_embedding.to(device)
            drug_index, gene_index = drug_index.to(device), gene_index.to(device)

            # getting projected embeddings
            optimizer.zero_grad()
            drug_emb, gene_emb = model(drug_embedding, gene_embedding)

            # calculating multi-task loss
            contrastive_loss = siglip_loss(model, drug_emb, gene_emb, drug_index, gene_index, dataset, type = 'drug-protein')
            drug_loss = siglip_loss(model, drug_emb, drug_emb.detach(), drug_index, drug_index, dataset, type = 'drug')
            protein_loss = siglip_loss(model, gene_emb, gene_emb.detach(), gene_index, gene_index, dataset, type = 'protein')
            loss = contrastive_loss + params['drug_lambda'] * drug_loss + params['protein_lambda'] * protein_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_loss_dp += contrastive_loss.item()
            total_loss_dd += drug_loss.item()
            total_loss_pp += protein_loss.item()

        n = len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        entry = {
        "epoch": epoch,
        "total_loss": total_loss / n,
        "drug-protein_loss": total_loss_dp / n,
        "drug-drug_loss": total_loss_dd / n,
        "protein-protein_loss": total_loss_pp / n,
        "lr": current_lr,
        "temps": model.temps.exp().detach().cpu().numpy().tolist(),
        "biases": model.biases.detach().cpu().numpy().tolist()}

        with open(f"{output_directory}/training_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

        if epoch % 10 == 0 or epoch == params["epochs"]:
            torch.save(model.state_dict(), f"{output_directory}/model_epoch_{epoch}.pt")
        if epoch % 5 == 0:
            valid_contrastive(model, valid_loader, dataset, output_directory, epoch, params, device, dtype)
    
    return model

def valid_contrastive(model, valid_dataloader, full_dataset, output_directory, epoch, params = base_parameters, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float32):
    """ Evaluates model on validation dataset using contrastive loss. Returns average loss across all batches. """
    model.eval()
    total_loss = 0.0
    total_loss_dp = 0.0
    total_loss_dd = 0.0
    total_loss_pp = 0.0

    with torch.no_grad():
        for step, batch in enumerate(valid_dataloader):

            # moving to device
            drug_index, drug_embedding, gene_index, gene_embedding = batch
            drug_embedding, gene_embedding = drug_embedding.to(device), gene_embedding.to(device)
            drug_index, gene_index = drug_index.to(device), gene_index.to(device)

            # getting projected embeddings
            drug_emb, gene_emb = model(drug_embedding, gene_embedding)

            # calculating multi-task loss
            contrastive_loss = siglip_loss(model, drug_emb, gene_emb, drug_index, gene_index, full_dataset, type = 'drug-protein')
            drug_loss = siglip_loss(model, drug_emb, drug_emb.detach(), drug_index, drug_index, full_dataset, type = 'drug')
            protein_loss = siglip_loss(model, gene_emb, gene_emb.detach(), gene_index, gene_index, full_dataset, type = 'protein')
            loss = (contrastive_loss + params['drug_lambda'] * drug_loss + params['protein_lambda'] * protein_loss)
            total_loss += loss.item()
            total_loss_dp += contrastive_loss.item()
            total_loss_dd += drug_loss.item() 
            total_loss_pp += protein_loss.item()
            
        n = len(valid_dataloader)

        entry = {
            "epoch": epoch,
            "total_loss": total_loss / n,
            "drug-protein_loss": total_loss_dp / n,
            "drug-drug_loss": total_loss_dd / n,
            "protein-protein_loss": total_loss_pp / n
        }

        with open(f"{output_directory}/validation_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    return model