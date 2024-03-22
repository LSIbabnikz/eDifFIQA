
import os
import argparse
from contextlib import nullcontext

import wandb
from tqdm import tqdm

import torch
import torch.nn.functional as f

from loss import *
from utils import *
from dataset import *


def train(args : Arguments,
          epoch : int,
          model : torch.nn.Module,
          train_dataloader : torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer,
          loss_fn_quality : torch.nn.Module,
          loss_fn_consistency : torch.nn.Module,
          grad_scaler : torch.cuda.amp.GradScaler,
          wandb_logger,
          ) -> None:
    """ Main training function for the eDifFIQA approach.

    Args:
        args (Arguments): Arguments from the training script.
        epoch (int): Current epoch.
        model (torch.nn.Module): Training model.
        train_dataloader (torch.utils.data.DataLoader): Train dataloader.
        optimizer (torch.optim.Optimizer): Used optimizer.
        loss_fn_quality (torch.nn.Module): Quality loss function.
        loss_fn_consistency (torch.nn.Module): Consistency loss function.
        grad_scaler (torch.cuda.amp.GradScaler): Gradient scaler.
        wandb_logger (_type_): Wandb logger.
    """

    # Enable the model to return features alongside the quality estimates
    model.return_feat = 1

    model.train()
    for (image_batch, label_batch, embedding_batch, id_batch) in (pbar := tqdm(train_dataloader, 
                                           desc=f" Training Epoch ({epoch}/{args.base.epochs}), Loss: NaN ", 
                                           disable=not args.base.verbose)):

        image_batch = image_batch.to(args.base.device)
        label_batch = label_batch.to(args.base.device)
        embedding_batch = embedding_batch.to(args.base.device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast() if grad_scaler else nullcontext():
            out_embedding_batch, out_quality_batch = model(image_batch)
            out_quality_batch = out_quality_batch.squeeze()

            # Get quality factor from relative position of input sample w.r.t the precomputed class center
            with torch.no_grad():
                ad_quality = loss_fn_consistency(out_embedding_batch, 
                                              train_dataloader.dataset.average_embedding_batch[id_batch],
                                              target=torch.tensor([1]).to(args.base.device))

            # Get both loss terms
            loss_quality = loss_fn_quality(out_quality_batch, label_batch, ad_quality)
            # Default consistency loss does not have reduction -> We use torch.mean separately
            loss_consistency = torch.mean(loss_fn_consistency(out_embedding_batch, 
                                                                 embedding_batch, 
                                                                 target=torch.tensor([1]).to(args.base.device)))
        
            loss = args.loss.theta * loss_consistency + (1. - args.loss.theta) * loss_quality

        if grad_scaler:
            grad_scaler.scale(loss).backward() 
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            grad_scaler.step(optimizer) 
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        pbar.set_description(f" Training Epoch ({epoch}/{args.base.epochs}), Loss: {loss.item():.4f} ")

        if args.wandb.use:
            wandb_logger.log({"rc_loss": loss_quality.item(),
                              "q_loss":  loss_consistency.item()})



@torch.no_grad()
def validate(args : Arguments,
             best_val_loss : float,
             model : torch.nn.Module,
             val_dataloader : torch.utils.data.DataLoader,
             loss_fn_quality : torch.nn.Module,
             loss_fn_consistency : torch.nn.Module,
             grad_scaler : torch.cuda.amp.GradScaler,
             wandb_logger) -> float:
    """ Main validation function for the eDifFIQA approach.

    Args:
        args (Arguments): Arguments from the training script.
        best_val_loss (float): Current best achieved validation loss.
        model (torch.nn.Module): Trained model.
        val_dataloader (torch.utils.data.DataLoader): Validation dataloader.
        loss_fn_quality (torch.nn.Module): Quality loss function.
        loss_fn_consistency (torch.nn.Module): Consistency loss function.
        grad_scaler (torch.cuda.amp.GradScaler): Gradient scaler.
        wandb_logger (_type_): Wandb logger.

    Returns:
        float: Best recored validation loss.
    """

    per_epoch_val_loss = 0.
    model.eval()
    for (image_batch, label_batch, embedding_batch, id_batch) in tqdm(val_dataloader, 
                                                                        desc=" Validation ", 
                                                                        disable=not args.base.verbose):

        image_batch = image_batch.to(args.base.device)
        label_batch = label_batch.to(args.base.device)
        embedding_batch = embedding_batch.to(args.base.device)

        with torch.cuda.amp.autocast() if grad_scaler else nullcontext():

            out_embedding_batch, out_quality_batch = model(image_batch)
            out_quality_batch = out_quality_batch.squeeze()

            out_quality_batch = out_quality_batch.squeeze()
            loss_quality = loss_fn_quality(out_quality_batch, label_batch)

            loss_consistency = torch.mean(loss_fn_consistency(out_embedding_batch, 
                                                            embedding_batch, 
                                                            target=torch.tensor([1]).to(args.base.device)))
        
            loss = args.loss.theta * loss_consistency + (1. - args.loss.theta) * loss_quality

        per_epoch_val_loss += loss.item()

    per_epoch_val_loss = per_epoch_val_loss / len(val_dataloader)

    if args.wandb.use:
        wandb_logger.log({"val_loss": per_epoch_val_loss})

    if per_epoch_val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(args.base.save_path, f"model.pth"))
        best_val_loss = per_epoch_val_loss
    
    return best_val_loss


def main(args):

    # Seed all libraries to ensure consistency between runs.
    seed_all(args.base.seed)

    # Check if save path valid
    assert os.path.exists(args.base.save_path), f"Path {args.base.save_path} does not exist"

    # Load the training FR model and construct the transformation
    model, trans = construct_full_model(args.base.model)
    model.to(args.base.device)

    # Construct validation and training dataloaders
    train_dataset, val_dataset = construct_datasets(args.dataset, trans)
    train_dataset.average_embedding_batch = train_dataset.average_embedding_batch.to(args.base.device)
    val_dataset.average_embedding_batch = val_dataset.average_embedding_batch.to(args.base.device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   **args_to_dict(args.dataloader.train.params, {}))
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 **args_to_dict(args.dataloader.val.params, {}))

    # Create optimizer from config
    optimizer = construct_optimizer(args.optimizer, model)

    # Load desired loss function
    loss_fn_quality = load_module(args.loss.quality)
    loss_fn_consistency = load_module(args.loss.consistency)

    # Use AMP if specified in config
    grad_scaler = None
    if args.base.amp:
        grad_scaler = torch.cuda.amp.GradScaler()

    # Construct WANDB logger
    wandb_logger = None
    if args.wandb.use:
        wandb_logger = wandb.init(project=args.wandb.project, config={"args": args_to_dict(args, {})})

    # Train loop
    best_val_loss = float("inf")
    for epoch in range(args.base.epochs):

        train(args, epoch, model, train_dataloader, optimizer, loss_fn_quality, loss_fn_consistency, grad_scaler, wandb_logger)

        best_val_loss = validate(args, best_val_loss, model, val_dataloader, loss_fn_quality, loss_fn_consistency, grad_scaler, wandb_logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config",
        type=str,
        help=' Location of the eDifFIQA training configuration. '
    )
    args = parser.parse_args()
    arguments = parse_config_file(args.config)

    main(arguments)