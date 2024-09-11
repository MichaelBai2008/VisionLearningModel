import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Config import Config
from datetime import datetime
from Image_Imbedding import ImageEmbedding
from Image_Transformer import ImgTransformer
from Image_Util import show_img_tensor_CHW
from Fliker_Comment_Tokenizer import FlikerCommentTokenizer
from Fliker_Image_Comment_Dataset import ImgCommentDataset
from Model_Util import count_parameters
from pathlib import Path
from text_token_embedding import TextTokenEmbedding
from text_casual_mask_transformer import TextMaskedTransformer
from vlm_model import ImgLanguageModel
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF


BATCH_SIZE = 20

config = Config()


def create_datasets(config: Config):
    train_dataset = ImgCommentDataset(config, split="train")
    eval_dataset = ImgCommentDataset(config, split="eval")
    test_dataset = ImgCommentDataset(config, split="test")
    print(f"train_dataset:  {len(train_dataset)}")
    print(f"eval_dataset:  {len(eval_dataset)}")
    print(f"test_dataset:  {len(test_dataset)}")

    return train_dataset, eval_dataset, test_dataset


def create_dataloaders(
    train_dataset: ImgCommentDataset,
    eval_dataset: ImgCommentDataset,
    test_dataset: ImgCommentDataset,
    batch_size: int = BATCH_SIZE,
):
    assert train_dataset is not None
    assert eval_dataset is not None
    assert test_dataset is not None
    assert batch_size > 0

    # Data Loader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"train_dataloader:  {len(train_dataloader)}")
    print(f"eval_data_loader:  {len(eval_dataloader)}")
    print(f"test_data_loader:  {len(test_dataloader)}")
    return train_dataloader, eval_dataloader, test_dataloader


# Create ImgLanguageModel
def create_model(config: Config):
    assert config is not None

    model = ImgLanguageModel(config=config)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"pytorch_total_params: {pytorch_total_params/10**6} m")
    print(f"pytorch_total_trainable_params: {pytorch_total_trainable_params/10**6} m")
    count_parameters(model)

    return model


# Train and Eval model
def eval_model(
    model: ImgLanguageModel,
    eval_dataloader: DataLoader,
    device: torch.device,
    global_step: int,
    eval_steps: int,
    writer: SummaryWriter,
):
    model.eval()

    avg_eval_loss = None
    eval_loss_std = None
    with torch.no_grad():
        eval_losses = []
        for i, data in enumerate(eval_dataloader):
            if i > eval_steps:
                # It takes significant time to do one full eval.
                break

            (
                batch_img_tensor,
                batch_img_id_tensor,
                batch_target_tensor,
                batch_target_mask,
            ) = data
            batch_img_tensor = batch_img_tensor.to(device)
            batch_img_id_tensor = batch_img_id_tensor.to(device)
            batch_target_tensor = batch_target_tensor.to(device)
            batch_target_mask = batch_target_mask.to(device)
            (
                img_loss,
                text_loss,
                img_contrastive_prob,
                text_contrastive_prob,
                lm_loss,
                lm_logit,
            ) = model(batch_img_tensor, batch_target_tensor)
            writer.add_scalar("eval/Img Loss", img_loss, global_step)
            writer.add_scalar("eval/Text Loss", text_loss, global_step)
            writer.add_scalar("eval/LM Loss", lm_loss, global_step)
            eval_losses.append(img_loss + text_loss + lm_loss)
        eval_losses = torch.tensor(eval_losses)
        avg_eval_loss = eval_losses.mean()
        eval_loss_std = eval_losses.std()
        writer.add_scalar("eval/Loss", avg_eval_loss, global_step)
        writer.add_scalar("Loss/eval-std", eval_loss_std, global_step)
    model.train()
    writer.flush()
    return avg_eval_loss, eval_loss_std


def train_model(
    model: ImgLanguageModel,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    eval_interval: int,
    eval_steps: int,
    optimizer,
    scheduler,
    writer: SummaryWriter,
):
    best_vloss = torch.tensor(1_000_000)
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs"),
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
        for epoch in range(epochs):
            for train_step, data in enumerate(train_dataloader):
                global_step = epoch * len(train_dataloader) + train_step

                # Profile
                if global_step < 1 + 1 + 3:
                    prof.step()

                (
                    batch_img_tensor,
                    batch_img_id_tensor,
                    batch_target_tensor,
                    batch_target_mask,
                ) = data
                batch_img_tensor = batch_img_tensor.to(device)
                batch_img_id_tensor = batch_img_id_tensor.to(device)
                batch_target_tensor = batch_target_tensor.to(device)
                batch_target_mask = batch_target_mask.to(device)

                # Viz Model
                # if global_step == 0:
                #     writer.add_graph(model, (batch_img_tensor, batch_target_tensor))

                optimizer.zero_grad()
                (
                    img_loss,
                    text_loss,
                    img_contrastive_prob,
                    text_contrastive_prob,
                    lm_loss,
                    lm_logit,
                ) = model(batch_img_tensor, batch_target_tensor)
                writer.add_scalar("train/Img Loss", img_loss, global_step)
                writer.add_scalar("train/Text Loss", text_loss, global_step)
                writer.add_scalar("train/LM Loss", lm_loss, global_step)
                writer.add_scalar(
                    "train/Loss", img_loss + text_loss + lm_loss, global_step
                )
                writer.add_scalar(
                    "Learning Rate", scheduler.get_last_lr()[-1], global_step
                )
                loss = img_loss + text_loss + lm_loss
                loss.backward()
                # ===============================================================================================================
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_l2_grad_norm)
                # ===============================================================================================================
                # Error: command buffer exited with error status.
                # The Metal Performance Shaders operations encoded on it may not have completed.
                # Error:
                # (null)
                # Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)
                # <AGXG13XFamilyCommandBuffer: 0xa5e418420>
                # label = <none>
                # device = <AGXG13XDevice: 0x15430ee00>
                #     name = Apple M1 Max
                # commandQueue = <AGXG13XFamilyCommandQueue: 0x157a05800>
                #     label = <none>
                #     device = <AGXG13XDevice: 0x15430ee00>
                #         name = Apple M1 Max
                # retainedReferences = 1
                # ---------------------------------------------------------------------------------------------------------------
                optimizer.step()
                scheduler.step()

                if train_step > 0 and train_step % eval_interval == 0:
                    avg_vloss, _ = eval_model(
                        model=model,
                        eval_dataloader=eval_dataloader,
                        device=device,
                        global_step=global_step,
                        eval_steps=eval_steps,
                        writer=writer,
                    )

                    if avg_vloss is not None and avg_vloss < best_vloss:
                        best_vloss = avg_vloss
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_path = f"vlm_caption_model_{epoch}_{timestamp}"
                        torch.save(model.state_dict(), model_path)


def train():
    """ """
    EPOCHES = 3
    EVAL_INTERVAL = 100
    EVAL_STEPS = 10
    lr = 4e-4
    max_l2_grad_norm = 5

    device = torch.device("mps")  # macbook pro GPU

    # Create Datasets
    train_dataset, eval_dataset, test_dataset = create_datasets(config=config)

    # Create Dataloads
    train_dataloader, eval_dataloader, test_dataloader = create_dataloaders(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
    )

    # Create Model
    model = create_model(config=config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=EPOCHES
    )

    with SummaryWriter(flush_secs=1) as writer:
        train_model(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            device=device,
            epochs=EPOCHES,
            eval_interval=EVAL_INTERVAL,
            eval_steps=EVAL_STEPS,
            optimizer=optimizer,
            scheduler=scheduler,
            writer=writer,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"vlm_caption_model_{timestamp}_final"
        torch.save(model.state_dict(), model_path)
