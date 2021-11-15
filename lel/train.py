import math
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import AverageMeter
from transformers import get_scheduler
from tqdm import tqdm

from lel.metrics import ClassificationReport


class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model

    def run(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 10,
        accumulation_steps: int = 1,
        scheduler_type: str = 'linear',
        num_warmup_steps: float = 0.05
    ):
        if num_warmup_steps < 1:
            num_warmup_steps = math.ceil(num_warmup_steps * len(train_loader))
        num_update_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        max_train_steps = n_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        for epoch in range(n_epochs):
            self.train(train_loader, epoch, optimizer, accumulation_steps, lr_scheduler)
            if val_loader is not None:
                self.eval(val_loader, epoch)

    def setup_scheduler(self):
        pass

    def train(
        self,
        loader: DataLoader,
        epoch: int,
        optimizer: Optimizer,
        accumulation_steps: int,
        lr_scheduler
    ):
        self.model.train()
        loss = AverageMeter()
        last_batch_index = len(loader) - 1
        bar = tqdm(loader, desc=f'Epoch[{epoch}] Train')
        for num, batch in enumerate(bar):
            outputs = self.model(**batch)
            outputs.loss = outputs.loss / accumulation_steps
            outputs.loss.backward()
            loss.update(outputs.loss)
            bar.set_postfix({'loss': outputs.loss.detach().item()})
            if (num + 1) % accumulation_steps == 0 or num == last_batch_index:
                optimizer.step()
                lr_scheduler.step()
                self.model.zero_grad()
        epoch_loss = loss.compute().item()
        bar.set_postfix({'loss': epoch_loss})

    @torch.inference_mode()
    def eval(self, loader, epoch: int):
        self.model.eval()
        labels = loader.dataset.label_set.label_to_id
        classification_report = ClassificationReport(
            num_classes=len(labels),
            label_ids=labels,
            ignore=[loader.dataset.label_set.outside_id]
        )
        loss = AverageMeter()
        bar = tqdm(loader, desc=f'Epoch[{epoch}] Eval')
        for num, batch in enumerate(bar):
            outputs = self.model(**batch)
            preds = outputs.logits.softmax(-1)
            loss.update(outputs.loss)
            classification_report.update(preds.argmax(-1), batch['labels'])
            bar.set_postfix({'loss': outputs.loss.detach().item()})
        epoch_loss = loss.compute().item()
        bar.set_postfix({'loss': epoch_loss})
        precision, recall, f1, report = classification_report.compute()
        print(report)
