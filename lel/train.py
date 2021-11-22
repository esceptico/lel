import math
from typing import Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from transformers import get_scheduler
from tqdm import tqdm

from lel.data.utils import move_to_device
from lel.metrics import ClassificationReport


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)

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
        try:
            for epoch in range(n_epochs):
                self.train(train_loader, optimizer, accumulation_steps, lr_scheduler, epoch)
                if val_loader is not None:
                    self.eval(val_loader)
        except KeyboardInterrupt:
            print('Keyboard interrupt...')

    def train(
        self,
        loader: DataLoader,
        optimizer: Optimizer,
        accumulation_steps: int,
        lr_scheduler,
        epoch: int
    ):
        self.model.train()
        loss = MeanMetric()
        last_batch_index = len(loader) - 1
        bar = tqdm(loader, desc=f'Epoch[{epoch}] Train')
        for num, batch in enumerate(bar):
            batch = move_to_device(batch, self.device)
            outputs = self.model(**batch)
            outputs.loss = outputs.loss / accumulation_steps
            outputs.loss.backward()
            loss.update(outputs.loss.cpu())
            bar.set_postfix({'loss': outputs.loss.detach().item()})
            if (num + 1) % accumulation_steps == 0 or num == last_batch_index:
                optimizer.step()
                lr_scheduler.step()
                self.model.zero_grad()
        epoch_loss = loss.compute().item()
        bar.set_postfix({'loss': epoch_loss})

    @torch.inference_mode()
    def eval(self, loader):
        self.model.eval()
        labels = loader.dataset.label_set.label_to_id
        classification_report = ClassificationReport(
            num_classes=len(labels),
            label_ids=labels,
            ignore=[loader.dataset.label_set.outside_id]
        ).to(self.device)
        loss = MeanMetric()
        bar = tqdm(loader, desc='Eval')
        for num, batch in enumerate(bar):
            batch = move_to_device(batch, self.device)
            outputs = self.model(**batch)
            preds = outputs.logits.softmax(-1)
            loss.update(outputs.loss.cpu())
            classification_report.update(preds.argmax(-1), batch['labels'])
            bar.set_postfix({'loss': outputs.loss.detach().item()})
        epoch_loss = loss.compute().item()
        bar.set_postfix({'loss': epoch_loss})
        precision, recall, f1, report = classification_report.compute()
        print(report)
