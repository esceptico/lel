# LEL
Light Entity Library ¯\\\_(ツ)_/¯

## Usage

### Train

```python
import torch
from transformers import AutoTokenizer

from lel.data import LabelSet, NERDataset, Sampler
from lel.train import Trainer

tokenizer = AutoTokenizer.from_pretrained('...')
label_set = LabelSet(labels=['NAME', 'ORG'], markup_type='BIO', align_type='word')
vocab_size = tokenizer.vocab_size
num_labels = len(label_set.label_to_id)
model = Model(vocab_size, num_labels)

dataset = NERDataset(
    data=data_iterator(),
    tokenizer=tokenizer,
    label_set=label_set,
    sampler=Sampler(512, overlap=256),
    skip_broken_samples=True
)
loader = dataset.loader(batch_size=8)

trainer = Trainer(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
trainer.run(loader, optimizer=optimizer, val_loader=loader, n_epochs=1, accumulation_steps=2)
```


### Infer

```python
from lel.infer import Inference

infer = Inference(model, tokenizer, label_set)
predicted = infer('random text...')
```
