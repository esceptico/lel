from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from lel.data.label import LabelSet
from lel.data.sampling.sample import Sample
from lel.data.sampling.sampler import Sampler


class NERDataset(Dataset):
    def __init__(
        self,
        data: Iterable,
        tokenizer: PreTrainedTokenizerFast,
        label_set: Optional[LabelSet] = None,
        sampler: Optional[Sampler] = None,
        skip_broken_samples: bool = False
    ):
        """Constructor.

        Args:
            data: Iterable of data samples.
            tokenizer: Instance of PreTrainedTokenizerFast class.
            label_set: LabelSet instance.
            sampler: Sampler instance. Defaults to None.
            skip_broken_samples: Indicates whether to skip samples with broken entities.
                Defaults to False.
        """
        self.tokenizer = tokenizer
        self.label_set = label_set
        self.sampler = sampler
        self.skip_broken_samples = skip_broken_samples
        self.samples = []

        for item in tqdm(data):
            text, labels = item['text'], item['labels']
            # TODO: Add special tokens after padding?
            encoding = tokenizer(text, add_special_tokens=False).encodings[0]
            aligned_labels = label_set.align(encoding, labels)
            for sample in self.iter_samples(encoding, aligned_labels):
                self.samples.append(sample)

    def iter_samples(self, encoding, labels):
        sample = Sample(encoding.ids, labels)
        sub_samples = [sample]
        if self.sampler is not None:
            sub_samples = self.sampler(sample)
        for sub_sample in sub_samples:
            if self.skip_broken_samples:
                is_broken = self.contains_entity_break(sub_sample.labels)
                if is_broken:
                    continue
            yield sub_sample

    def contains_entity_break(self, labels):
        """Checks the labels for entity break on start and end of sequence.

        Args:
            labels: Sequence of labels.

        Returns:
             If entities in start or end of sequence is broken,
             returns True, otherwise False.
        """
        labels = [label for label in labels if label != self.label_set.pass_id]
        to_label = self.label_set.id_to_label
        if not labels:
            return True
        markup = self.label_set.markup
        allowed_at_start = {markup.outside, markup.unit, markup.beginning}
        allowed_at_end = {markup.outside, markup.unit, markup.last}
        first_is_broken = to_label[labels[0]].prefix not in allowed_at_start
        last_is_broken = to_label[labels[-1]].prefix not in allowed_at_end
        return first_is_broken or last_is_broken

    def collate_fn(self, batch: Sequence[Sample]) -> Dict[str, torch.Tensor]:
        """Collate function for data loader. Pads samples to the longest sample
        in the batch and pack them to the dictionary.

        Args:
            batch: List of samples.

        Returns:
            Batch. Consists of input ids, label ids and attention mask.
        """
        def pad(sequence: Sequence, value: Any, max_length: int):
            sequence = sequence[:max_length]
            padding = [value] * (max_length - len(sequence))
            return [*sequence, *padding]

        input_ids = []
        labels = []
        lens = [len(item.input_ids) for item in batch]
        max_len = max(lens)
        for item in batch:
            input_ids.append(pad(item.input_ids, self.tokenizer.pad_token_id, max_len))
            labels.append(pad(item.labels, self.label_set.pass_id, max_len))
        mask = torch.arange(max_len)[None, :] < torch.tensor(lens)[:, None]
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'attention_mask': mask
        }

    def loader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self, collate_fn=self.collate_fn, *args, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
