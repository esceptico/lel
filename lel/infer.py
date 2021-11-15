from typing import List, Optional

import torch
from torch import nn
from transformers import PreTrainedTokenizerFast

from lel.data.label import LabelSet
from lel.utils import mask_words, word_offsets


class Inference:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        label_set: LabelSet
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.label_set = label_set

    def predict(self, tokenized, sample_len=None):
        seq_len = len(tokenized.input_ids)
        if not seq_len:
            return []
        if not sample_len:
            sample_len = seq_len
        preds = []
        for i in range(0, seq_len, sample_len):
            ids = tokenized.input_ids[i: i + sample_len]
            input_ids = torch.tensor(ids).unsqueeze(0)
            outputs = self.model(input_ids)
            sample_preds = outputs.logits.argmax(-1).squeeze(0).tolist()
            preds.extend(sample_preds)
        return preds

    def decode(self, preds, encoding, text):
        offsets = encoding.offsets
        ignore_ids = []
        if self.label_set.align_type == 'word':
            offsets = word_offsets(encoding)
            ignore_ids = mask_words(encoding.word_ids)

        result = []
        for span in self.label_set.spans_from_ids(preds, offsets, ignore_ids):
            if span.start == span.end:
                continue
            entity_text = text[span.start:span.end]
            entity = {
                'start': span.start,
                'end': span.end,
                'entity': span.name,
                'text': entity_text
            }
            result.append(entity)
        return result

    @torch.inference_mode()
    def __call__(self, text: str, sample_len: Optional[int] = None) -> List[dict]:
        tokenized = self.tokenizer(text, add_special_tokens=False)
        encoding = tokenized.encodings[0]
        preds = self.predict(tokenized, sample_len)
        decoded = self.decode(preds, encoding, text)
        return decoded
