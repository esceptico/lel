import itertools
import json
import os
from typing import List, Sequence

from lel.data.label.alignment import get_aligner
from lel.data.label.markup import get_markup
from lel.data.label.dto import Label, Span


class LabelSet:
    """Provides entire pipeline of storing, encoding and decoding of entities."""
    outside_id = 0
    pass_id = -100

    def __init__(self, labels: Sequence[str], markup_type: str, align_type: str = 'word'):
        """Constructor.

        Args:
            labels: Sequence of labels (e.g. ['LOC', 'NAME', 'PER']).
            markup_type: Type of label markup. Possible values: 'IO', 'BIO', 'BILOU'.
            align_type: Type of token alignment. Possible values:
                - 'word' for aligning only beginning of each word in entity span.
                - 'token' for aligning all tokens of each span.
        """
        self.labels = labels
        self.markup_type = markup_type
        self.markup = get_markup(self.markup_type)
        self.label_to_id = {
            Label(self.markup.outside): self.outside_id
        }
        self.align_type = align_type
        self.aligner = get_aligner(align_type)
        prefix_set = sorted({
            self.markup.outside, self.markup.inside, self.markup.beginning,
            self.markup.last, self.markup.unit
        })
        prefix_set.remove(self.markup.outside)
        for i, (entity, prefix) in enumerate(itertools.product(self.labels, prefix_set), 1):
            label = Label(prefix, entity)
            self.label_to_id[label] = i
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def align(self, tokenized, labels) -> List[int]:
        aligned_labels = self.aligner(tokenized, labels, markup=self.markup)
        print(f'{aligned_labels=}')
        return [self.label_to_id.get(label, self.pass_id) for label in aligned_labels]

    def spans_from_ids(self, ids, offsets, ignore=None) -> List[Span]:
        labels = [self.id_to_label.get(idx, None) for idx in ids]
        print(f'{labels=}')
        return list(self.markup.decode(labels, offsets, ignore))

    def save(self, save_dir: str):
        path = os.path.join(save_dir, 'label_set.json')
        with open(path, 'w') as fp:
            config = {
                'labels': self.labels,
                'markup_type': self.markup_type,
                'align_type': self.align_type
            }
            json.dump(config, fp, ensure_ascii=False)

    @classmethod
    def load(cls, load_dir: str):
        path = os.path.join(load_dir, 'label_set.json')
        with open(path) as fp:
            config = json.load(fp)
        return cls(**config)
