from typing import List, Tuple

from tokenizers import Encoding

from lel.data.label.markup import Markup
from lel.data.label.dto import Label


def align_span_tokens_to_labels_token(
    tokenized: Encoding,
    spans: List[Tuple[int, int, str]],
    markup: Markup
):
    aligned = [
        None if idx is None else Label(markup.outside)
        for idx in tokenized.word_ids
    ]
    for entity_start, entity_end, entity_name in spans:
        token_ids = set()
        for char_idx in range(entity_start, entity_end):
            token_idx = tokenized.char_to_token(char_idx)
            if token_idx is not None:
                token_ids.add(token_idx)

        last_token_idx = len(token_ids) - 1
        for i, token_idx in enumerate(sorted(token_ids)):
            if len(token_ids) == 1:
                prefix = markup.unit
            elif i == 0:
                prefix = markup.beginning
            elif i == last_token_idx:
                prefix = markup.last
            else:
                prefix = markup.inside
            aligned[token_idx] = Label(prefix, entity_name)
    return aligned


def align_span_tokens_to_labels_word(
    tokenized: Encoding,
    spans: List[Tuple[int, int, str]],
    markup: Markup,
    label_all_tokens=False
):
    aligned = [
        None if idx is None else Label(markup.outside)
        for idx in tokenized.word_ids
    ]

    for entity_start, entity_end, entity_name in spans:
        word_ids = set()
        for char_idx in range(entity_start, entity_end):
            word_idx = tokenized.char_to_word(char_idx)
            if word_idx is not None:
                word_ids.add(word_idx)

        last_word_idx = len(word_ids) - 1
        for i, word_idx in enumerate(sorted(word_ids)):
            if len(word_ids) == 1:
                prefix = markup.unit
            elif i == 0:
                prefix = markup.beginning
            elif i == last_word_idx:
                prefix = markup.last
            else:
                prefix = markup.inside
            token_range = range(*tokenized.word_to_tokens(word_idx))
            for token_pos_in_word, token_idx in enumerate(token_range):
                label = Label(prefix, entity_name)
                if not label_all_tokens:
                    if token_pos_in_word != 0:
                        label = None
                aligned[token_idx] = label
    return aligned


def align_conll_tokens_to_labels(
    tokenized: Encoding,
    labels: List[str],
    label_all_tokens=False
):
    # TODO: add the ability to change the encoding markup
    word_ids = tokenized.word_ids()

    aligned = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(None)
        elif word_idx != previous_word_idx:
            aligned.append(labels[word_idx])
        else:
            aligned.append(labels[word_idx] if label_all_tokens else None)
        previous_word_idx = word_idx
    return aligned


def get_aligner(name: str):
    aligners_dict = {
        'token': align_span_tokens_to_labels_token,
        'word': align_span_tokens_to_labels_word,
        # 'conll': align_conll_tokens_to_labels
    }
    if name not in aligners_dict:
        raise ValueError(f'{name} aligner is not supported.')
    return aligners_dict[name]
