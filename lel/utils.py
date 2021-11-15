from typing import List, Tuple

from tokenizers import Encoding


def word_offsets(encoding: Encoding) -> List[Tuple[int, int]]:
    offsets = []
    for word_id in encoding.word_ids:
        offset = (
            (0, 0) if word_id is None
            else encoding.word_to_chars(word_id)
        )
        offsets.append(offset)
    return offsets


def mask_words(word_ids):
    token_ids = []
    seen = set()
    for i, word_id in enumerate(word_ids):
        if word_id in seen or word_id is None:
            token_ids.append(i)
        seen.add(word_id)
    return token_ids
