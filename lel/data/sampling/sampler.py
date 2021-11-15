from typing import Iterator, Sequence, TypeVar

from lel.data.sampling.sample import Sample


T = TypeVar('T')


def windowed(sequence: Sequence[T],
             n: int, step: int = 1) -> Iterator[Sequence[T]]:
    seq_len = len(sequence)
    for i in range(0, seq_len, step):
        window = sequence[i:i+n]
        yield window
        if i + n >= seq_len:
            return


class Sampler:
    def __init__(self, max_length: int = 512, overlap: int = 0):
        self.max_length = max_length
        self.overlap = overlap
        self.step = self.max_length - self.overlap

    def __call__(self, sample: Sample) -> Iterator[Sample]:
        packed = list(zip(sample.input_ids, sample.labels))
        samples = windowed(
            sequence=packed,
            n=self.max_length,
            step=self.step
        )
        for item in samples:
            tokenized, labels = zip(*item)
            yield Sample(tokenized, labels)
