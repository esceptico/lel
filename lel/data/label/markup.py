from typing import Iterator, List, Optional, Type

from lel.data.label.dto import Span


class Markup:
    beginning = None
    inside = None
    last = None
    outside = None
    unit = None

    @classmethod
    def decode(cls, *args, **kwargs):
        raise NotImplementedError()


class IO(Markup):
    beginning: str = 'I'
    inside: str = 'I'
    last: str = 'I'
    outside: str = 'O'
    unit: str = 'I'

    @classmethod
    def decode(cls, tags, offsets, ignore: Optional[List] = None) -> Iterator[Span]:
        if ignore is None:
            ignore = []
        previous = None
        start = None
        end = None
        for i, (tag, (token_start, token_end)) in enumerate(zip(tags, offsets)):
            if (tag is None) or (i in ignore):
                continue
            if previous != tag.entity:
                if not previous and tag.prefix == cls.inside:
                    start = token_start
                elif previous and tag.prefix == cls.inside:
                    yield Span(start, end, previous)
                    start = token_start
                elif previous and tag.prefix == cls.outside:
                    yield Span(start, end, previous)
            previous = tag.entity
            end = token_end
        if previous:
            yield Span(start, end, previous)


class BIO(IO):
    beginning: str = 'B'
    unit: str = 'B'

    @classmethod
    def decode(cls, tags, offsets, ignore: Optional[List] = None):
        if ignore is None:
            ignore = []
        previous = None
        start = None
        end = None
        for i, (tag, (token_start, token_end)) in enumerate(zip(tags, offsets)):
            if (tag is None) or (i in ignore):
                continue
            if tag.prefix == cls.outside:
                if previous:
                    yield Span(start, end, previous)
                    previous = None
            elif tag.prefix == cls.beginning:
                if previous:
                    yield Span(start, end, previous)
                previous = tag.entity
                start = token_start
                end = token_end
            elif tag.prefix == cls.inside:
                if previous and previous != tag.entity:
                    yield Span(start, end, previous)
                    previous = None
                end = token_end
        if previous:
            yield Span(start, end, previous)


class BILOU(BIO):
    last: str = 'L'
    unit: str = 'U'

    @classmethod
    def decode(cls, tags, offsets, ignore: Optional[List] = None):
        if ignore is None:
            ignore = []
        previous = None
        start = None
        end = None
        for i, (tag, (token_start, token_end)) in enumerate(zip(tags, offsets)):
            if (tag is None) or (i in ignore):
                continue
            if tag.prefix == cls.outside:
                previous = None
            elif tag.prefix == cls.beginning:
                previous = tag.entity
                start = token_start
            elif tag.prefix == cls.inside:
                if previous and previous == tag.entity:
                    end = token_end
                else:
                    previous = None
            elif tag.prefix == cls.last:
                if previous and previous == tag.entity:
                    yield Span(start, token_end, previous)
            elif tag.prefix == cls.unit:
                previous = tag.entity
                start = token_start
                end = token_end
                yield Span(start, end, tag.entity)


def get_markup(name: str) -> Type[Markup]:
    markup_mapping = {
        'IO': IO,
        'BIO': BIO,
        'BILOU': BILOU
    }
    if name not in markup_mapping:
        raise ValueError(f'{name} markup is not supported.')
    return markup_mapping[name]
