from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Label:
    prefix: str
    entity: Optional[str] = None

    def __repr__(self):
        if self.entity is None:
            return self.prefix
        return f'{self.prefix}-{self.entity}'


@dataclass
class Span:
    start: int
    end: int
    name: str
