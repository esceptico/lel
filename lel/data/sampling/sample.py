from typing import Optional, Sequence

from dataclasses import dataclass


@dataclass
class Sample:
    input_ids: Sequence
    labels: Optional[Sequence] = None
    # label_mask: Optional[Sequence] = None
