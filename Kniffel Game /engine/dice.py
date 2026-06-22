from dataclasses import dataclass, field
import random
from typing import Optional


@dataclass
class Die:
    value: int = 1
    held: bool = False
    bbox: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) from CV
    from_cv: bool = False


class DiceSet:
    def __init__(self, count: int = 5):
        self.dice: list[Die] = [Die() for _ in range(count)]

    def roll(self) -> None:
        for die in self.dice:
            if not die.held:
                die.value = random.randint(1, 6)
                die.bbox = None
                die.from_cv = False

    def values(self) -> list[int]:
        return [d.value for d in self.dice]

    def toggle_hold(self, index: int) -> None:
        if 0 <= index < len(self.dice):
            self.dice[index].held = not self.dice[index].held

    def release_all(self) -> None:
        for die in self.dice:
            die.held = False

    def __iter__(self):
        return iter(self.dice)

    def __len__(self) -> int:
        return len(self.dice)

    def __getitem__(self, index: int) -> Die:
        return self.dice[index]

    def update_from_cv(self, detections: list[dict]) -> None:
        """
        detections: list of {'value': int, 'bbox': (x, y, w, h), 'confidence': float}
        Sorted left-to-right by bbox x. Held dice are identified by their stored bbox
        center and filtered out before assigning the remaining detections to non-held slots.
        """
        if not detections:
            return
        sorted_det = sorted(detections, key=lambda d: d["bbox"][0] + d["bbox"][2] / 2)

        # For each held die, claim the single closest detection so it's not
        # reassigned to a free slot — but only when detections outnumber
        # free slots, i.e. the held die is visible in the frame.
        # If the camera returns ≤ n_non_held detections, skip claiming so
        # that every detected die still fills a slot.
        remaining = list(sorted_det)
        n_non_held = sum(1 for d in self.dice if not d.held)
        for die in self.dice:
            if die.held and die.bbox is not None and len(remaining) > n_non_held:
                x, y, w, h = die.bbox
                held_cx = x + w / 2
                best_i = min(range(len(remaining)),
                             key=lambda i: abs(remaining[i]["bbox"][0]
                                               + remaining[i]["bbox"][2] / 2
                                               - held_cx))
                remaining.pop(best_i)
        free_dets = remaining

        non_held_indices = [i for i, d in enumerate(self.dice) if not d.held]
        for die_idx, det in zip(non_held_indices, free_dets):
            self.dice[die_idx].value = int(det["value"])
            self.dice[die_idx].bbox = det["bbox"]
            self.dice[die_idx].from_cv = True
