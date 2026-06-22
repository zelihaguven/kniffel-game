from enum import Enum, auto
from collections import Counter
from typing import Optional


class Category(Enum):
    ONES = auto()
    TWOS = auto()
    THREES = auto()
    FOURS = auto()
    FIVES = auto()
    SIXES = auto()
    THREE_OF_A_KIND = auto()
    FOUR_OF_A_KIND = auto()
    FULL_HOUSE = auto()
    SMALL_STRAIGHT = auto()
    LARGE_STRAIGHT = auto()
    KNIFFEL = auto()
    CHANCE = auto()


UPPER_CATEGORIES = {
    Category.ONES,
    Category.TWOS,
    Category.THREES,
    Category.FOURS,
    Category.FIVES,
    Category.SIXES,
}

UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS = 35

CATEGORY_LABELS = {
    Category.ONES: "Ones",
    Category.TWOS: "Twos",
    Category.THREES: "Threes",
    Category.FOURS: "Fours",
    Category.FIVES: "Fives",
    Category.SIXES: "Sixes",
    Category.THREE_OF_A_KIND: "Three of a Kind",
    Category.FOUR_OF_A_KIND: "Four of a Kind",
    Category.FULL_HOUSE: "Full House",
    Category.SMALL_STRAIGHT: "Sm. Straight",
    Category.LARGE_STRAIGHT: "Lg. Straight",
    Category.KNIFFEL: "Kniffel!",
    Category.CHANCE: "Chance",
}


def calculate(category: Category, dice: list[int]) -> int:
    counts = Counter(dice)
    total = sum(dice)

    match category:
        case Category.ONES:
            return counts[1] * 1
        case Category.TWOS:
            return counts[2] * 2
        case Category.THREES:
            return counts[3] * 3
        case Category.FOURS:
            return counts[4] * 4
        case Category.FIVES:
            return counts[5] * 5
        case Category.SIXES:
            return counts[6] * 6
        case Category.THREE_OF_A_KIND:
            return total if any(v >= 3 for v in counts.values()) else 0
        case Category.FOUR_OF_A_KIND:
            return total if any(v >= 4 for v in counts.values()) else 0
        case Category.FULL_HOUSE:
            vals = sorted(counts.values())
            return 25 if vals == [2, 3] else 0
        case Category.SMALL_STRAIGHT:
            unique = set(dice)
            straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
            return 30 if any(s.issubset(unique) for s in straights) else 0
        case Category.LARGE_STRAIGHT:
            return 40 if set(dice) in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0
        case Category.KNIFFEL:
            return 50 if any(v == 5 for v in counts.values()) else 0
        case Category.CHANCE:
            return total
        case _:
            return 0


class Scorecard:
    def __init__(self):
        self.scores: dict[Category, Optional[int]] = {c: None for c in Category}

    def assign(self, category: Category, dice: list[int]) -> int:
        if self.scores[category] is not None:
            raise ValueError(f"{category.name} is already filled")
        score = calculate(category, dice)
        self.scores[category] = score
        return score

    def potential_scores(self, dice: list[int]) -> dict[Category, int]:
        return {c: calculate(c, dice) for c, v in self.scores.items() if v is None}

    def upper_subtotal(self) -> int:
        return sum(v for c, v in self.scores.items() if c in UPPER_CATEGORIES and v is not None)

    def upper_bonus(self) -> int:
        return UPPER_BONUS if self.upper_subtotal() >= UPPER_BONUS_THRESHOLD else 0

    def total(self) -> int:
        return sum(v for v in self.scores.values() if v is not None) + self.upper_bonus()

    def is_complete(self) -> bool:
        return all(v is not None for v in self.scores.values())

    def available(self) -> list[Category]:
        return [c for c, v in self.scores.items() if v is None]
