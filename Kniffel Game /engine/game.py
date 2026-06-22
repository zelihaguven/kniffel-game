from enum import Enum, auto
from .dice import DiceSet
from .scoring import Category, Scorecard

MAX_ROLLS   = 3
NUM_PLAYERS = 2


class Phase(Enum):
    ROLL      = auto()   # waiting for first capture/roll of the turn
    HOLD      = auto()   # captured once/twice — can hold dice or score early
    SCORE     = auto()   # 3 captures used — must pick a category
    GAME_OVER = auto()


class GameState:
    def __init__(self, player_names: list[str] | None = None):
        self.player_names = player_names or ["Player 1", "Player 2"]
        self.dice          = DiceSet()
        self.scorecards    = [Scorecard() for _ in range(NUM_PLAYERS)]
        self.current_player = 0
        self.round          = 1
        self.rolls_this_turn = 0
        self.phase          = Phase.ROLL
        self.last_scored: tuple[int, Category, int] | None = None  # (player_idx, cat, pts)

    # ── convenience props ─────────────────────────────────────────────────────

    @property
    def scorecard(self) -> Scorecard:
        return self.scorecards[self.current_player]

    @property
    def player_name(self) -> str:
        return self.player_names[self.current_player]

    # ── actions ───────────────────────────────────────────────────────────────

    def capture(self, detections: list[dict]) -> bool:
        """Consume CV detections as a roll. Updates non-held dice only."""
        if self.phase not in (Phase.ROLL, Phase.HOLD):
            return False
        if not detections:
            return False
        self.dice.update_from_cv(detections)
        self._increment_roll()
        return True

    def roll_random(self) -> bool:
        """Fallback: roll non-held dice randomly (no camera)."""
        if self.phase not in (Phase.ROLL, Phase.HOLD):
            return False
        self.dice.roll()
        self._increment_roll()
        return True

    def toggle_hold(self, index: int) -> bool:
        if self.phase != Phase.HOLD:
            return False
        self.dice.toggle_hold(index)
        return True

    def score(self, category: Category) -> int:
        if not self.can_score():
            return -1
        if category not in self.scorecard.available():
            return -1
        pts = self.scorecard.assign(category, self.dice.values())
        self.last_scored = (self.current_player, category, pts)
        self._advance_turn()
        return pts

    # ── guards ────────────────────────────────────────────────────────────────

    def can_capture(self) -> bool:
        return self.phase in (Phase.ROLL, Phase.HOLD)

    def can_score(self) -> bool:
        return self.phase in (Phase.HOLD, Phase.SCORE) and self.rolls_this_turn > 0

    # ── internal ─────────────────────────────────────────────────────────────

    def _increment_roll(self) -> None:
        self.rolls_this_turn += 1
        self.phase = Phase.SCORE if self.rolls_this_turn >= MAX_ROLLS else Phase.HOLD

    def _advance_turn(self) -> None:
        self.dice.release_all()
        self.rolls_this_turn = 0
        next_p = (self.current_player + 1) % NUM_PLAYERS
        if next_p == 0:
            self.round += 1
        self.current_player = next_p
        if all(sc.is_complete() for sc in self.scorecards):
            self.phase = Phase.GAME_OVER
        else:
            self.phase = Phase.ROLL
