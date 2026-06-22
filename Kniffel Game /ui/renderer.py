import pygame
import cv2
import math
import numpy as np
from typing import Optional

from engine.game import GameState, Phase
from engine.scoring import Category, UPPER_BONUS_THRESHOLD, UPPER_BONUS
from cv_interface.detector import DetectorMode
from .constants import *

# ── Layout ────────────────────────────────────────────────────────────────────

_CC_X = PAD                         # 14
_CC_Y = HEADER_H + PAD              # 68
_CC_W = CAM_PANEL_W - PAD * 2       # 728
_CC_H = 346

_CV_X = _CC_X + 10
_CV_Y = _CC_Y + 38                  # below card title
_CV_W = _CC_W - 20                  # 708
_CV_H = _CC_H - 48                  # 298

_DT_X = PAD
_DT_Y = _CC_Y + _CC_H + 10         # 424
_DT_W = _CC_W                       # 728
_DT_H = 158

_D_SZ     = 74
_D_GAP    = 20
_D_ROW_W  = 5 * _D_SZ + 4 * _D_GAP   # 450
_D_X0     = _DT_X + (_DT_W - _D_ROW_W) // 2
_D_Y0     = _DT_Y + 26              # die top inside tray
_D_HOLD_Y = _D_Y0 + _D_SZ + 7

_BTN_Y  = _DT_Y + _DT_H + 10       # 592
_BTN_H  = 44
_BTN_W  = (_CC_W - 20) // 3        # 236

_SP_PAD   = 14
_SP_X0    = SCORE_PANEL_X + _SP_PAD   # 770
_SP_LBL_X = _SP_X0 + MINI_AREA_W + 8  # 841
_SP_P1_X  = _SP_LBL_X + 130 + 6       # 977
_SP_PCW   = 130
_SP_P2_X  = _SP_P1_X + _SP_PCW        # 1107

# Category data
UPPER_ORDER = [
    Category.ONES, Category.TWOS, Category.THREES,
    Category.FOURS, Category.FIVES, Category.SIXES,
]
LOWER_ORDER = [
    Category.THREE_OF_A_KIND, Category.FOUR_OF_A_KIND,
    Category.FULL_HOUSE, Category.SMALL_STRAIGHT,
    Category.LARGE_STRAIGHT, Category.KNIFFEL, Category.CHANCE,
]
SHORT_LABELS = {
    Category.ONES:            "Ones",
    Category.TWOS:            "Twos",
    Category.THREES:          "Threes",
    Category.FOURS:           "Fours",
    Category.FIVES:           "Fives",
    Category.SIXES:           "Sixes",
    Category.THREE_OF_A_KIND: "3 of a Kind",
    Category.FOUR_OF_A_KIND:  "4 of a Kind",
    Category.FULL_HOUSE:      "Full House",
    Category.SMALL_STRAIGHT:  "Sm. Straight",
    Category.LARGE_STRAIGHT:  "Lg. Straight",
    Category.KNIFFEL:         "Kniffel!",
    Category.CHANCE:          "Chance",
}
CATEGORY_PREVIEW: dict[Category, list[tuple[int, bool]]] = {
    Category.ONES:            [(1, False)],
    Category.TWOS:            [(2, False)],
    Category.THREES:          [(3, False)],
    Category.FOURS:           [(4, False)],
    Category.FIVES:           [(5, False)],
    Category.SIXES:           [(6, False)],
    Category.THREE_OF_A_KIND: [(5,False),(5,False),(5,False),(2,True),(4,True)],
    Category.FOUR_OF_A_KIND:  [(4,False),(4,False),(4,False),(4,False),(1,True)],
    Category.FULL_HOUSE:      [(3,False),(3,False),(3,False),(2,False),(2,False)],
    Category.SMALL_STRAIGHT:  [(1,False),(2,False),(3,False),(4,False)],
    Category.LARGE_STRAIGHT:  [(1,False),(2,False),(3,False),(4,False),(5,False)],
    Category.KNIFFEL:         [(6,False),(6,False),(6,False),(6,False),(6,False)],
    Category.CHANCE:          [(1,True),(3,True),(5,True),(2,True),(4,True)],
}


def _cv_to_surf(frame: np.ndarray, tw: int, th: int) -> pygame.Surface:
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        pygame.font.init()

        # Serif  — Playfair Display feel (headings, brand, scorecard)
        self.f_brand   = pygame.font.SysFont("Georgia", 18, bold=True)
        self.f_serif_m = pygame.font.SysFont("Georgia", 13, bold=True)
        self.f_serif   = pygame.font.SysFont("Georgia", 12)
        self.f_total   = pygame.font.SysFont("Georgia", 18, bold=True)

        # Monospace — HUD/camera overlays
        self.f_mono    = pygame.font.SysFont("Courier New", 10)
        self.f_mono_bd = pygame.font.SysFont("Courier New", 10, bold=True)

        # Sans — UI controls, buttons, scores
        self.f_sans    = pygame.font.SysFont("Arial", 13)
        self.f_sans_sm = pygame.font.SysFont("Arial", 11)
        self.f_sans_xs = pygame.font.SysFont("Arial", 10, bold=True)
        self.f_sans_bd = pygame.font.SysFont("Arial", 13, bold=True)
        self.f_sans_sc = pygame.font.SysFont("Arial", 12, bold=True)
        self.f_input   = pygame.font.SysFont("Arial", 22)

        # Precompute scanlines overlay for camera viewport
        self._scanlines = pygame.Surface((_CV_W, _CV_H), pygame.SRCALPHA)
        for sy in range(0, _CV_H, 4):
            pygame.draw.line(self._scanlines, (0, 0, 0, 55), (0, sy), (_CV_W, sy))

        # Precompute vignette overlay for the felt background
        self._vignette = pygame.Surface((CAM_PANEL_W, HEIGHT), pygame.SRCALPHA)
        cx0, cy0 = CAM_PANEL_W // 2, HEIGHT // 2
        max_r = int(math.hypot(cx0, cy0))
        for step in range(6):
            r = max_r - step * max_r // 7
            alpha = 8 + step * 10
            if r > 0:
                s = pygame.Surface((CAM_PANEL_W, HEIGHT), pygame.SRCALPHA)
                pygame.draw.ellipse(s, (0, 0, 0, 0),
                                    pygame.Rect(cx0 - r, cy0 - r, r*2, r*2))
                s.fill((0, 0, 0, alpha), special_flags=pygame.BLEND_RGBA_SUB)
                self._vignette.blit(s, (0, 0))

        self.die_rects:        list[pygame.Rect] = [pygame.Rect(0,0,0,0)] * 5
        self.hold_btn_rects:   list[pygame.Rect] = [pygame.Rect(-999,-999,0,0)] * 5
        self.capture_btn_rect  = pygame.Rect(0, 0, 0, 0)
        self.hold_all_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_btn_rect    = pygame.Rect(0, 0, 0, 0)
        self.release_all_rect  = pygame.Rect(-999, -999, 0, 0)
        self._score_rects: dict[Category, pygame.Rect] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def draw(self, game, cam_frame, detections, hovered_category,
             view_player, detector_mode=DetectorMode.CONTOURS, cam_source=0):
        self._base()
        self._header(game, detector_mode, cam_source)
        self._camera_card(cam_frame, detections, detector_mode, cam_source)
        self._dice_tray(game)
        self._action_buttons(game, detections)
        self._score_panel(game, hovered_category)
        if game.phase == Phase.GAME_OVER:
            self._game_over(game)

    def get_hovered_category(self, pos) -> Optional[Category]:
        for cat, rect in self._score_rects.items():
            if rect.collidepoint(pos):
                return cat
        return None

    # ── Base ──────────────────────────────────────────────────────────────────

    def _base(self):
        # Left: casino felt green
        self.screen.fill(BG, (0, 0, CAM_PANEL_W, HEIGHT))
        # Right: warm paper
        self.screen.fill(PAPER, (SCORE_PANEL_X, 0, SCORE_PANEL_W, HEIGHT))
        # Paper left shadow line
        pygame.draw.line(self.screen, PAPER_BORDER,
                         (SCORE_PANEL_X, HEADER_H), (SCORE_PANEL_X, HEIGHT), 1)
        pygame.draw.line(self.screen, (190, 178, 150),
                         (SCORE_PANEL_X + 1, HEADER_H), (SCORE_PANEL_X + 1, HEIGHT), 1)

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self, game, detector_mode, cam_source):
        pygame.draw.rect(self.screen, HDR_BG, (0, 0, WIDTH, HEADER_H))
        pygame.draw.line(self.screen, HDR_BORDER, (0, HEADER_H-1), (WIDTH, HEADER_H-1))
        cy = HEADER_H // 2

        # Brand — serif gold
        dot_r = 5
        t = pygame.time.get_ticks()
        pulse = 0.5 + 0.5 * math.sin(t / 600)
        dot_col = tuple(int(a + (b-a)*pulse) for a, b in zip(TECH_DIM, TECH_GREEN))
        pygame.draw.circle(self.screen, dot_col, (22, cy), dot_r)
        logo = self.f_brand.render("Kniffel Vision AI", True, HDR_GOLD)
        self.screen.blit(logo, (34, cy - logo.get_height() // 2))
        ai_lbl = self.f_sans_xs.render("AI", True, TECH_DIM)
        self.screen.blit(ai_lbl, (34 + logo.get_width() + 6, cy - ai_lbl.get_height()//2 + 1))

        # Center chips
        phase_labels = {Phase.ROLL: "ROLL", Phase.HOLD: "HOLD", Phase.SCORE: "SCORE"}
        phase_txt = phase_labels.get(game.phase, "END")
        phase_col = TECH_GREEN if game.phase == Phase.ROLL else (
                    GOLD       if game.phase == Phase.HOLD else
                    (205,100,40))
        chips = [
            (f"Round {game.round} / 13", HDR_MUTED),
            (f"Roll {game.rolls_this_turn} / 3", HDR_MUTED),
            (phase_txt, phase_col),
        ]
        chip_h = 22
        total_w = sum(self.f_mono.size(t)[0] + 20 for t, _ in chips) + 8*(len(chips)-1)
        x = (CAM_PANEL_W - total_w) // 2
        for text, col in chips:
            sw = self.f_mono.size(text)[0]
            cw = sw + 20
            pygame.draw.rect(self.screen, (18, 40, 22),
                             (x, cy-chip_h//2, cw, chip_h), border_radius=11)
            pygame.draw.rect(self.screen, col,
                             (x, cy-chip_h//2, cw, chip_h), width=1, border_radius=11)
            s = self.f_mono.render(text, True, col)
            self.screen.blit(s, (x+10, cy-s.get_height()//2))
            x += cw + 8

        # Player tabs
        tab_w, tab_h = 150, 34
        gap = 8
        tx0 = SCORE_PANEL_X + (SCORE_PANEL_W - 2*tab_w - gap) // 2
        for p in range(2):
            active = (p == game.current_player and game.phase != Phase.GAME_OVER)
            col_on = (TECH_GREEN if p == 0 else (80, 160, 240))
            tx, ty = tx0 + p*(tab_w+gap), cy - tab_h//2
            if active:
                pygame.draw.rect(self.screen, (16, 38, 20),
                                 (tx, ty, tab_w, tab_h), border_radius=10)
                pygame.draw.rect(self.screen, col_on,
                                 (tx, ty, tab_w, tab_h), width=2, border_radius=10)
                nc, sc2 = HDR_TEXT, col_on
            else:
                pygame.draw.rect(self.screen, (12, 28, 15),
                                 (tx, ty, tab_w, tab_h), border_radius=10)
                pygame.draw.rect(self.screen, HDR_BORDER,
                                 (tx, ty, tab_w, tab_h), width=1, border_radius=10)
                nc, sc2 = HDR_MUTED, (60, 80, 62)
            ns = self.f_sans_sc.render(game.player_names[p], True, nc)
            ps = self.f_sans_xs.render(str(game.scorecards[p].total()) + " pts", True, sc2)
            self.screen.blit(ns, (tx+12, ty+tab_h//2-ns.get_height()//2-1))
            self.screen.blit(ps, (tx+tab_w-ps.get_width()-10, ty+tab_h-ps.get_height()-5))

    # ── Camera card ───────────────────────────────────────────────────────────

    def _camera_card(self, cam_frame, detections, detector_mode, cam_source):
        count = len(detections)
        hud_col = CAM_HUD_HI if count > 0 else CAM_HUD_LO

        card = pygame.Rect(_CC_X, _CC_Y, _CC_W, _CC_H)
        # Card shadow
        pygame.draw.rect(self.screen, (3, 8, 4),
                         card.move(2, 3), border_radius=12)
        pygame.draw.rect(self.screen, CAM_CARD, card, border_radius=12)
        pygame.draw.rect(self.screen, hud_col, card, width=1, border_radius=12)

        # Card title (mono / tech feel)
        tl = self.f_mono_bd.render("OpenCV Live Stream", True, hud_col)
        self.screen.blit(tl, (_CC_X + 12, _CC_Y + 10))

        # Pulsing LIVE dot
        t = pygame.time.get_ticks()
        pulse = 0.5 + 0.5 * math.sin(t / 280)
        if cam_frame is not None:
            dot_r  = int(3 + 2 * pulse)
            dot_col = tuple(int(a + (b-a)*pulse) for a, b in
                            zip(TECH_DIM, TECH_GREEN))
            live_txt = "LIVE"
        else:
            dot_r   = 3
            dot_col = (160, 40, 40)
            live_txt = "NO SIGNAL"
        live_s = self.f_mono_bd.render(live_txt, True, dot_col)
        lx = _CC_X + _CC_W - live_s.get_width() - 18
        self.screen.blit(live_s, (lx, _CC_Y + 11))
        pygame.draw.circle(self.screen, dot_col, (lx - 7, _CC_Y + 19), dot_r)

        pygame.draw.line(self.screen, hud_col,
                         (_CC_X+1, _CC_Y+32), (_CC_X+_CC_W-1, _CC_Y+32))

        # Viewport
        vp = pygame.Rect(_CV_X, _CV_Y, _CV_W, _CV_H)
        pygame.draw.rect(self.screen, CAM_VIEWPORT, vp)

        if cam_frame is not None:
            try:
                surf = _cv_to_surf(cam_frame, _CV_W, _CV_H)
                bx = vp.x + (vp.width  - surf.get_width())  // 2
                by = vp.y + (vp.height - surf.get_height()) // 2
                clip = self.screen.get_clip()
                self.screen.set_clip(vp)
                self.screen.blit(surf, (bx, by))
                # Scanlines over the feed
                self.screen.blit(self._scanlines, vp.topleft)
                self.screen.set_clip(clip)
            except Exception:
                pass
        else:
            msg = self.f_mono.render("CAMERA SIGNAL NOT FOUND", True, TECH_DIM)
            self.screen.blit(msg, msg.get_rect(center=vp.center))

        pygame.draw.rect(self.screen, hud_col, vp, width=1)

        # HUD corner brackets
        bs, bt = 16, 2
        for cx2, cy2, lf, top in [
            (vp.x, vp.y, True, True), (vp.right, vp.y, False, True),
            (vp.x, vp.bottom, True, False), (vp.right, vp.bottom, False, False),
        ]:
            dx, dy = (1 if lf else -1), (1 if top else -1)
            pygame.draw.line(self.screen, hud_col, (cx2, cy2), (cx2+dx*bs, cy2), bt)
            pygame.draw.line(self.screen, hud_col, (cx2, cy2), (cx2, cy2+dy*bs), bt)

        # HUD badges
        mode_name = "BinContours" if detector_mode == DetectorMode.CONTOURS else "Watershed"
        self._hud_badge(vp.x+8, vp.bottom-8,
                        f"OpenCV Active  |  {mode_name}", "bottom-left")
        det_col = TECH_GREEN if count==5 else (GOLD if count>0 else TECH_DIM)
        self._hud_badge(vp.right-8, vp.bottom-8,
                        f"Detected: {count}/5", "bottom-right", det_col)
        self._hud_badge(vp.right-8, vp.y+8,
                        f"CAM {cam_source}", "top-right")

    # ── Dice tray ─────────────────────────────────────────────────────────────

    def _dice_tray(self, game):
        outer = pygame.Rect(_DT_X, _DT_Y, _DT_W, _DT_H)

        # Shadow
        pygame.draw.rect(self.screen, (4, 10, 6),
                         outer.move(2, 4), border_radius=12)

        # Wooden outer edge (darkest)
        pygame.draw.rect(self.screen, TRAY_WOOD_DK, outer, border_radius=12)

        # Wooden face (inset 3px)
        wood_face = outer.inflate(-6, -6)
        pygame.draw.rect(self.screen, TRAY_WOOD, wood_face, border_radius=10)

        # Wood top highlight
        pygame.draw.line(self.screen, TRAY_WOOD_LT,
                         (wood_face.x+10, wood_face.y+2),
                         (wood_face.right-10, wood_face.y+2))

        # Felt interior (inset further)
        felt = wood_face.inflate(-8, -8)
        pygame.draw.rect(self.screen, TRAY_FELT, felt, border_radius=8)

        # Felt inner shadow top line
        pygame.draw.line(self.screen, (8, 22, 12),
                         (felt.x+4, felt.y+2), (felt.right-4, felt.y+2))

        # Card title (engraved look)
        tl = self.f_mono_bd.render("DICE TRAY", True, TRAY_WOOD_LT)
        self.screen.blit(tl, (_DT_X + 16, _DT_Y + 8))

        held_n = sum(1 for d in game.dice if d.held)
        if held_n:
            hs = self.f_sans_xs.render(f"{held_n} LOCKED", True, GOLD)
            self.screen.blit(hs, (_DT_X+_DT_W-hs.get_width()-16, _DT_Y+9))

        rolled   = game.rolls_this_turn > 0
        can_hold = game.phase == Phase.HOLD
        hold_h   = 22

        self.die_rects      = []
        self.hold_btn_rects = []
        self.release_all_rect = pygame.Rect(-999, -999, 0, 0)

        for i, die in enumerate(game.dice):
            dx    = _D_X0 + i * (_D_SZ + _D_GAP)
            drect = self._die(dx, _D_Y0, die, rolled)
            self.die_rects.append(drect)

            if can_hold:
                brect = pygame.Rect(dx, _D_HOLD_Y, _D_SZ, hold_h)
                self.hold_btn_rects.append(brect)
                if die.held:
                    pygame.draw.rect(self.screen, GOLD_BG,
                                     brect, border_radius=11)
                    pygame.draw.rect(self.screen, GOLD,
                                     brect, width=1, border_radius=11)
                    ls = self.f_sans_xs.render("UNLOCK", True, GOLD)
                else:
                    pygame.draw.rect(self.screen, (14, 36, 20),
                                     brect, border_radius=11)
                    pygame.draw.rect(self.screen, TECH_DIM,
                                     brect, width=1, border_radius=11)
                    ls = self.f_sans_xs.render("LOCK", True, TECH_GREEN)
                self.screen.blit(ls, ls.get_rect(center=brect.center))
            else:
                self.hold_btn_rects.append(pygame.Rect(-999, -999, 0, 0))

    def _die(self, x: int, y: int, die, rolled: bool) -> pygame.Rect:
        sz, r = _D_SZ, 9

        if die.held:
            face_c   = DIE_HELD_FACE
            border_c = DIE_HELD_BORDER
            dot_c    = DIE_HELD_DOT
            bw       = 2
        elif die.from_cv:
            face_c   = DIE_FACE
            border_c = DIE_CV_BORDER
            dot_c    = DIE_DOT
            bw       = 2
        else:
            face_c   = DIE_FACE
            border_c = DIE_BORDER
            dot_c    = DIE_DOT
            bw       = 2

        # Multi-layer shadow for weight
        for offs, alpha_shade in [(4, 2), (3, 5), (2, 8)]:
            shd = pygame.Rect(x+offs, y+offs, sz, sz)
            pygame.draw.rect(self.screen, DIE_SHADOW, shd, border_radius=r)

        face = pygame.Rect(x, y, sz, sz)
        pygame.draw.rect(self.screen, face_c, face, border_radius=r)

        # Wooden border — slightly offset for depth
        pygame.draw.rect(self.screen, TRAY_WOOD_DK,
                         face.inflate(4, 4).move(-2, -2), border_radius=r+2)
        pygame.draw.rect(self.screen, border_c, face.inflate(2, 2).move(-1,-1),
                         border_radius=r+1, width=1)
        pygame.draw.rect(self.screen, face_c, face, border_radius=r)

        # Top highlight — soft sheen
        pygame.draw.line(self.screen, (255, 255, 255),
                         (x+r, y+3), (x+sz-r, y+3))
        pygame.draw.line(self.screen, (240, 236, 226),
                         (x+r, y+5), (x+sz//2, y+5))

        # Pips
        if rolled:
            pr = 5
            for fx, fy in DOT_MAP.get(die.value, []):
                px, py = int(x+fx*sz), int(y+fy*sz)
                pygame.draw.circle(self.screen, dot_c, (px, py), pr)
                pygame.draw.circle(self.screen, (255,255,255), (px-1,py-1), pr//3)

        # Held gold badge
        if die.held:
            bw2, bh2 = 44, 13
            bx2 = x + (sz - bw2) // 2
            by2 = y + sz - bh2 - 5
            pygame.draw.rect(self.screen, GOLD, (bx2, by2, bw2, bh2), border_radius=6)
            hl = self.f_sans_xs.render("LOCKED", True, GOLD_TEXT)
            self.screen.blit(hl, hl.get_rect(center=(bx2+bw2//2, by2+bh2//2)))

        return face

    # ── Action buttons ────────────────────────────────────────────────────────

    def _action_buttons(self, game, detections):
        count   = len(detections)
        roll_en = game.can_capture()

        roll_label = "ROLL DICE"
        if 0 < count < 5: roll_label = f"ROLL DICE  ({count}/5)"
        elif count == 5:   roll_label = "ROLL DICE  (5/5)"

        # Two buttons: wide ROLL DICE + narrow NEW GAME
        roll_w  = _CC_W - 160 - 10   # ~558
        reset_w = 150
        bx0     = _CC_X

        self.capture_btn_rect  = pygame.Rect(bx0,               _BTN_Y, roll_w,  _BTN_H)
        self.hold_all_btn_rect = pygame.Rect(-999, -999, 0, 0)  # removed
        self.reset_btn_rect    = pygame.Rect(bx0 + roll_w + 10, _BTN_Y, reset_w, _BTN_H)

        self._button(self.capture_btn_rect, roll_label, roll_en, "primary")
        self._button(self.reset_btn_rect,   "NEW GAME", True,    "ghost")

        hints = {
            Phase.ROLL:  "Point camera at dice, then press ROLL DICE  /  SPACE for random roll",
            Phase.HOLD:  "Click dice below to lock them, then press ROLL DICE to re-roll the rest",
            Phase.SCORE: "All rolls used  —  click a category on the scorecard to record your score",
        }
        ht = hints.get(game.phase, "")
        hs = self.f_sans_sm.render(ht, True, ON_DARK_DIM)
        self.screen.blit(hs, hs.get_rect(
            center=(_CC_X + _CC_W//2, _BTN_Y + _BTN_H + 10)))

    def _button(self, rect, label, enabled, style="primary"):
        hov = rect.collidepoint(pygame.mouse.get_pos()) and enabled
        r   = rect.height // 2

        if not enabled:
            pygame.draw.rect(self.screen, (18, 44, 28), rect, border_radius=r)
            pygame.draw.rect(self.screen, (30, 60, 35), rect, width=1, border_radius=r)
            s = self.f_sans_bd.render(label, True, (50, 80, 55))
        elif style == "primary":
            # Gold glow on hover
            if hov:
                glow = rect.inflate(10, 10)
                gs   = pygame.Surface((glow.width, glow.height), pygame.SRCALPHA)
                pygame.draw.rect(gs, (*GOLD, 80), gs.get_rect(), border_radius=r+5)
                self.screen.blit(gs, glow)
            bg = GOLD_HV if hov else GOLD
            pygame.draw.rect(self.screen, bg, rect, border_radius=r)
            # Sheen on top half
            sheen = pygame.Surface((rect.width, rect.height//2), pygame.SRCALPHA)
            sheen.fill((255, 255, 255, 30))
            self.screen.blit(sheen, rect.topleft)
            s = self.f_sans_bd.render(label, True, GOLD_TEXT)
        elif style == "secondary":
            bg = (30, 70, 38) if hov else (22, 55, 28)
            pygame.draw.rect(self.screen, bg, rect, border_radius=r)
            pygame.draw.rect(self.screen, TECH_DIM, rect, width=1, border_radius=r)
            s = self.f_sans_bd.render(label, True, TECH_GREEN if hov else ON_DARK_DIM)
        else:  # ghost
            bg = (25, 58, 30) if hov else (18, 44, 22)
            pygame.draw.rect(self.screen, bg, rect, border_radius=r)
            pygame.draw.rect(self.screen, (35, 72, 40), rect, width=1, border_radius=r)
            s = self.f_sans_bd.render(label, True, ON_DARK_DIM)

        self.screen.blit(s, s.get_rect(center=rect.center))

    # ── Score panel ───────────────────────────────────────────────────────────

    def _score_panel(self, game, hovered):
        self._score_rects = {}
        x0 = _SP_X0
        y  = HEADER_H + 14

        title = self.f_serif_m.render("SCORECARD", True, TEXT_SEC)
        self.screen.blit(title, (x0, y))
        y += title.get_height() + 10

        # Player column headers
        for p in range(2):
            cx    = (_SP_P1_X if p==0 else _SP_P2_X) + _SP_PCW//2
            active= p == game.current_player
            col   = (P1_COL if p==0 else P2_COL) if active else TEXT_DIM
            ns    = self.f_serif_m.render(game.player_names[p], True, col)
            self.screen.blit(ns, ns.get_rect(center=(cx, y+ns.get_height()//2)))
            if active:
                pygame.draw.line(self.screen, col,
                                 (cx-ns.get_width()//2-2, y+ns.get_height()+4),
                                 (cx+ns.get_width()//2+2, y+ns.get_height()+4), 2)
        y += 26
        pygame.draw.line(self.screen, PAPER_BORDER,
                         (SCORE_PANEL_X+1, y), (WIDTH-1, y))
        y += 6

        potential = (
            game.scorecards[game.current_player].potential_scores(game.dice.values())
            if game.can_score() else {}
        )
        row_idx = 0

        sec = self.f_sans_xs.render("UPPER SECTION", True, TEXT_DIM)
        self.screen.blit(sec, (x0, y))
        y += sec.get_height() + 4

        for cat in UPPER_ORDER:
            y, row_idx = self._score_row(game, cat, potential, hovered, y, row_idx)

        y += 3
        pygame.draw.line(self.screen, PAPER_BORDER, (SCORE_PANEL_X+1,y),(WIDTH-1,y))
        y += 6

        sub = game.scorecards[game.current_player].upper_subtotal()
        met = sub >= UPPER_BONUS_THRESHOLD
        bcol = P1_COL if met else AMBER_COL
        bl = self.f_sans_sm.render(
            f"Bonus +35:  {sub}/{UPPER_BONUS_THRESHOLD}" + ("  ACHIEVED!" if met else ""),
            True, bcol)
        self.screen.blit(bl, (x0, y))
        y += bl.get_height() + 5

        bar_w = WIDTH - x0 - 14
        pygame.draw.rect(self.screen, PAPER_LINE, (x0, y, bar_w, 5), border_radius=3)
        fill = int(bar_w * min(sub/UPPER_BONUS_THRESHOLD, 1.0))
        if fill > 0:
            pygame.draw.rect(self.screen, bcol, (x0, y, fill, 5), border_radius=3)
        y += 12

        pygame.draw.line(self.screen, PAPER_BORDER, (SCORE_PANEL_X+1,y),(WIDTH-1,y))
        y += 6

        sec2 = self.f_sans_xs.render("LOWER SECTION", True, TEXT_DIM)
        self.screen.blit(sec2, (x0, y))
        y += sec2.get_height() + 4

        for cat in LOWER_ORDER:
            y, row_idx = self._score_row(game, cat, potential, hovered, y, row_idx)

        y += 3
        pygame.draw.line(self.screen, PAPER_BORDER, (SCORE_PANEL_X+1,y),(WIDTH-1,y))
        y += 8

        tl = self.f_total.render("Total", True, TEXT)
        self.screen.blit(tl, (x0, y))
        for p in range(2):
            col = (P1_COL if p==0 else P2_COL) if p==game.current_player else TEXT_SEC
            cx  = (_SP_P1_X if p==0 else _SP_P2_X) + _SP_PCW//2
            ts  = self.f_total.render(str(game.scorecards[p].total()), True, col)
            self.screen.blit(ts, ts.get_rect(center=(cx, y+ts.get_height()//2)))
        y += self.f_total.get_height() + 10

        if game.last_scored:
            pidx, cat, pts = game.last_scored
            col = P1_COL if pts > 0 else TEXT_DIM
            msg = self.f_sans_sm.render(
                f"{game.player_names[pidx]} scored {pts} pts on {SHORT_LABELS[cat]}",
                True, col)
            self.screen.blit(msg, (x0, y))

    def _score_row(self, game, cat, potential, hovered, y, row_idx):
        rect   = pygame.Rect(SCORE_PANEL_X+1, y, SCORE_PANEL_W-1, SCORE_ROW_H)
        filled = game.scorecards[game.current_player].scores[cat] is not None
        is_hov = cat==hovered and game.can_score() and not filled
        self._score_rects[cat] = rect

        ROW_H  = 25
        bg = PAPER_ALT if row_idx%2 else PAPER
        pygame.draw.rect(self.screen, bg, rect)
        if is_hov:
            hover_bg = (232, 245, 232)  # very subtle green tint on paper
            pygame.draw.rect(self.screen, hover_bg, rect)
            pygame.draw.rect(self.screen, P1_COL, (SCORE_PANEL_X+1, y, 3, ROW_H))

        # Grid line (ruled paper effect)
        pygame.draw.line(self.screen, PAPER_LINE,
                         (SCORE_PANEL_X+1, y+ROW_H-1), (WIDTH-1, y+ROW_H-1))

        # Mini dice
        die_y = y + (ROW_H - MINI_DIE_SZ) // 2
        mx    = _SP_X0
        for val, wild in CATEGORY_PREVIEW.get(cat, []):
            fc = MINI_FACE_WILD if wild else MINI_FACE
            dc = MINI_DOT_WILD  if wild else MINI_DOT
            mr = pygame.Rect(mx, die_y, MINI_DIE_SZ, MINI_DIE_SZ)
            pygame.draw.rect(self.screen, fc, mr, border_radius=2)
            pygame.draw.rect(self.screen, MINI_BORDER, mr, width=1, border_radius=2)
            for fx, fy in DOT_MAP.get(val, []):
                pygame.draw.circle(self.screen, dc,
                                   (int(mx+fx*MINI_DIE_SZ), int(die_y+fy*MINI_DIE_SZ)), 2)
            mx += MINI_DIE_SZ + MINI_DIE_GAP

        # Category label (serif for analog feel)
        lbl_col = TEXT_HINT if filled else (TEXT if is_hov else TEXT_SEC)
        lbl_s   = self.f_serif.render(SHORT_LABELS[cat], True, lbl_col)
        self.screen.blit(lbl_s, (_SP_LBL_X, y+(ROW_H-lbl_s.get_height())//2))

        # Score values
        for p in range(2):
            val = game.scorecards[p].scores[cat]
            cx  = (_SP_P1_X if p==0 else _SP_P2_X) + _SP_PCW//2
            cy  = y + ROW_H//2
            if val is not None:
                col = TEXT_HINT if val==0 else TEXT
                s   = self.f_sans_sc.render(str(val), True, col)
            elif p==game.current_player and is_hov and cat in potential:
                pts = potential[cat]
                col = P1_COL if pts>0 else RED_COL
                s   = self.f_sans_sc.render(f"+{pts}", True, col)
            elif p==game.current_player and cat in potential:
                pts = potential[cat]
                col = (80, 160, 80) if pts>0 else TEXT_HINT
                s   = self.f_sans_sm.render(f"({pts})", True, col)
            else:
                s = self.f_sans_sm.render("-", True, TEXT_HINT)
            self.screen.blit(s, s.get_rect(center=(cx, cy)))

        return y+ROW_H, row_idx+1

    # ── HUD badge ─────────────────────────────────────────────────────────────

    def _hud_badge(self, x, y, text, anchor="top-left", text_col=None):
        if text_col is None:
            text_col = TECH_DIM
        s = self.f_mono.render(text, True, text_col)
        w, h = s.get_width()+12, s.get_height()+6
        if anchor == "bottom-right": x -= w; y -= h
        elif anchor == "bottom-left": y -= h
        elif anchor == "top-right": x -= w
        chip = pygame.Surface((w, h), pygame.SRCALPHA)
        chip.fill((3, 8, 4, 205))
        pygame.draw.rect(chip, (*TECH_DIM, 160), chip.get_rect(), width=1, border_radius=3)
        self.screen.blit(chip, (x, y))
        self.screen.blit(s, (x+6, y+3))

    # ── Game over ─────────────────────────────────────────────────────────────

    def _game_over(self, game):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((3, 12, 6, 210))
        self.screen.blit(overlay, (0, 0))

        cx, cy = WIDTH//2, HEIGHT//2
        card = pygame.Rect(cx-270, cy-165, 540, 330)
        # Card shadow
        pygame.draw.rect(self.screen, (2, 8, 4), card.move(4, 5), border_radius=18)
        # Wooden outer
        pygame.draw.rect(self.screen, TRAY_WOOD_DK, card, border_radius=18)
        # Paper inner
        inner = card.inflate(-8, -8)
        pygame.draw.rect(self.screen, PAPER, inner, border_radius=15)
        pygame.draw.rect(self.screen, PAPER_BORDER, inner, width=1, border_radius=15)

        title = self.f_brand.render("Game Over", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(cx, cy-110)))

        pygame.draw.line(self.screen, PAPER_LINE, (cx-200, cy-82), (cx+200, cy-82))

        totals = sorted(
            [(p, game.player_names[p], game.scorecards[p].total()) for p in range(2)],
            key=lambda t: t[2], reverse=True)
        wcol = P1_COL if totals[0][0]==0 else P2_COL
        ws = self.f_serif_m.render(
            f"{totals[0][1]}  wins  with  {totals[0][2]} pts", True, wcol)
        self.screen.blit(ws, ws.get_rect(center=(cx, cy-50)))

        for i, (p, name, pts) in enumerate(totals):
            s = self.f_sans.render(f"{name}:  {pts} points", True,
                                   P1_COL if p==0 else P2_COL)
            self.screen.blit(s, s.get_rect(center=(cx, cy+5+i*28)))

        hint = self.f_sans_sm.render("Press  R  or click  NEW GAME  to play again",
                                     True, TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(center=(cx, inner.bottom-24)))

    # ── Name entry ────────────────────────────────────────────────────────────

    def draw_name_entry(self, name_inputs: list[str], current_idx: int) -> None:
        # Full dark felt background
        self.screen.fill(BG)

        cx = WIDTH // 2
        # Card with wooden frame + paper interior
        outer = pygame.Rect(cx-310, 95, 620, 480)
        pygame.draw.rect(self.screen, (3, 8, 4), outer.move(3,4), border_radius=18)
        pygame.draw.rect(self.screen, TRAY_WOOD_DK, outer, border_radius=18)
        pygame.draw.rect(self.screen, TRAY_WOOD, outer.inflate(-4,-4), border_radius=16)
        inner = outer.inflate(-10, -10)
        pygame.draw.rect(self.screen, PAPER, inner, border_radius=14)
        pygame.draw.rect(self.screen, PAPER_BORDER, inner, width=1, border_radius=14)

        # Brand
        t = pygame.time.get_ticks()
        pulse = 0.5 + 0.5 * math.sin(t / 600)
        dot_col = tuple(int(a + (b-a)*pulse) for a, b in zip(TECH_DIM, TECH_GREEN))
        pygame.draw.circle(self.screen, dot_col, (cx-86, 192), 5)
        logo = self.f_brand.render("Kniffel Vision AI", True, TEXT)
        self.screen.blit(logo, (cx-86+14, 192-logo.get_height()//2))
        sub = self.f_sans_xs.render("COMPUTER VISION POWERED", True, TEXT_DIM)
        self.screen.blit(sub, sub.get_rect(center=(cx, 222)))
        pygame.draw.line(self.screen, PAPER_LINE, (cx-220, 240), (cx+220, 240))

        for i in range(2):
            active = (i == current_idx)
            pcol   = P1_COL if i==0 else P2_COL
            fy     = 258 + i * 118

            lbl = self.f_serif.render(f"Player {i+1}", True, pcol if active else TEXT_DIM)
            self.screen.blit(lbl, (cx-220, fy))

            box = pygame.Rect(cx-220, fy+22, 440, 48)
            pygame.draw.rect(self.screen, PAPER, box, border_radius=8)
            pygame.draw.rect(self.screen, pcol if active else PAPER_BORDER,
                             box, width=2 if active else 1, border_radius=8)

            display = name_inputs[i]
            if active and (pygame.time.get_ticks()//530) % 2 == 0:
                display += "|"
            placeholder = f"Enter name for Player {i+1}..." if not name_inputs[i] else ""
            ts = self.f_input.render(display or placeholder,
                                     True, TEXT if name_inputs[i] else TEXT_HINT)
            self.screen.blit(ts, ts.get_rect(midleft=(box.x+14, box.centery)))

            if i < current_idx:
                ck = self.f_sans_xs.render("confirmed", True, P1_COL)
                self.screen.blit(ck, (box.right+10, box.centery-ck.get_height()//2))

        hint = self.f_sans_sm.render("ENTER to confirm each name     TAB to switch",
                                     True, TEXT_HINT)
        self.screen.blit(hint, hint.get_rect(center=(cx, inner.bottom-24)))
