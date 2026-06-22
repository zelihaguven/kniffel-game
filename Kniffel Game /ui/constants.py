WIDTH, HEIGHT = 1280, 720
FPS = 60

HEADER_H      = 54
CAM_PANEL_W   = 756
SCORE_PANEL_X = 756
SCORE_PANEL_W = 524
PAD           = 14

# ── Felt green (left panel + window bg) ──────────────────────────────────────
BG            = (13,  31,  20)   # #0D1F14  deep casino felt
FELT          = (18,  44,  28)   # #1A3A2A  medium felt
FELT_DARK     = (10,  24,  15)   # darker felt shadow

# ── Header ────────────────────────────────────────────────────────────────────
HDR_BG        = (6,   14,   8)   # very dark green
HDR_GOLD      = (210, 175,  80)  # title gold
HDR_TEXT      = (230, 220, 195)  # warm ivory
HDR_MUTED     = (100, 130, 105)  # muted on dark
HDR_BORDER    = (24,  52,  28)   # subtle separator

# ── Camera / tech zone ───────────────────────────────────────────────────────
CAM_CARD      = (6,   14,   8)   # card background
CAM_VIEWPORT  = (3,    7,   4)   # near black inner
CAM_HUD_HI    = (0,  220,  90)   # bright HUD green
CAM_HUD_LO    = (0,   80,  35)   # dimmed HUD green

# ── Dice tray zone (wooden) ───────────────────────────────────────────────────
TRAY_FELT     = (14,  36,  22)   # inner felt surface
TRAY_WOOD     = (139, 115,  85)  # #8B7355  main wood
TRAY_WOOD_DK  = (90,  70,  45)   # shadow wood
TRAY_WOOD_LT  = (175, 152, 112)  # highlight wood

# ── Die face (physical, ivory + wood border) ─────────────────────────────────
DIE_FACE        = (245, 240, 232)  # #F5F0E8  warm ivory
DIE_BORDER      = (139, 115,  85)  # #8B7355  wood
DIE_DOT         = (40,  26,  14)   # warm dark brown
DIE_SHADOW      = (6,   15,   8)   # deep shadow

DIE_HELD_FACE   = (255, 248, 210)  # warm amber ivory
DIE_HELD_BORDER = (195, 152,  38)  # gold border
DIE_HELD_DOT    = (80,  45,  10)   # dark amber pip

DIE_CV_BORDER   = (0,  200,  85)   # bright green CV detection

# ── Paper / scorecard zone ────────────────────────────────────────────────────
PAPER        = (250, 247, 240)  # #FAF7F0  warm cream
PAPER_ALT    = (244, 239, 226)  # slightly darker alt row
PAPER_BORDER = (212, 201, 168)  # #D4C9A8  parchment border
PAPER_LINE   = (232, 222, 202)  # faint grid lines

# ── Paper typography ──────────────────────────────────────────────────────────
TEXT      = (42,  28,  16)   # warm near-black
TEXT_SEC  = (88,  70,  48)   # medium warm brown
TEXT_DIM  = (148, 130, 100)  # light brown
TEXT_HINT = (190, 172, 145)  # very faint

# ── On-dark typography (left panel) ───────────────────────────────────────────
ON_DARK     = (225, 218, 200)  # ivory on felt
ON_DARK_DIM = (120, 140, 122)  # muted on felt

# ── Gold accent ───────────────────────────────────────────────────────────────
GOLD      = (208, 168,  50)   # #D0A832
GOLD_HV   = (186, 148,  35)   # darker hover
GOLD_TEXT = (22,  12,   4)    # dark text on gold
GOLD_BG   = (55,  45,  18)    # subtle gold tint bg

# ── Tech green ────────────────────────────────────────────────────────────────
TECH_GREEN = (0,  220,  90)
TECH_DIM   = (0,  100,  42)

# ── Player / score accents (on paper) ────────────────────────────────────────
P1_COL     = (18, 140,  70)   # dark green
P2_COL     = (45, 100, 200)   # blue
AMBER_COL  = (180, 120,  20)  # amber on paper
RED_COL    = (180,  40,  40)  # red on paper

# ── Score rows ───────────────────────────────────────────────────────────────
SCORE_ROW_H   = 25

# ── Mini dice ─────────────────────────────────────────────────────────────────
MINI_DIE_SZ   = 11
MINI_DIE_GAP  = 2
MINI_AREA_W   = 5 * MINI_DIE_SZ + 4 * MINI_DIE_GAP   # 63

MINI_FACE     = (232, 224, 208)
MINI_BORDER   = (175, 155, 120)
MINI_DOT      = (88,  70,  48)
MINI_FACE_WILD = (218, 210, 196)
MINI_DOT_WILD  = (155, 140, 112)

# ── Pip positions (normalised 0–1 inside face) ────────────────────────────────
DOT_MAP: dict[int, list[tuple[float, float]]] = {
    1: [(0.50, 0.50)],
    2: [(0.30, 0.30), (0.70, 0.70)],
    3: [(0.30, 0.30), (0.50, 0.50), (0.70, 0.70)],
    4: [(0.30, 0.30), (0.70, 0.30), (0.30, 0.70), (0.70, 0.70)],
    5: [(0.30, 0.30), (0.70, 0.30), (0.50, 0.50), (0.30, 0.70), (0.70, 0.70)],
    6: [(0.30, 0.22), (0.70, 0.22), (0.30, 0.50), (0.70, 0.50),
        (0.30, 0.78), (0.70, 0.78)],
}
