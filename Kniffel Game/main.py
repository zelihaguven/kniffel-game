import pygame
import sys
import argparse

from engine.game import GameState, Phase
from ui.renderer import Renderer
from ui.constants import WIDTH, HEIGHT, FPS
from cv_interface.camera import CameraFeed
from cv_interface.detector import DetectorMode


def make_camera(source: int) -> tuple[CameraFeed, bool]:
    cam = CameraFeed(source=source)
    ok  = cam.start()
    if not ok:
        print(f"[warning] Camera {source} not found — press SPACE for random roll")
    else:
        print(f"[camera] Using camera source {source}")
    return cam, ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Kniffel")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera source index (0 = built-in, 1 = external). Default: 0")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock  = pygame.time.Clock()

    cam_source = args.camera
    camera, cam_ok = make_camera(cam_source)
    pygame.display.set_caption(f"Kniffel  [cam {cam_source}]")

    renderer = Renderer(screen)

    # ── App state ──────────────────────────────────────────────────────────────
    app_state        = "name_entry"   # "name_entry" | "playing"
    name_inputs      = ["", ""]       # raw typed names
    current_name_idx = 0              # which player is typing
    player_names     = ["Player 1", "Player 2"]
    game             = None
    hovered_category = None
    frame_count      = 0

    while True:
        cam_frame, _, detections = camera.get_latest()

        if frame_count < 5 and cam_frame is not None and app_state == "playing":
            print(f"[main] Frame {frame_count}: {cam_frame.shape} | Detections: {len(detections)}")
        frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                camera.stop()
                pygame.quit()
                sys.exit()

            # ── Name entry input ───────────────────────────────────────────────
            if app_state == "name_entry":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        entered = name_inputs[current_name_idx].strip()
                        if entered:
                            player_names[current_name_idx] = entered
                        if current_name_idx == 0:
                            current_name_idx = 1
                        else:
                            game = GameState(player_names=list(player_names))
                            hovered_category = None
                            app_state = "playing"
                    elif event.key == pygame.K_BACKSPACE:
                        name_inputs[current_name_idx] = name_inputs[current_name_idx][:-1]
                    elif event.key == pygame.K_TAB:
                        current_name_idx = 1 - current_name_idx
                    else:
                        if len(name_inputs[current_name_idx]) < 18:
                            name_inputs[current_name_idx] += event.unicode
                continue  # skip game events during name entry

            # ── Game events ────────────────────────────────────────────────────
            elif event.type == pygame.KEYDOWN:
                # R — go back to name entry (full reset)
                if event.key == pygame.K_r:
                    app_state        = "name_entry"
                    name_inputs      = ["", ""]
                    current_name_idx = 0
                    player_names     = ["Player 1", "Player 2"]
                    game             = None
                    hovered_category = None

                # SPACE — random roll fallback
                if event.key == pygame.K_SPACE and game and game.can_capture():
                    game.roll_random()

                # C — toggle camera source
                if event.key == pygame.K_c:
                    camera.stop()
                    cam_source = 1 - cam_source
                    camera, cam_ok = make_camera(cam_source)
                    pygame.display.set_caption(f"Kniffel  [cam {cam_source}]")

                # D — toggle CV detector mode
                if event.key == pygame.K_d:
                    new_mode = camera.toggle_mode()
                    print(f"[detector] Switched to {new_mode.value}")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and game:
                pos = event.pos
                count = len(detections)

                # ROLL DICE
                if renderer.capture_btn_rect.collidepoint(pos) and game.can_capture():
                    if detections:
                        game.capture(detections)
                    elif not cam_ok:
                        game.roll_random()

                # HOLD ALL
                elif renderer.hold_all_btn_rect.collidepoint(pos) and game.phase == Phase.HOLD:
                    for i in range(len(game.dice)):
                        if not game.dice[i].held:
                            game.toggle_hold(i)

                # NEW GAME — full reset back to name entry
                elif renderer.reset_btn_rect.collidepoint(pos):
                    app_state        = "name_entry"
                    name_inputs      = ["", ""]
                    current_name_idx = 0
                    player_names     = ["Player 1", "Player 2"]
                    game             = None
                    hovered_category = None

                # RELEASE ALL
                elif renderer.release_all_rect.collidepoint(pos):
                    game.dice.release_all()

                # Per-die HOLD / RELEASE buttons
                elif any(rect.collidepoint(pos) for rect in renderer.hold_btn_rects):
                    for i, rect in enumerate(renderer.hold_btn_rects):
                        if rect.collidepoint(pos):
                            game.toggle_hold(i)

                # Clicking the die face toggles hold
                elif any(rect.collidepoint(pos) for rect in renderer.die_rects):
                    for i, rect in enumerate(renderer.die_rects):
                        if rect.collidepoint(pos):
                            game.toggle_hold(i)

                # Scorecard row click
                elif hovered_category is not None and game.can_score():
                    game.score(hovered_category)
                    hovered_category = None

            elif event.type == pygame.MOUSEMOTION and game:
                hovered_category = renderer.get_hovered_category(event.pos)

        # ── Draw ──────────────────────────────────────────────────────────────
        if app_state == "name_entry":
            renderer.draw_name_entry(name_inputs, current_name_idx)
        else:
            renderer.draw(game, cam_frame, detections, hovered_category,
                          view_player=game.current_player,
                          detector_mode=camera.get_mode(),
                          cam_source=cam_source)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
