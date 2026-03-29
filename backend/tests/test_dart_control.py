"""Test DartControl component with goal sequences.

Test 1 – square walk: feeds waypoints (0,4,0) → (4,4,0) → (4,0,0) → (0,0,0)
with instruction "walk" and checks the figure reaches each within a threshold.

Test 2 – orientation goals: sends heading-only goals and checks the figure
turns to face the target direction.

Coordinate systems:
  DART internal: Z-up, ground plane is XY
  Our output (Y-up): -90° rotation around X → (x, z_up, -y) = (pos_x, pos_y, pos_z)
  So: pos_x = DART_x, pos_z = -DART_y
"""

from __future__ import annotations

import math
import time

import numpy as np

from src.core.channel import Channel, Sender, Receiver
from src.core.conduit.dart_control.component import DartControl, DartControlConfig
from src.core.frames import GoalFrame, TextFrame


SUCCESS_THRESHOLD = 0.3  # meters
TIMEOUT = 30  # seconds per waypoint


def test_square_walk():
    # Figure starts facing +Z in Y-up space, +X = figure's right
    # Goals are Y-up: (x=right, y=height, z=forward)
    waypoints = [
        (0.0, 0.0, 4.0),  # straight ahead
        (-4.0, 0.0, 4.0),  # turn left
        (-4.0, 0.0, 0.0),  # turn left again
        (0.0, 0.0, 0.0),  # back to origin
    ]

    # Create channels
    goal_ch = Channel()
    goal_sender = Sender(goal_ch)
    goal_receiver = Receiver(goal_ch)

    instruction_ch = Channel()
    instruction_sender = Sender(instruction_ch)
    instruction_receiver = Receiver(instruction_ch)

    motion_ch = Channel()
    motion_sender = Sender(motion_ch)
    motion_receiver = Receiver(motion_ch)

    # Use CPU for reproducibility (policy trained on CUDA, MPS may differ)
    config = DartControlConfig(device="cpu")
    dart = DartControl(config)

    from src.core.conduit.dart_control.component import (
        DartControlInputs,
        DartControlOutputs,
    )

    inputs = DartControlInputs(
        goal=goal_receiver,
        instruction=instruction_receiver,
    )
    outputs = DartControlOutputs(motion=motion_sender)

    # Start component
    dart.start(inputs, outputs)

    # Give it time to load models
    time.sleep(5)

    # Send initial instruction
    instruction_sender.send(TextFrame.new(text="walk"))

    # Create motion iterator
    motion_it = motion_receiver(dart, newest=True)

    reached = []
    for wp_idx, (gx, gy, gz) in enumerate(waypoints):
        print(f"\n--- Waypoint {wp_idx}: DART({gx}, {gy}, {gz}) ---")
        instruction_sender.send(TextFrame.new(text="walk"))
        goal_sender.send(GoalFrame.new(x=gx, y=gy, z=gz))

        start_time = time.time()
        step = 0
        while time.time() - start_time < TIMEOUT:
            frame = next(motion_it)
            if frame is None:
                break

            poses = frame.get()
            waist = poses.get("waist")
            if waist is None:
                continue

            # Y-up: ground plane is (x, z), y is height
            px, pz = waist.pos_x, waist.pos_z

            dist = np.sqrt((px - gx) ** 2 + (pz - gz) ** 2)

            step += 1
            if step % 30 == 0:  # print every ~1 second at 30fps
                print(
                    f"  pos=({px:.2f}, {pz:.2f}), "
                    f"goal=({gx:.2f}, {gz:.2f}), dist={dist:.2f}"
                )

            if dist < SUCCESS_THRESHOLD:
                elapsed = time.time() - start_time
                print(
                    f"  Reached waypoint {wp_idx} in {elapsed:.1f}s (dist={dist:.2f})"
                )
                reached.append(wp_idx)
                break
        else:
            print(f"  TIMEOUT: did not reach waypoint {wp_idx}")

    dart.stop()

    print(f"\n=== Results: reached {len(reached)}/{len(waypoints)} waypoints ===")
    print(f"Reached: {reached}")

    assert len(reached) == len(waypoints), (
        f"Only reached {len(reached)}/{len(waypoints)} waypoints: {reached}"
    )


def _heading_from_quat(w: float, x: float, y: float, z: float) -> float:
    """Extract heading in degrees (clockwise from +Z) from a Y-up quaternion."""
    fwd_x = 2 * (x * z + w * y)
    fwd_z = 1 - 2 * (x * x + y * y)
    return -math.degrees(math.atan2(fwd_x, fwd_z))


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angle difference in degrees, wrapped to [-180, 180]."""
    d = (a - b) % 360
    if d > 180:
        d -= 360
    return d


HEADING_THRESHOLD = 20.0  # degrees
HEADING_TIMEOUT = 15  # seconds per target heading


def test_orientation_goal():
    """Send heading-only goals and verify the figure turns to face them."""
    # Start facing ~0° (+Z). Ask to turn to 90°, 180°, then back to 0°.
    target_headings = [90.0, 180.0, 0.0]

    # Create channels
    goal_ch = Channel()
    goal_sender = Sender(goal_ch)
    goal_receiver = Receiver(goal_ch)

    instruction_ch = Channel()
    instruction_sender = Sender(instruction_ch)
    instruction_receiver = Receiver(instruction_ch)

    motion_ch = Channel()
    motion_sender = Sender(motion_ch)
    motion_receiver = Receiver(motion_ch)

    config = DartControlConfig(device="cpu")
    dart = DartControl(config)

    from src.core.conduit.dart_control.component import (
        DartControlInputs,
        DartControlOutputs,
    )

    inputs = DartControlInputs(
        goal=goal_receiver,
        instruction=instruction_receiver,
    )
    outputs = DartControlOutputs(motion=motion_sender)

    dart.start(inputs, outputs)
    time.sleep(5)

    # Send a turn instruction (no position goal, heading only)
    instruction_sender.send(TextFrame.new(text="turn"))

    motion_it = motion_receiver(dart, newest=True)

    reached = []
    for h_idx, target in enumerate(target_headings):
        print(f"\n--- Heading {h_idx}: target={target}° ---")
        goal_sender.send(GoalFrame.new(heading=target))

        start_time = time.time()
        step = 0
        while time.time() - start_time < HEADING_TIMEOUT:
            frame = next(motion_it)
            if frame is None:
                break

            poses = frame.get()
            waist = poses.get("waist")
            if waist is None:
                continue

            heading = _heading_from_quat(
                waist.rot_w, waist.rot_x, waist.rot_y, waist.rot_z
            )
            diff = abs(_angle_diff(heading, target))

            step += 1
            if step % 30 == 0:
                print(f"  heading={heading:.1f}°, target={target}°, diff={diff:.1f}°")

            if diff < HEADING_THRESHOLD:
                elapsed = time.time() - start_time
                print(
                    f"  Reached heading {h_idx} in {elapsed:.1f}s "
                    f"(heading={heading:.1f}°, diff={diff:.1f}°)"
                )
                reached.append(h_idx)
                break
        else:
            print(f"  TIMEOUT: did not reach heading {h_idx}")

    dart.stop()

    print(f"\n=== Results: reached {len(reached)}/{len(target_headings)} headings ===")
    print(f"Reached: {reached}")

    assert len(reached) == len(target_headings), (
        f"Only reached {len(reached)}/{len(target_headings)} headings: {reached}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "orientation":
        test_orientation_goal()
    elif len(sys.argv) > 1 and sys.argv[1] == "walk":
        test_square_walk()
    else:
        test_square_walk()
        test_orientation_goal()
