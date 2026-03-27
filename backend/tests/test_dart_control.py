"""Test DartControl component with a square walk goal sequence.

Feeds the same waypoints as DART's demo_walk.json:
  (0,4,0) → (4,4,0) → (4,0,0) → (0,0,0)
with instruction "walk" at each waypoint.

Checks that the figure reaches each waypoint within a threshold.

Coordinate systems:
  DART internal: Z-up, ground plane is XY
  Our output (Y-up): -90° rotation around X → (x, z_up, -y) = (pos_x, pos_y, pos_z)
  So: pos_x = DART_x, pos_z = -DART_y
"""

from __future__ import annotations

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


if __name__ == "__main__":
    test_square_walk()
