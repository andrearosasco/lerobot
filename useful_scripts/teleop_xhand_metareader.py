#!/usr/bin/env python3

import argparse
import time

from lerobot.robots.custom_manipulator.grippers.xhand import XHand, XHandConfig
from lerobot.teleoperators.metareader import MetaReaderConfig, MetaReaderTeleoperator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate xHand directly from MetaReader without Panda.")
    parser.add_argument("--protocol", choices=["EtherCAT", "RS485"], default="EtherCAT")
    parser.add_argument("--hand-id", type=int, default=0)
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--baud-rate", type=int, default=3000000)
    parser.add_argument("--port", type=int, default=5005, help="MetaReader Quest transport port")
    parser.add_argument("--tcp-port", type=int, default=5005, help="Host TCP port for adb reverse")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--enable-tip-ik", action="store_true")
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--palm-link-name", default="palm")
    parser.add_argument("--thumb-link", default="thumb_tip")
    parser.add_argument("--index-link", default="index_tip")
    parser.add_argument("--middle-link", default="middle_tip")
    parser.add_argument("--ring-link", default="ring_tip")
    parser.add_argument("--little-link", default="little_tip")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    teleop = MetaReaderTeleoperator(
        MetaReaderConfig(
            port=args.port,
            tcp_port=args.tcp_port,
        )
    )

    hand = XHand(
        XHandConfig(
            protocol=args.protocol,
            hand_id=args.hand_id,
            serial_port=args.serial_port,
            baud_rate=args.baud_rate,
            startup_delay_s=args.startup_delay,
            enable_tip_ik=args.enable_tip_ik,
            require_ik=args.enable_tip_ik,
            connect_reset=False,
            urdf_path=args.urdf_path,
            palm_link_name=args.palm_link_name,
            tip_link_names={
                "thumb": args.thumb_link,
                "index": args.index_link,
                "middle": args.middle_link,
                "ring": args.ring_link,
                "little": args.little_link,
            },
        )
    )

    loop_dt = 1.0 / args.rate_hz

    teleop.connect()
    hand.connect()

    print("Connected to MetaReader and xHand.", flush=True)
    print("Hold the right hand tracked by Quest to drive the hand. Ctrl+C to stop.", flush=True)
    if args.enable_tip_ik:
        print("Tip IK enabled: fingertip targets will be sent to xHand.", flush=True)
    else:
        print("Tip IK disabled: only the pinch-derived gripper scalar will be used.", flush=True)

    try:
        while True:
            start_t = time.perf_counter()
            action = teleop.get_action()

            if action.get("exit_episode", 0.0) > 0.5:
                break

            if action.get("is_engaged", 0.0) > 0.5:
                hand.apply_commands(action=action, gripper_state=action.get("gripper", 1.0))

            elapsed = time.perf_counter() - start_t
            if elapsed < loop_dt:
                time.sleep(loop_dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        hand.disconnect()
        teleop.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())