from __future__ import annotations

import argparse
import json
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    valid: bool = False


@dataclass(frozen=True)
class JointState:
    tracked: bool = False
    pose: Pose | None = None
    radius: float | None = None


@dataclass(frozen=True)
class HandState:
    tracked: bool = False
    wrist: JointState = field(default_factory=JointState)
    palm: JointState = field(default_factory=JointState)
    fingertips: dict[str, JointState] = field(default_factory=dict)


@dataclass(frozen=True)
class StatusState:
    connection_state: str = ""
    target_host: str = ""


@dataclass(frozen=True)
class TelemetryFrame:
    sequence: int
    transport: str
    tracking_valid: bool
    display_time_ns: int
    head_pose: Pose
    left_hand: HandState
    right_hand: HandState
    status: StatusState
    raw: dict[str, Any]


@dataclass(frozen=True)
class ReceiverMode:
    name: str
    detail: str


@dataclass
class PacketDecoder:
    buffer: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buffer.extend(chunk)
        packets: list[bytes] = []
        while len(self.buffer) >= 4:
            packet_length = int.from_bytes(self.buffer[:4], byteorder="big", signed=False)
            if len(self.buffer) < 4 + packet_length:
                break
            start = 4
            end = 4 + packet_length
            packets.append(bytes(self.buffer[start:end]))
            del self.buffer[:end]
        return packets


def _float_tuple(values: list[Any], size: int, default: list[float]) -> tuple[float, ...]:
    padded = list(values)[:size] + default[len(values[:size]):size]
    return tuple(float(value) for value in padded)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for Quest telemetry and expose it through a simple MetaReader API.")
    parser.add_argument("--bind-host", default="0.0.0.0", help="Bind host for the active listener. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=5005, help="Quest transport port and wireless UDP port. Default: 5005")
    parser.add_argument("--tcp-port", type=int, default=5005, help="Host TCP port to bind for adb reverse. Default: 5005")
    parser.add_argument("--service-name", default="Quest Bridge", help="mDNS/DNS-SD service name.")
    parser.add_argument("--ros-domain-id", default="0", help="TXT ros_domain_id value.")
    parser.add_argument("--node", default="quest_bridge", help="TXT node value.")
    parser.add_argument("--caps", default="pose,hands,fingertips,hmd", help="TXT caps value.")
    parser.add_argument("--no-advertise", action="store_true", help="Skip Avahi publication in wireless mode.")
    parser.add_argument("--no-auto-adb-reverse", action="store_true", help="Do not auto-detect ADB devices or configure adb reverse.")
    return parser.parse_args(argv)


def _make_pose(data: dict[str, Any], valid_key: str = "valid") -> Pose:
    return Pose(
        position=_float_tuple(data.get("position", [0.0, 0.0, 0.0]), 3, [0.0, 0.0, 0.0]),
        orientation=_float_tuple(data.get("orientation", [0.0, 0.0, 0.0, 1.0]), 4, [0.0, 0.0, 0.0, 1.0]),
        valid=bool(data.get(valid_key, False)),
    )


def _make_joint(data: dict[str, Any]) -> JointState:
    tracked = bool(data.get("tracked", False))
    if tracked:
        pose_data = dict(data.get("pose", {}))
        pose_data["valid"] = True
        pose = _make_pose(pose_data)
    else:
        pose = None
    radius = data.get("radius")
    return JointState(tracked=tracked, pose=pose, radius=float(radius) if radius is not None else None)


def _make_hand(data: dict[str, Any]) -> HandState:
    fingertips = {
        name: _make_joint(value)
        for name, value in data.get("fingertips", {}).items()
    }
    return HandState(
        tracked=bool(data.get("tracked", False)),
        wrist=_make_joint(data.get("wrist", {})),
        palm=_make_joint(data.get("palm", {})),
        fingertips=fingertips,
    )


def parse_frame(payload: bytes) -> TelemetryFrame:
    packet = json.loads(payload.decode("utf-8"))
    hands = packet.get("hands", {})
    status = packet.get("status", {})
    return TelemetryFrame(
        sequence=int(packet.get("sequence", 0)),
        transport=str(packet.get("transport", "unknown")),
        tracking_valid=bool(packet.get("tracking_valid", False)),
        display_time_ns=int(packet.get("display_time_ns", 0)),
        head_pose=_make_pose(packet.get("head_pose", {})),
        left_hand=_make_hand(hands.get("left", {})),
        right_hand=_make_hand(hands.get("right", {})),
        status=StatusState(
            connection_state=str(status.get("connection_state", "")),
            target_host=str(status.get("target_host", "")),
        ),
        raw=packet,
    )


def describe_frame(frame: TelemetryFrame) -> str:
    def tracked_fingertips(hand: HandState) -> str:
        tracked = [name for name, value in hand.fingertips.items() if value.tracked]
        return ",".join(tracked) if tracked else "none"

    return (
        f"seq={frame.sequence} transport={frame.transport} tracking_valid={frame.tracking_valid} "
        f"pos=({frame.head_pose.position[0]:.3f}, {frame.head_pose.position[1]:.3f}, {frame.head_pose.position[2]:.3f}) "
        f"left_hand={frame.left_hand.tracked}[{tracked_fingertips(frame.left_hand)}] "
        f"right_hand={frame.right_hand.tracked}[{tracked_fingertips(frame.right_hand)}]"
    )


def _start_avahi_publisher(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if args.no_advertise:
        return None

    avahi_binary = shutil.which("avahi-publish-service")
    if avahi_binary is None:
        print("avahi-publish-service not found; running wireless receiver without DNS-SD advertisement.", file=sys.stderr)
        return None

    command = [
        avahi_binary,
        args.service_name,
        "_quest-teleop._udp",
        str(args.port),
        "proto=1",
        f"ros_domain_id={args.ros_domain_id}",
        f"node={args.node}",
        f"caps={args.caps}",
    ]
    print("Advertising service:", " ".join(command))
    return subprocess.Popen(command)


def _make_udp_socket(bind_host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_host, port))
    sock.setblocking(False)
    return sock


def _make_tcp_socket(bind_host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(f"Setting up TCP socket on {bind_host}:{port} for adb reverse")
    sock.bind((bind_host, port))
    sock.listen()
    sock.setblocking(False)
    return sock


def _detect_adb_device(adb_binary: str | None) -> bool:
    if not adb_binary:
        return False
    result = subprocess.run([adb_binary, "devices"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if "\tdevice" in stripped:
            return True
    return False


def _configure_adb_reverse(adb_binary: str, quest_port: int, host_port: int) -> None:
    command = [adb_binary, "reverse", f"tcp:{quest_port}", f"tcp:{host_port}"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown adb reverse failure"
        raise RuntimeError(f"Failed to configure adb reverse: {message}")


def _remove_adb_reverse(adb_binary: str | None, quest_port: int) -> None:
    if not adb_binary:
        return
    subprocess.run([adb_binary, "reverse", "--remove", f"tcp:{quest_port}"], capture_output=True, text=True, check=False)


class MetaReader:
    def __init__(
        self,
        *,
        bind_host: str = "0.0.0.0",
        port: int = 5005,
        tcp_port: int = 5005,
        service_name: str = "Quest Bridge",
        ros_domain_id: str = "0",
        node: str = "quest_bridge",
        caps: str = "pose,hands,fingertips,hmd",
        no_advertise: bool = False,
        auto_adb_reverse: bool = True,
    ) -> None:
        self.args = argparse.Namespace(
            bind_host=bind_host,
            port=port,
            tcp_port=tcp_port,
            service_name=service_name,
            ros_domain_id=ros_domain_id,
            node=node,
            caps=caps,
            no_advertise=no_advertise,
            no_auto_adb_reverse=not auto_adb_reverse,
        )
        self.selector = selectors.DefaultSelector()
        self.publisher: subprocess.Popen[str] | None = None
        self.pending_frames: deque[TelemetryFrame] = deque()
        self.adb_binary = shutil.which("adb")
        self.reverse_configured = False
        self.mode = self._select_mode()
        self._configure_transport()

    def _select_mode(self) -> ReceiverMode:
        if not self.args.no_auto_adb_reverse and _detect_adb_device(self.adb_binary):
            detail = f"adb reverse tcp:{self.args.port} -> tcp:{self.args.tcp_port}"
            return ReceiverMode(name="wired", detail=detail)
        detail = f"avahi dns-sd on udp:{self.args.port}"
        return ReceiverMode(name="wireless", detail=detail)

    def _configure_transport(self) -> None:
        if self.mode.name == "wired":
            if not self.adb_binary:
                raise RuntimeError("ADB was selected for wired mode but adb is not available in PATH.")
            _configure_adb_reverse(self.adb_binary, self.args.port, self.args.tcp_port)
            self.reverse_configured = True
            tcp_socket = _make_tcp_socket(self.args.bind_host, self.args.tcp_port)
            self.selector.register(tcp_socket, selectors.EVENT_READ, ("tcp-listener", None))
            return

        udp_socket = _make_udp_socket(self.args.bind_host, self.args.port)
        self.selector.register(udp_socket, selectors.EVENT_READ, ("udp", PacketDecoder()))
        self.publisher = _start_avahi_publisher(self.args)

    def close(self) -> None:
        for registered_key in list(self.selector.get_map().values()):
            self.selector.unregister(registered_key.fileobj)
            registered_key.fileobj.close()
        self.selector.close()
        if self.publisher is not None:
            self.publisher.terminate()
            try:
                self.publisher.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.publisher.kill()
        if self.reverse_configured:
            _remove_adb_reverse(self.adb_binary, self.args.port)

    def __enter__(self) -> MetaReader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def read(self, timeout: float | None = None) -> TelemetryFrame | None:
        return self._read_frame(timeout, latest=False)

    def read_latest(self, timeout: float | None = None) -> TelemetryFrame | None:
        return self._read_frame(timeout, latest=True)

    def _read_frame(self, timeout: float | None, *, latest: bool) -> TelemetryFrame | None:
        pop_frame = self._pop_latest_pending if latest else self.pending_frames.popleft
        if self.pending_frames:
            return pop_frame()

        deadline = None if timeout is None else time.monotonic() + timeout
        frame = None
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if deadline is not None and remaining == 0.0:
                return None

            events = self.selector.select(timeout=remaining)
            if not events:
                return None

            for key, _ in events:
                self._handle_event(key)

            if self.pending_frames:
                frame = pop_frame()
                if not latest:
                    return frame

            if not latest:
                continue

            while True:
                events = self.selector.select(timeout=0.0)
                if not events:
                    return frame
                for key, _ in events:
                    self._handle_event(key)
                if self.pending_frames:
                    frame = self._pop_latest_pending()

    def _pop_latest_pending(self) -> TelemetryFrame | None:
        if not self.pending_frames:
            return None

        latest = self.pending_frames.pop()
        self.pending_frames.clear()
        return latest

    def _handle_event(self, key: selectors.SelectorKey) -> None:
        kind = key.data[0]
        if kind == "udp":
            decoder = key.data[1]
            payload, _address = key.fileobj.recvfrom(65535)
            for packet in decoder.feed(payload):
                self.pending_frames.append(parse_frame(packet))
            return

        if kind == "tcp-listener":
            client_socket, address = key.fileobj.accept()
            client_socket.setblocking(False)
            client_decoder = PacketDecoder()
            self.selector.register(client_socket, selectors.EVENT_READ, ("tcp-client", client_decoder, address))
            return

        _, client_decoder, _address = key.data
        payload = key.fileobj.recv(65535)
        if not payload:
            self.selector.unregister(key.fileobj)
            key.fileobj.close()
            return

        self.pending_frames.extend(parse_frame(packet) for packet in client_decoder.feed(payload))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with MetaReader(
        bind_host=args.bind_host,
        port=args.port,
        tcp_port=args.tcp_port,
        service_name=args.service_name,
        ros_domain_id=args.ros_domain_id,
        node=args.node,
        caps=args.caps,
        no_advertise=args.no_advertise,
        auto_adb_reverse=not args.no_auto_adb_reverse,
    ) as reader:
        print(f"Receiver mode: {reader.mode.name} ({reader.mode.detail})")
        if reader.mode.name == "wired":
            print(f"Wired TCP receiver listening on {args.bind_host}:{args.tcp_port}")
        else:
            print(f"Wireless UDP receiver listening on {args.bind_host}:{args.port}")

        def shutdown_handler(signum, frame):
            del signum, frame
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        packet_count = 0
        try:
            while True:
                frame = reader.read(timeout=1.0)
                if frame is None:
                    continue
                packet_count += 1
                print(f"[{packet_count}] {describe_frame(frame)}")
        except KeyboardInterrupt:
            print("Stopping receiver.")

    return 0