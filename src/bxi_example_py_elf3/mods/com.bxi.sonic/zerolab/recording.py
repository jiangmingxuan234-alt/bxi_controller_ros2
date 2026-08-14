import json
import os
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterator, Mapping

from .protocol import PACKET_SIZE


PROTOCOL_VERSION = "zerolab-f2pro-udp-v1"
MAPPING_VERSION = "zerolab17-smpl24-v1"
RECORD_STRUCT = struct.Struct("<QQ992s")
RECORD_SIZE = RECORD_STRUCT.size
BODY_JOINT_ORDER = (
    "head", "chest", "left_shoulder", "left_upper_arm", "left_forearm",
    "left_hand", "right_shoulder", "right_upper_arm", "right_forearm",
    "right_hand", "hips", "left_thigh", "left_calf", "left_foot",
    "right_thigh", "right_calf", "right_foot",
)
JOINT_ORDER = BODY_JOINT_ORDER + tuple(
    f"raw_joint_{index}" for index in range(17, 47)
)


@dataclass(frozen=True)
class RawRecord:
    receive_timestamp_ns: int
    local_frame_index: int
    payload: bytes


def build_recording_metadata(
    sender_address: tuple[str, int], start_time_utc: str
) -> dict[str, object]:
    return {
        "format": "zerolab_f2pro_raw",
        "format_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "packet_size": PACKET_SIZE,
        "record_size": RECORD_SIZE,
        "expected_rate_hz": 50,
        "endianness": "little",
        "quaternion_order": "xyzw",
        "receive_clock": "time.monotonic_ns",
        "sender": {"host": sender_address[0], "port": sender_address[1]},
        "start_time_utc": start_time_utc,
        "joint_order": JOINT_ORDER,
        "mapping_version": MAPPING_VERSION,
        "sensor_count": 15,
        "shoulder_sensors": {"left": True, "right": True},
    }


class RawRecordingWriter:
    def __init__(
        self,
        directory: Path,
        metadata: Mapping[str, object],
        flush_every: int = 50,
    ) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self._directory = Path(directory)
        self._directory.mkdir()
        with (self._directory / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(dict(metadata), stream, indent=2, sort_keys=True)
            stream.write("\n")
        self._stream = (self._directory / "records.bin").open("wb")
        self._flush_every = flush_every
        self._record_count = 0
        self._closed = False

    def append(self, record: RawRecord) -> None:
        if self._closed:
            raise ValueError("cannot append to a closed recording")
        if len(record.payload) != PACKET_SIZE:
            raise ValueError(f"record payload must be exactly {PACKET_SIZE} bytes")
        self._stream.write(
            RECORD_STRUCT.pack(
                record.receive_timestamp_ns, record.local_frame_index, record.payload
            )
        )
        self._record_count += 1
        if self._record_count % self._flush_every == 0:
            self._stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._closed = True

    def __enter__(self) -> "RawRecordingWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def read_recording_metadata(directory: Path) -> dict[str, object]:
    metadata_path = Path(directory) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def iter_raw_records(directory: Path) -> Iterator[RawRecord]:
    directory = Path(directory)
    read_recording_metadata(directory)
    with (directory / "records.bin").open("rb") as stream:
        while True:
            encoded_record = stream.read(RECORD_SIZE)
            if not encoded_record:
                return
            if len(encoded_record) != RECORD_SIZE:
                raise ValueError("truncated trailing record")
            timestamp_ns, frame_index, payload = RECORD_STRUCT.unpack(encoded_record)
            yield RawRecord(timestamp_ns, frame_index, payload)
