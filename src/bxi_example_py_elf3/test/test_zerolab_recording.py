import pytest

from zerolab.recording import (
    RECORD_SIZE,
    RawRecord,
    RawRecordingWriter,
    build_recording_metadata,
    iter_raw_records,
    read_recording_metadata,
)


def test_recording_round_trip_preserves_timestamp_index_and_all_payload_bytes(tmp_path):
    out = tmp_path / "trial"
    metadata = build_recording_metadata(
        sender_address=("192.168.1.20", 50000),
        start_time_utc="2026-08-04T10:00:00+08:00",
    )
    records = [
        RawRecord(1000, 3, bytes([17]) * 992),
        RawRecord(1020, 4, bytes(range(256)) * 3 + bytes(range(224))),
    ]
    with RawRecordingWriter(out, metadata, flush_every=1) as writer:
        for record in records:
            writer.append(record)
    assert (out / "records.bin").stat().st_size == 2 * RECORD_SIZE
    assert list(iter_raw_records(out)) == records
    loaded = read_recording_metadata(out)
    assert loaded["packet_size"] == 992
    assert loaded["expected_rate_hz"] == 50
    assert loaded["sensor_count"] == 15
    assert loaded["shoulder_sensors"] == {"left": True, "right": True}
    assert loaded["sender"] == {"host": "192.168.1.20", "port": 50000}
    assert len(loaded["joint_order"]) == 47


def test_writer_rejects_non_992_payload(tmp_path):
    metadata = build_recording_metadata(("127.0.0.1", 1), "2026-08-04T00:00:00Z")
    with RawRecordingWriter(tmp_path / "trial", metadata) as writer:
        with pytest.raises(ValueError, match="992"):
            writer.append(RawRecord(1, 0, bytes(991)))


def test_reader_rejects_truncated_trailing_record(tmp_path):
    out = tmp_path / "trial"
    metadata = build_recording_metadata(("127.0.0.1", 1), "2026-08-04T00:00:00Z")
    with RawRecordingWriter(out, metadata) as writer:
        writer.append(RawRecord(1, 0, bytes(992)))
    with (out / "records.bin").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="truncated"):
        list(iter_raw_records(out))
