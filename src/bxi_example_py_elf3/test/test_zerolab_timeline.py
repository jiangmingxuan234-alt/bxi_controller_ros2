import pytest

from zerolab.timeline import BurstTimelineReconstructor


def test_clustered_arrivals_are_spread_back_from_newest_arrival():
    timeline = BurstTimelineReconstructor(rate_hz=50.0)

    reconstructed = timeline.reconstruct_batch(
        (1_000_000_000, 1_000_100_000, 1_000_200_000)
    )

    assert reconstructed == (
        960_200_000,
        980_200_000,
        1_000_200_000,
    )
    assert timeline.stats.redistributed_packets == 2
    assert timeline.stats.maximum_adjustment_ns == 39_800_000


def test_regular_arrivals_are_not_changed():
    timeline = BurstTimelineReconstructor(rate_hz=50.0)
    actual = (1_000_000_000, 1_020_000_000, 1_040_000_000)

    assert timeline.reconstruct_batch(actual) == actual
    assert timeline.stats.redistributed_packets == 0
    assert timeline.stats.maximum_adjustment_ns == 0


def test_batch_that_cannot_fit_after_previous_sample_stays_causal_and_ordered():
    timeline = BurstTimelineReconstructor(rate_hz=50.0)
    assert timeline.reconstruct_batch((1_000_000_000,)) == (1_000_000_000,)

    reconstructed = timeline.reconstruct_batch(
        (1_000_100_000, 1_000_100_000, 1_000_200_000)
    )

    assert reconstructed == (1_000_066_666, 1_000_133_332, 1_000_199_998)
    assert all(
        left < right for left, right in zip(reconstructed, reconstructed[1:])
    )
    assert reconstructed[-1] <= 1_000_200_000


def test_adjustment_stat_counts_forward_compression_as_absolute_distance():
    timeline = BurstTimelineReconstructor(rate_hz=50.0)
    timeline.reconstruct_batch((1_000_000_000,))

    timeline.reconstruct_batch((1_000_000_001, 1_000_000_100))

    assert timeline.stats.maximum_adjustment_ns == 49


def test_empty_batch_does_not_change_timeline_state():
    timeline = BurstTimelineReconstructor(rate_hz=50.0)

    assert timeline.reconstruct_batch(()) == ()
    assert timeline.reconstruct_batch((500_000_000,)) == (500_000_000,)


def test_batch_with_no_causal_timestamp_room_is_rejected():
    timeline = BurstTimelineReconstructor(rate_hz=50.0)
    timeline.reconstruct_batch((1_000_000_000,))

    with pytest.raises(ValueError, match="causal timestamp room"):
        timeline.reconstruct_batch((1_000_000_000, 1_000_000_000))


@pytest.mark.parametrize("rate_hz", (0.0, -1.0, float("nan"), True))
def test_invalid_source_rate_is_rejected(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        BurstTimelineReconstructor(rate_hz=rate_hz)


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        ((1_000_000_000, 999_999_999), "monotonic"),
        ((1_000_000_000, True), "integer"),
        ((1_000_000_000, 1.5), "integer"),
    ],
)
def test_invalid_receive_timestamps_are_rejected(timestamps, message):
    timeline = BurstTimelineReconstructor(rate_hz=50.0)

    with pytest.raises(ValueError, match=message):
        timeline.reconstruct_batch(timestamps)
