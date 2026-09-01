"""Fallback source timeline reconstruction for timestamp-free ZeroLab UDP."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class TimelineStats:
    redistributed_packets: int = 0
    maximum_adjustment_ns: int = 0


class BurstTimelineReconstructor:
    def __init__(self, *, rate_hz: float) -> None:
        if (
            isinstance(rate_hz, bool)
            or not isinstance(rate_hz, Real)
            or not math.isfinite(float(rate_hz))
            or float(rate_hz) <= 0.0
        ):
            raise ValueError("rate_hz must be finite and positive")
        self._period_ns = int(1_000_000_000 / float(rate_hz))
        self._last_sample_timestamp_ns: int | None = None
        self._stats = TimelineStats()

    @property
    def stats(self) -> TimelineStats:
        return self._stats

    def reconstruct_batch(
        self, receive_timestamps_ns: Iterable[int]
    ) -> tuple[int, ...]:
        actual = tuple(receive_timestamps_ns)
        if not actual:
            return ()
        for timestamp_ns in actual:
            if isinstance(timestamp_ns, bool) or not isinstance(
                timestamp_ns, Integral
            ):
                raise ValueError("receive timestamp must be an integer")
        actual = tuple(int(timestamp_ns) for timestamp_ns in actual)
        if any(right < left for left, right in zip(actual, actual[1:])):
            raise ValueError("receive timestamps must be monotonic")

        reconstructed = list(actual)
        for index in range(len(reconstructed) - 2, -1, -1):
            reconstructed[index] = min(
                reconstructed[index],
                reconstructed[index + 1] - self._period_ns,
            )

        if (
            self._last_sample_timestamp_ns is not None
            and reconstructed[0] <= self._last_sample_timestamp_ns
        ):
            available_ns = actual[-1] - self._last_sample_timestamp_ns
            if available_ns < len(actual):
                raise ValueError(
                    "batch has insufficient causal timestamp room"
                )
            compressed_period_ns = available_ns // len(actual)
            reconstructed = [
                self._last_sample_timestamp_ns
                + compressed_period_ns * (index + 1)
                for index in range(len(actual))
            ]

        redistributed = sum(
            reconstructed_ns != actual_ns
            for reconstructed_ns, actual_ns in zip(reconstructed, actual)
        )
        maximum_adjustment_ns = max(
            (
                abs(actual_ns - reconstructed_ns)
                for reconstructed_ns, actual_ns in zip(
                    reconstructed, actual
                )
            ),
            default=0,
        )
        self._stats = replace(
            self._stats,
            redistributed_packets=(
                self._stats.redistributed_packets + redistributed
            ),
            maximum_adjustment_ns=max(
                self._stats.maximum_adjustment_ns,
                maximum_adjustment_ns,
            ),
        )
        self._last_sample_timestamp_ns = reconstructed[-1]
        return tuple(reconstructed)


__all__ = ["BurstTimelineReconstructor", "TimelineStats"]
