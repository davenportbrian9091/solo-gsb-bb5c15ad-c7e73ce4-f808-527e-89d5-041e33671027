"""Core wake-separation domain logic.

The service operates on *time in seconds* so the separation matrix (defined
in seconds) can be applied with plain integer arithmetic, no float rounding.

Wake turbulence categories follow the four-class scheme:

``L``  LIGHT    – light aircraft (e.g. PA28, C152)
``M``  MEDIUM   – medium aircraft (most airliners below 136 t)
``H``  HEAVY    – heavy aircraft (>= 136 t MTOW)
``J``  SUPER    – super aircraft (e.g. A380)

Only *adjacent* pairs on the same runway are considered, matching the
operational practice for departures on a single runway: the leader of a pair
is the flight immediately ahead in the departure order.

A runway may additionally declare a *rolling capacity*: at most
``max_departures`` take-offs inside any half-open window
``[t, t + window_seconds)``. With flights in departure order, flight ``j``
breaks the capacity exactly when it departs less than ``window_seconds``
after its ``max_departures``-th predecessor — the window starting at that
predecessor then holds at least ``max_departures + 1`` departures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .timeutils import to_epoch_seconds


class Wake(str, Enum):
    LIGHT = "L"
    MEDIUM = "M"
    HEAVY = "H"
    SUPER = "J"


# Default leader -> follower minimum separation in seconds.
# ICAO-style departure values (recat categories map loosely to this matrix).
#   LIGHT leader  : 100 s behind light, otherwise 60 s
#   MEDIUM leader : 100 s behind light, otherwise 60 s
#   HEAVY leader  : 120 s behind medium, 100 s behind heavy, 100 s light
#   SUPER leader  : 180 s behind heavy/medium, 180 s light, 120 s super
# Values are intentionally simple defaults; a request may override the whole
# matrix, in which case any omitted leader/follower pair defaults to zero
# (i.e. no required radar minimum for that pair).
DEFAULT_MATRIX: dict[str, dict[str, int]] = {
    "L": {"L": 100, "M": 60, "H": 60, "J": 60},
    "M": {"L": 100, "M": 60, "H": 60, "J": 60},
    "H": {"L": 100, "M": 120, "H": 100, "J": 60},
    "J": {"L": 180, "M": 180, "H": 180, "J": 120},
}


@dataclass(frozen=True, slots=True)
class Flight:
    """A flight plan item after timezone normalisation."""

    flight_no: str
    scheduled_off: datetime  # always tz-aware, converted to UTC
    wake: str

    @property
    def t(self) -> int:
        return to_epoch_seconds(self.scheduled_off)

    def sort_key(self) -> tuple[int, str]:
        """Stable ordering key: scheduled time, then flight number."""
        return self.t, self.flight_no


@dataclass(frozen=True, slots=True)
class ClosureWindow:
    """A temporary runway closure ``[start, end)`` after UTC normalisation."""

    start: datetime  # always tz-aware, converted to UTC
    end: datetime  # always tz-aware, converted to UTC

    @property
    def s(self) -> int:
        return to_epoch_seconds(self.start)

    @property
    def e(self) -> int:
        return to_epoch_seconds(self.end)

    def contains(self, t: int) -> bool:
        return self.s <= t < self.e


@dataclass(frozen=True, slots=True)
class ClosureConflict:
    """A flight whose scheduled off-time falls inside a closure window."""

    flight_no: str
    wake: str
    scheduled_off: datetime
    position: int
    window_index: int  # 1-based index into the runway's sorted closure list
    window_start: datetime
    window_end: datetime


@dataclass(frozen=True, slots=True)
class Capacity:
    """Per-runway rolling departure capacity: at most ``max_departures``
    departures inside any half-open window ``[t, t + window_seconds)``."""

    window_seconds: int
    max_departures: int


@dataclass(frozen=True, slots=True)
class CapacityWindow:
    """The rolling window ``[start, end)`` that triggered a capacity finding."""

    start: datetime  # always tz-aware, converted to UTC
    end: datetime  # always tz-aware, converted to UTC


@dataclass(frozen=True, slots=True)
class CapacityViolation:
    """A flight whose scheduled off-time makes it the (``max_departures`` + 1)-th
    departure inside the rolling window starting at its ``max_departures``-th
    predecessor."""

    flight_no: str
    wake: str
    scheduled_off: datetime
    position: int
    window_start: datetime
    window_end: datetime
    departures_in_window: int
    earliest_compliant_off: datetime


@dataclass(frozen=True, slots=True)
class Violation:
    leader_no: str
    follower_no: str
    leader_wake: str
    follower_wake: str
    leader_time: datetime
    follower_time: datetime
    required_seconds: int
    actual_seconds: int
    shortfall_seconds: int
    leader_position: int
    follower_position: int


@dataclass(slots=True)
class TrialItem:
    flight_no: str
    wake: str
    original_time: datetime
    adjusted_time: datetime
    delay_seconds: int
    position: int
    triggered_by: str | None  # flight number of the leader that forced the pushback
    closure_hit: bool  # a closure window forced (further) delay
    closure_window: ClosureWindow | None  # the last window that pushed this flight
    capacity_hit: bool = False  # the rolling capacity forced (further) delay
    capacity_window: CapacityWindow | None = None  # the rolling window that pushed this flight


def build_matrix(
    entries: dict[tuple[str, str], int] | None,
) -> dict[tuple[str, str], int]:
    """Return a ``(leader, follower) -> seconds`` lookup.

    ``entries`` is the client-supplied override. When omitted/empty the built-in
    :data:`DEFAULT_MATRIX` is used; otherwise every unspecified pair is zero
    (no minimum for that pair).
    """
    if not entries:
        return {
            (leader, follower): seconds
            for leader, followers in DEFAULT_MATRIX.items()
            for follower, seconds in followers.items()
        }
    return dict(entries)


def required_gap(
    matrix: dict[tuple[str, str], int], leader_wake: str, follower_wake: str
) -> int:
    return matrix.get((leader_wake, follower_wake), 0)


def sort_flights(flights: list[Flight]) -> list[Flight]:
    """Stable sort by (scheduled time, flight number)."""
    return sorted(flights, key=Flight.sort_key)


def sort_windows(windows: list[ClosureWindow]) -> list[ClosureWindow]:
    """Sort closure windows by (start, end); input is already non-overlapping."""
    return sorted(windows, key=lambda w: (w.s, w.e))


def find_closure_conflicts(
    flights: list[Flight], windows: list[ClosureWindow]
) -> list[ClosureConflict]:
    """List flights whose scheduled off-time falls inside a closure window.

    Windows are half-open ``[start, end)`` and non-overlapping, so a flight
    hits at most one window. ``window_index`` is 1-based over the sorted
    window list, matching the closures echoed back in the API responses.
    """
    ordered = sort_flights(flights)
    wins = sort_windows(windows)
    conflicts: list[ClosureConflict] = []
    for position, flight in enumerate(ordered, start=1):
        for index, window in enumerate(wins, start=1):
            if window.contains(flight.t):
                conflicts.append(
                    ClosureConflict(
                        flight_no=flight.flight_no,
                        wake=flight.wake,
                        scheduled_off=flight.scheduled_off,
                        position=position,
                        window_index=index,
                        window_start=window.start,
                        window_end=window.end,
                    )
                )
                break
    return conflicts


def find_violations(
    flights: list[Flight], matrix: dict[tuple[str, str], int]
) -> list[Violation]:
    """List violations between every adjacent pair of the (sorted) sequence."""
    ordered = sort_flights(flights)
    violations: list[Violation] = []
    for i in range(len(ordered) - 1):
        leader, follower = ordered[i], ordered[i + 1]
        required = required_gap(matrix, leader.wake, follower.wake)
        actual = follower.t - leader.t
        if actual < required:
            violations.append(
                Violation(
                    leader_no=leader.flight_no,
                    follower_no=follower.flight_no,
                    leader_wake=leader.wake,
                    follower_wake=follower.wake,
                    leader_time=leader.scheduled_off,
                    follower_time=follower.scheduled_off,
                    required_seconds=required,
                    actual_seconds=actual,
                    shortfall_seconds=required - actual,
                    leader_position=i + 1,
                    follower_position=i + 2,
                )
            )
    return violations


def find_capacity_violations(
    flights: list[Flight], capacity: Capacity | None
) -> list[CapacityViolation]:
    """List flights that exceed the rolling departure capacity.

    Flights are considered in stable ``(time, flight_no)`` order. Flight ``j``
    exceeds the capacity when it departs less than ``window_seconds`` after
    its ``max_departures``-th predecessor ``i``: the half-open window
    ``[t_i, t_i + window_seconds)`` then contains at least
    ``max_departures + 1`` departures. The reported earliest compliant instant
    is the end of that triggering window — the moment the oldest of the
    ``max_departures`` predecessors leaves the rolling window.
    """
    if capacity is None:
        return []
    ordered = sort_flights(flights)
    n = capacity.max_departures
    span = capacity.window_seconds
    violations: list[CapacityViolation] = []
    for j in range(n, len(ordered)):
        first = ordered[j - n]
        flight = ordered[j]
        if flight.t - first.t >= span:
            continue
        window_end_t = first.t + span
        count = sum(1 for f in ordered if first.t <= f.t < window_end_t)
        violations.append(
            CapacityViolation(
                flight_no=flight.flight_no,
                wake=flight.wake,
                scheduled_off=flight.scheduled_off,
                position=j + 1,
                window_start=first.scheduled_off,
                window_end=datetime.fromtimestamp(window_end_t, tz=timezone.utc),
                departures_in_window=count,
                earliest_compliant_off=datetime.fromtimestamp(
                    window_end_t, tz=timezone.utc
                ),
            )
        )
    return violations


def trial_reschedule(
    flights: list[Flight],
    matrix: dict[tuple[str, str], int],
    windows: list[ClosureWindow] | None = None,
    capacity: Capacity | None = None,
) -> list[TrialItem]:
    """Greedily push flights back to the earliest safe time, keeping order.

    Every flight is placed at the earliest instant that (a) is not earlier
    than its scheduled time, (b) keeps at least the matrix gap behind the
    (possibly already delayed) flight directly ahead of it, (c) satisfies the
    rolling capacity — at most ``max_departures`` departures in any
    ``[t, t + window_seconds)`` — against the already-placed predecessors,
    and (d) avoids every closure window. If the wake-separation or capacity
    push lands the flight inside a closure window, it is pushed further to
    that window's end — iterating because a window end may coincide with the
    next window's start. Delays cascade to successors. Original schedules are
    never rewritten; only :class:`TrialItem` proposals are returned.
    """
    ordered = sort_flights(flights)
    wins = sort_windows(windows or [])
    items: list[TrialItem] = []
    placed: list[int] = []  # adjusted off-times, in (non-decreasing) order
    prev: Flight | None = None
    prev_adjusted_t = 0
    for position, flight in enumerate(ordered, start=1):
        adjusted_t = flight.t
        triggered_by: str | None = None
        if prev is not None:
            required = required_gap(matrix, prev.wake, flight.wake)
            earliest = prev_adjusted_t + required
            if adjusted_t < earliest:
                adjusted_t = earliest
                triggered_by = prev.flight_no
        capacity_hit = False
        capacity_window: CapacityWindow | None = None
        if capacity is not None and len(placed) >= capacity.max_departures:
            # the max_departures-th predecessor is the binding one: departing
            # earlier than its window end would make this flight the
            # (max_departures + 1)-th departure inside that window
            ref_t = placed[-capacity.max_departures]
            earliest = ref_t + capacity.window_seconds
            if adjusted_t < earliest:
                adjusted_t = earliest
                capacity_hit = True
                capacity_window = CapacityWindow(
                    start=datetime.fromtimestamp(ref_t, tz=timezone.utc),
                    end=datetime.fromtimestamp(earliest, tz=timezone.utc),
                )
        closure_hit = False
        closure_window: ClosureWindow | None = None
        for window in wins:
            if adjusted_t < window.s:
                break  # in a gap before this (and every later) window
            if window.contains(adjusted_t):
                adjusted_t = window.e
                closure_hit = True
                closure_window = window
        items.append(
            TrialItem(
                flight_no=flight.flight_no,
                wake=flight.wake,
                original_time=flight.scheduled_off,
                adjusted_time=datetime.fromtimestamp(adjusted_t, tz=timezone.utc),
                delay_seconds=adjusted_t - flight.t,
                position=position,
                triggered_by=triggered_by,
                closure_hit=closure_hit,
                closure_window=closure_window,
                capacity_hit=capacity_hit,
                capacity_window=capacity_window,
            )
        )
        placed.append(adjusted_t)
        prev = flight
        prev_adjusted_t = adjusted_t
    return items
