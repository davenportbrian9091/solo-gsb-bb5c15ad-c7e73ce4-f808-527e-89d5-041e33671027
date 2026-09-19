"""Unit tests for the separation domain logic."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.separation import (
    DEFAULT_MATRIX,
    ClosureWindow,
    Flight,
    build_matrix,
    find_closure_conflicts,
    find_violations,
    required_gap,
    sort_flights,
    sort_windows,
    trial_reschedule,
)

UTC = timezone.utc


def dt(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


def make(no: str, t: int, wake: str) -> Flight:
    return Flight(flight_no=no, scheduled_off=dt(t), wake=wake)


def window(start: int, end: int) -> ClosureWindow:
    return ClosureWindow(start=dt(start), end=dt(end))


def test_stable_sort_by_time_then_flight_no():
    flights = [
        make("BBB", 200, "M"),
        make("AAA", 100, "H"),
        make("CCC", 100, "L"),  # same time as AAA -> flight_no tiebreak
    ]
    assert [f.flight_no for f in sort_flights(flights)] == ["AAA", "CCC", "BBB"]
    # input order unchanged (immutable objects, new list)
    assert flights[0].flight_no == "BBB"


def test_default_matrix_contents():
    matrix = build_matrix(None)
    assert matrix[("J", "M")] == DEFAULT_MATRIX["J"]["M"] == 180
    assert required_gap(matrix, "H", "M") == 120


def test_custom_matrix_unknown_pair_is_zero():
    matrix = build_matrix({("H", "M"): 90})
    assert required_gap(matrix, "H", "M") == 90
    assert required_gap(matrix, "J", "L") == 0


def test_find_violations_flags_only_adjacent_breaches():
    # H leader (100-> next M needs 120s); M->M needs 60s
    flights = [
        make("CA101", 0, "H"),
        make("CA102", 60, "M"),    # gap 60 < 120 -> violation
        make("CA103", 130, "M"),   # gap 70 >= 60 -> ok
    ]
    violations = find_violations(flights, build_matrix(None))
    assert len(violations) == 1
    v = violations[0]
    assert (v.leader_no, v.follower_no) == ("CA101", "CA102")
    assert v.required_seconds == 120
    assert v.actual_seconds == 60
    assert v.shortfall_seconds == 60
    assert (v.leader_position, v.follower_position) == (1, 2)


def test_no_violation_when_exact_gap_met():
    flights = [make("CA101", 0, "H"), make("CA102", 120, "M")]
    assert find_violations(flights, build_matrix(None)) == []


def test_trial_cascades_delays_without_mutating_originals():
    # J(0) -> M needs 180; M -> M needs 60
    flights = [
        make("CA1", 0, "J"),
        make("CA2", 60, "M"),    # pushed to 180 (triggered by CA1), delay 120
        make("CA3", 200, "M"),   # must be >= 240 (triggered by CA2), delay 40
    ]
    items = trial_reschedule(flights, build_matrix(None))
    assert [i.flight_no for i in items] == ["CA1", "CA2", "CA3"]
    assert items[0].delay_seconds == 0
    assert items[0].triggered_by is None
    assert items[1].adjusted_time == dt(180)
    assert items[1].delay_seconds == 120
    assert items[1].triggered_by == "CA1"
    assert items[2].adjusted_time == dt(240)
    assert items[2].delay_seconds == 40
    assert items[2].triggered_by == "CA2"

    # original schedules are untouched
    assert flights[1].scheduled_off == dt(60)
    assert flights[2].scheduled_off == dt(200)


def test_trial_never_moves_flight_earlier():
    flights = [make("CA1", 1000, "M"), make("CA2", 5000, "M")]
    items = trial_reschedule(flights, build_matrix(None))
    assert items[1].delay_seconds == 0
    assert items[1].adjusted_time == dt(5000)


def test_trial_with_custom_matrix():
    matrix = build_matrix({("H", "H"): 300})
    flights = [make("CA1", 0, "H"), make("CA2", 100, "H")]
    items = trial_reschedule(flights, matrix)
    assert items[1].adjusted_time == dt(300)
    assert items[1].delay_seconds == 200
    assert items[1].triggered_by == "CA1"


def test_single_flight_has_no_violation_and_no_delay():
    flights = [make("SOLO", 42, "J")]
    assert find_violations(flights, build_matrix(None)) == []
    (item,) = trial_reschedule(flights, build_matrix(None))
    assert item.delay_seconds == 0 and item.triggered_by is None


# ---------- closure windows ----------


def test_sort_windows_orders_by_start():
    wins = sort_windows([window(300, 400), window(0, 100), window(100, 200)])
    assert [(w.s, w.e) for w in wins] == [(0, 100), (100, 200), (300, 400)]


def test_closure_conflict_window_is_half_open():
    flights = [
        make("AT_START", 100, "M"),  # start is inclusive -> conflict
        make("AT_END", 200, "M"),    # end is exclusive -> no conflict
        make("BEFORE", 99, "M"),
        make("AFTER", 201, "M"),
    ]
    conflicts = find_closure_conflicts(flights, [window(100, 200)])
    assert [c.flight_no for c in conflicts] == ["AT_START"]
    c = conflicts[0]
    assert c.position == 2  # sorted: BEFORE(99), AT_START(100), ...
    assert c.window_index == 1
    assert c.window_start == dt(100) and c.window_end == dt(200)


def test_closure_conflicts_sorted_positions_and_window_index():
    flights = [
        make("LATE", 500, "H"),   # unsorted input on purpose
        make("EARLY", 50, "M"),
        make("MID", 350, "L"),
    ]
    wins = [window(300, 400), window(0, 100)]  # unsorted input on purpose
    conflicts = find_closure_conflicts(flights, wins)
    assert [(c.flight_no, c.position, c.window_index) for c in conflicts] == [
        ("EARLY", 1, 1),   # window [0,100) is index 1 after sorting
        ("MID", 2, 2),     # window [300,400) is index 2
    ]


def test_trial_first_flight_pushed_out_of_window():
    flights = [make("CA1", 30, "M"), make("CA2", 200, "M")]
    items = trial_reschedule(flights, build_matrix(None), [window(0, 90)])
    assert items[0].adjusted_time == dt(90)
    assert items[0].delay_seconds == 60
    assert items[0].triggered_by is None
    assert items[0].closure_hit is True
    assert items[0].closure_window == window(0, 90)
    # follower unaffected: 200 >= 90 + 60
    assert items[1].delay_seconds == 0
    assert items[1].closure_hit is False
    assert items[1].closure_window is None


def test_trial_wake_push_into_window_iterates_to_window_end():
    # J->M requires 180s: wake pushes CA2 from 60 to 180, which lands inside
    # the [150, 210) window, so it must move on to 210.
    flights = [make("CA1", 0, "J"), make("CA2", 60, "M")]
    items = trial_reschedule(flights, build_matrix(None), [window(150, 210)])
    assert items[1].adjusted_time == dt(210)
    assert items[1].delay_seconds == 150
    assert items[1].triggered_by == "CA1"
    assert items[1].closure_hit is True
    assert items[1].closure_window == window(150, 210)


def test_trial_iterates_through_touching_windows():
    # window end == next window start: landing on the boundary is still closed
    flights = [make("CA1", 50, "M")]
    items = trial_reschedule(
        flights, build_matrix(None), [window(0, 100), window(100, 160)]
    )
    assert items[0].adjusted_time == dt(160)
    assert items[0].closure_hit is True
    assert items[0].closure_window == window(100, 160)


def test_trial_closure_delay_cascades_to_successors():
    # CA2 is closure-pushed to 210; CA3 must cascade to 210 + 60 (M->M)
    flights = [
        make("CA1", 0, "J"),
        make("CA2", 60, "M"),
        make("CA3", 220, "M"),
    ]
    items = trial_reschedule(flights, build_matrix(None), [window(150, 210)])
    assert items[1].adjusted_time == dt(210)
    assert items[2].adjusted_time == dt(270)
    assert items[2].delay_seconds == 50
    assert items[2].triggered_by == "CA2"
    assert items[2].closure_hit is False


def test_trial_wake_push_past_window_is_not_closure_hit():
    # wake earliest (300) already clears the [100, 200) window entirely
    flights = [make("CA1", 0, "H"), make("CA2", 150, "H")]
    matrix = build_matrix({("H", "H"): 300})
    items = trial_reschedule(flights, matrix, [window(100, 200)])
    assert items[1].adjusted_time == dt(300)
    assert items[1].triggered_by == "CA1"
    assert items[1].closure_hit is False
    assert items[1].closure_window is None


def test_trial_without_windows_keeps_legacy_flags():
    flights = [make("CA1", 0, "M"), make("CA2", 10, "M")]
    items = trial_reschedule(flights, build_matrix(None))
    assert all(i.closure_hit is False and i.closure_window is None for i in items)
