"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .separation import DEFAULT_MATRIX, Wake


class FlightIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flight_no: str = Field(..., min_length=1, examples=["CA1234"])
    scheduled_off: datetime = Field(
        ..., description="计划离场时刻，ISO 8601，必须带时区，例如 2026-09-18T08:00:00+08:00"
    )
    wake: Wake = Field(..., description="尾流等级: L 轻型 / M 中型 / H 重型 / J 超重型")

    @field_validator("flight_no")
    @classmethod
    def _strip_flight_no(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("flight_no 不能为空")
        return v

    @field_validator("scheduled_off")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("scheduled_off 必须携带时区信息（ISO 8601 offset 或 Z）")
        return v


class ClosureWindowIn(BaseModel):
    """A temporary runway closure window ``[start, end)`` (end excluded)."""

    model_config = ConfigDict(extra="forbid")

    start: datetime = Field(
        ..., description="关闭开始时刻，ISO 8601，必须带时区，例如 2026-09-18T08:10:00+08:00"
    )
    end: datetime = Field(
        ..., description="关闭结束时刻（不含），ISO 8601，必须带时区"
    )

    @field_validator("start", "end")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("关闭窗口时刻必须携带时区信息（ISO 8601 offset 或 Z）")
        return v

    @model_validator(mode="after")
    def _end_after_start(self) -> ClosureWindowIn:
        if self.end <= self.start:
            raise ValueError("关闭窗口结束时刻必须晚于开始时刻（[start, end) 左闭右开）")
        return self


class CapacityIn(BaseModel):
    """Optional per-runway rolling departure capacity: at most
    ``max_departures`` departures in any half-open window
    ``[t, t + window_seconds)``."""

    model_config = ConfigDict(extra="forbid")

    window_seconds: int = Field(
        ..., ge=1, description="滚动窗口长度（秒），正整数", examples=[600]
    )
    max_departures: int = Field(
        ..., ge=1, description="单个滚动窗口内允许的最大离场架次，正整数", examples=[5]
    )


class RunwayIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runway: str = Field(..., min_length=1, examples=["36L"])
    flights: list[FlightIn] = Field(..., min_length=1)
    closures: list[ClosureWindowIn] = Field(
        default_factory=list,
        description="本跑道的临时关闭窗口列表，[start, end) 左闭右开，彼此不得重叠。",
    )
    capacity: CapacityIn | None = Field(
        default=None,
        description="本跑道的滚动容量约束：任意 [t, t+window_seconds) 半开区间内"
        "离场数不得超过 max_departures；缺省表示不限制。",
    )

    @field_validator("runway")
    @classmethod
    def _strip_runway(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("runway 不能为空")
        return v

    @model_validator(mode="after")
    def _validate_runway(self) -> RunwayIn:
        seen: set[str] = set()
        for f in self.flights:
            key = f.flight_no
            if key in seen:
                raise ValueError(f"跑道 {self.runway} 内航班号重复: {key}")
            seen.add(key)
        # closures are half-open [start, end): touching is allowed, overlap is not
        ordered = sorted(self.closures, key=lambda w: (w.start, w.end))
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.start < prev.end:
                raise ValueError(
                    f"跑道 {self.runway} 的关闭窗口重叠: "
                    f"[{prev.start.isoformat()}, {prev.end.isoformat()}) 与 "
                    f"[{nxt.start.isoformat()}, {nxt.end.isoformat()})"
                )
        return self


class MatrixEntryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leader: Wake
    follower: Wake
    seconds: int = Field(..., ge=0, description="前机—后机最小雷达间隔（秒）")


class WakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runways: list[RunwayIn] = Field(..., min_length=1)
    matrix: list[MatrixEntryIn] | None = Field(
        default=None,
        description="前机—后机最小间隔矩阵；缺省使用内置 ICAO 风格默认矩阵。"
        "提供时仅列出的组合生效，未列出的组合视为无最小间隔(0 秒)。",
    )

    @model_validator(mode="after")
    def _validate_request(self) -> WakeRequest:
        runway_names: set[str] = set()
        global_flights: set[str] = set()
        for rw in self.runways:
            if rw.runway in runway_names:
                raise ValueError(f"跑道标识重复: {rw.runway}")
            runway_names.add(rw.runway)
            for f in rw.flights:
                if f.flight_no in global_flights:
                    raise ValueError(f"航班号在整个请求中必须唯一: {f.flight_no}")
                global_flights.add(f.flight_no)
        if self.matrix is not None:
            pairs: set[tuple[str, str]] = set()
            for entry in self.matrix:
                pair = (entry.leader.value, entry.follower.value)
                if pair in pairs:
                    raise ValueError(
                        f"间隔矩阵组合重复: leader={pair[0]} follower={pair[1]}"
                    )
                pairs.add(pair)
        return self


# ---------- response models ----------


class ClosureWindowOut(BaseModel):
    start: datetime
    end: datetime


class CapacityOut(BaseModel):
    window_seconds: int
    max_departures: int


class CapacityViolationOut(BaseModel):
    flight_no: str
    wake: Wake
    scheduled_off: datetime
    position: int
    window_start: datetime
    window_end: datetime
    departures_in_window: int
    earliest_compliant_off: datetime


class ClosureConflictOut(BaseModel):
    flight_no: str
    wake: Wake
    scheduled_off: datetime
    position: int
    window_index: int
    window_start: datetime
    window_end: datetime


class ViolationOut(BaseModel):
    leader: str
    follower: str
    leader_wake: Wake
    follower_wake: Wake
    leader_scheduled_off: datetime
    follower_scheduled_off: datetime
    required_seconds: int
    actual_seconds: int
    shortfall_seconds: int
    leader_position: int
    follower_position: int


class RunwayReportOut(BaseModel):
    runway: str
    order: list[str]
    total: int
    closures: list[ClosureWindowOut]
    capacity: CapacityOut | None
    violation_count: int
    violations: list[ViolationOut]
    closure_conflict_count: int
    closure_conflicts: list[ClosureConflictOut]
    capacity_violation_count: int
    capacity_violations: list[CapacityViolationOut]


class ValidateResponse(BaseModel):
    batch_id: str
    mode: str = "validate"
    matrix: dict[str, dict[str, int]]
    total_runways: int
    total_flights: int
    total_violations: int
    total_closure_conflicts: int
    total_capacity_violations: int
    runways: list[RunwayReportOut]


class TrialItemOut(BaseModel):
    position: int
    flight_no: str
    wake: Wake
    scheduled_off: datetime
    adjusted_off: datetime
    delay_seconds: int
    triggered_by: str | None
    closure_hit: bool
    closure_window: ClosureWindowOut | None
    capacity_hit: bool
    capacity_window: ClosureWindowOut | None


class RunwayTrialOut(BaseModel):
    runway: str
    order: list[str]
    total: int
    closures: list[ClosureWindowOut]
    capacity: CapacityOut | None
    delayed_count: int
    total_delay_seconds: int
    max_delay_seconds: int
    items: list[TrialItemOut]


class TrialResponse(BaseModel):
    batch_id: str
    mode: str = "trial"
    matrix: dict[str, dict[str, int]]
    total_runways: int
    total_flights: int
    total_delayed: int
    total_delay_seconds: int
    runways: list[RunwayTrialOut]


class MatrixOut(BaseModel):
    matrix: dict[str, dict[str, int]]


class HealthOut(BaseModel):
    status: str


class BatchResponse(BaseModel):
    """Stored batch (either mode) fetched by id; summary fields are mode-specific."""

    batch_id: str
    mode: str
    created_at: datetime
    matrix: dict[str, dict[str, int]]
    total_runways: int
    total_flights: int
    total_violations: int | None = None
    total_closure_conflicts: int | None = None
    total_capacity_violations: int | None = None
    total_delayed: int | None = None
    total_delay_seconds: int | None = None
    runways: list[dict]


def matrix_to_nested(flat: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    """Convert the internal flat matrix to a leader->follower nested dict."""
    nested: dict[str, dict[str, int]] = {}
    for (leader, follower), seconds in flat.items():
        nested.setdefault(leader, {})[follower] = seconds
    return nested


def default_matrix_nested() -> dict[str, dict[str, int]]:
    return {k: dict(v) for k, v in DEFAULT_MATRIX.items()}
