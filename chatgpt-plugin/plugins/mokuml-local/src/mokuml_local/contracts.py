"""Versioned task, catalog and explanation contracts. Inputs never contain records."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Expression(Contract):
    kind: Literal["column", "literal", "operation", "aggregate"]
    column_id: str | None = None
    value: str | int | float | bool | None = None
    operator: (
        Literal[
            "add",
            "subtract",
            "multiply",
            "divide",
            "equal",
            "not_equal",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "and",
            "or",
            "is_null",
            "is_not_null",
            "coalesce",
            "case",
            "abs",
        ]
        | None
    ) = None
    function: Literal["sum", "mean", "min", "max", "count", "count_distinct"] | None = (
        None
    )
    args: list[Expression] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def shape(self):
        if self.kind == "column":
            valid = (
                bool(self.column_id)
                and not self.args
                and self.operator is None
                and self.function is None
                and self.value is None
            )
        elif self.kind == "literal":
            valid = (
                self.column_id is None
                and not self.args
                and self.operator is None
                and self.function is None
            )
        elif self.kind == "operation":
            valid = (
                self.operator is not None
                and self.column_id is None
                and self.function is None
                and self.value is None
            )
        else:
            valid = (
                self.function is not None
                and self.column_id is None
                and self.operator is None
                and self.value is None
                and len(self.args) <= 1
            )
        if not valid:
            raise ValueError("Invalid expression shape")
        return self


class NamedExpression(Contract):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    expression: Expression
    availability: Literal["at_prediction", "historical"] = "at_prediction"


class Join(Contract):
    table_id: str
    kind: Literal["inner", "left"] = "left"
    column_pairs: list[tuple[str, str]] = Field(min_length=1, max_length=10)


class RelationalQuery(Contract):
    table_id: str
    joins: list[Join] = Field(default_factory=list, max_length=12)
    filters: list[Expression] = Field(default_factory=list, max_length=20)
    group_by_column_ids: list[str] = Field(default_factory=list, max_length=20)


class FutureWindow(Contract):
    """Historical anchors and a scoring anchor use exactly the same recipes."""

    event_time_column_id: str
    entity_column_id: str | None = None
    start: datetime
    observed_through: datetime
    unit: Literal["day", "week", "month"] = "month"
    horizon: int = Field(default=1, ge=1, le=120)
    alignment: Literal["rolling", "calendar_next"] = "rolling"
    lookback_periods: list[int] = Field(
        default_factory=lambda: [1, 3], min_length=1, max_length=8
    )

    @model_validator(mode="after")
    def dates(self):
        if self.start >= self.observed_through or any(
            n < 1 or n > 120 for n in self.lookback_periods
        ):
            raise ValueError("Invalid observation window")
        if len(set(self.lookback_periods)) != len(self.lookback_periods):
            raise ValueError("Duplicate lookbacks")
        return self


class PredictionTask(Contract):
    question: str = Field(min_length=1, max_length=2000)
    task: Literal["classification", "regression"]
    query: RelationalQuery
    target: NamedExpression
    features: list[NamedExpression] = Field(min_length=1, max_length=200)
    identifier_column_ids: list[str] = Field(default_factory=list, max_length=10)
    group_column_id: str | None = None
    time_column_id: str | None = None
    future: FutureWindow | None = None
    units: str | None = Field(default=None, max_length=80)
    scoring_table_id: str | None = None
    scoring_column_map: dict[str, str] = Field(default_factory=dict)
    # Owner-defined dataset semantics, including post-outcome columns, must be explicit.
    unavailable_column_ids: list[str] = Field(default_factory=list, max_length=2000)
    output: Literal["individual", "summary"] = "individual"


class CatalogColumn(Contract):
    id: str
    name: str
    type: Literal["number", "boolean", "datetime", "string"]
    database_type: str
    nullable: bool
    primary_key: bool = False
    description: str | None = None


class Relationship(Contract):
    column_ids: list[str]
    referenced_table_id: str
    referenced_column_ids: list[str]


class Constraint(Contract):
    kind: Literal["primary_key", "unique", "check", "index"]
    column_ids: list[str]
    unique: bool = False


class CatalogTable(Contract):
    id: str
    name: str
    schema_name: str | None = None
    kind: Literal["table", "view", "file"]
    purpose: Literal["history", "scoring"] = "history"
    row_count: int | None = Field(default=None, ge=0)
    row_count_estimated: bool = True
    columns: list[CatalogColumn]
    description: str | None = None
    constraints: list[Constraint] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class SourceCatalog(Contract):
    id: str
    revision: str
    kind: Literal["files", "postgresql", "mysql", "sqlite"]
    table_count: int
    tables: list[CatalogTable]
    next_cursor: int | None = None


class ColumnProfile(Contract):
    column_id: str
    sampled_rows: int
    estimated: bool
    null_fraction: float = Field(ge=0, le=1)
    distinct_count: int = Field(ge=0)
    details_suppressed: bool
    minimum: float | str | None = None
    maximum: float | str | None = None
    mean: float | None = None
    quantiles: list[float] | None = None


class DataProfile(Contract):
    source_id: str
    revision: str
    table_id: str
    profiles: list[ColumnProfile]
    next_cursor: int | None = None


class PreparedPredictionPlan(Contract):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    id: str
    source_id: str
    source_revision: str
    task: Literal["classification", "regression"]
    sql: str
    scoring_sql: str | None = None
    target: str
    feature_names: list[str]
    identifier_names: list[str]
    split: Literal["stratified", "random", "grouped", "temporal"]
    units: str | None = None
    horizon: str | None = None
    findings: list[str]


class CandidateResult(Contract):
    model: Literal["tabicl", "tabpfn", "kumo"]
    status: Literal["selected", "evaluated", "excluded", "failed"]
    selection_score: float | None = None
    reason: (
        Literal[
            "models_not_prepared",
            "checkpoint_limits",
            "local_candidate_failed",
            "same_model_cpu_retry",
        ]
        | None
    ) = None


class ModelProvenance(Contract):
    engine: Literal["tabicl", "tabpfn", "kumo"]
    version: str
    checkpoint_revision: str
    checkpoint_name: str | None = None
    checkpoint_version: str | None = None
    checkpoint_sha256: str | None = None
    encoder: str
    device: Literal["cpu", "mps"]


class PredictionOutput(Contract):
    case_id: str
    prediction: float | str | bool
    probabilities: dict[str, float] | None = None
    lower: float | None = None
    upper: float | None = None
    interval_level: float | None = None


class FeatureInfluence(Contract):
    feature: str
    effect: float
    method: Literal["grouped_permutation", "reference_sensitivity"]
    interpretation: Literal["association", "sensitivity"]
    case_id: str | None = None


class PredictionSummary(Contract):
    count: int
    mean: float | None = None
    total: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    class_counts: dict[str, int] | None = None


class AugmentedPredictionResult(Contract):
    task: Literal["classification", "regression"]
    result_kind: Literal["predictions", "evaluation"]
    target: str
    units: str | None = None
    horizon: str | None = None
    as_of: str | None = None
    prediction_window_start: str | None = None
    prediction_window_end: str | None = None
    model: ModelProvenance
    selection_metric: Literal["log_loss", "mae"]
    candidates: list[CandidateResult]
    split: Literal["stratified", "random", "grouped", "temporal"]
    training_rows: int
    calibration_rows: int
    test_rows: int
    prediction_rows: int
    metrics: dict[
        Literal[
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "brier",
            "roc_auc",
            "mae",
            "rmse",
            "r2",
            "interval_coverage",
        ],
        float,
    ]
    baseline_metrics: dict[
        Literal[
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "brier",
            "roc_auc",
            "mae",
            "rmse",
            "r2",
            "interval_coverage",
        ],
        float,
    ]
    probability_calibration: Literal["uncalibrated", "not_applicable"]
    interval_method: Literal["split_conformal_90", "unavailable"]
    predictions: list[PredictionOutput] = Field(default_factory=list, max_length=20)
    summary: PredictionSummary
    feature_influences: list[FeatureInfluence] = Field(
        default_factory=list, max_length=110
    )
    data_quality: list[str]
    limitations: list[str]
    export_available: bool = True
