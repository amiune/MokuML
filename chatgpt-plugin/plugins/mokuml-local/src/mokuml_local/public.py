"""Explicit allowlists for every byte returned by MCP; never serialize private objects."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import (
    AugmentedPredictionResult,
    CandidateResult,
    DataProfile,
    PreparedPredictionPlan,
    SourceCatalog,
)
from .errors import MESSAGES


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Column(PublicModel):
    id: str
    name: str
    type: Literal["number", "boolean", "datetime", "string"]
    nullable: bool
    primary_key: bool = False


class DatasetMetadata(PublicModel):
    id: str
    source_kind: Literal["csv", "xlsx", "files", "postgresql", "mysql", "sqlite"]
    row_count: int = Field(ge=0, le=100000)
    columns: list[Column]


class Metrics(PublicModel):
    accuracy: float | None = Field(default=None, ge=0, le=1)
    roc_auc: float | None = Field(default=None, ge=0, le=1)
    mae: float | None = Field(default=None, ge=0)
    rmse: float | None = Field(default=None, ge=0)
    r2: float | None = None
    log_loss: float | None = Field(default=None, ge=0)
    balanced_accuracy: float | None = Field(default=None, ge=0, le=1)
    brier: float | None = Field(default=None, ge=0)
    interval_coverage: float | None = Field(default=None, ge=0, le=1)


class RunSummary(PublicModel):
    id: str
    status: Literal["running", "completed", "failed", "cancelled"]
    stage: Literal[
        "preparing",
        "materializing",
        "embedding",
        "selecting",
        "evaluating",
        "explaining",
        "predicting",
        "finished",
    ]
    task: Literal["classification", "regression"]
    result_kind: Literal["predictions", "evaluation"] | None = None
    validation_rows: int | None = None
    prediction_rows: int | None = None
    model: str = "pending"
    model_version: str = "pending"
    encoder: Literal["Qwen3-0.6B-Base/16", "none"] | None = None
    device: Literal["cpu", "mps"] | None = None
    metrics: Metrics | None = None
    baseline_metrics: Metrics | None = None
    warnings: list[str] = Field(default_factory=list)
    result: AugmentedPredictionResult | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)


class ModelAvailability(PublicModel):
    tabicl: bool
    qwen: bool
    tabpfn: bool = False
    kumo: bool = False
    downloading: bool


class Reply(PublicModel):
    ok: bool
    code: str
    message: str
    dataset: DatasetMetadata | None = None
    run: RunSummary | None = None
    models: ModelAvailability | None = None
    catalog: SourceCatalog | None = None
    profile: DataProfile | None = None
    plan: PreparedPredictionPlan | None = None


def reply(code: str = "ready", *, ok: bool = True, **fields) -> dict:
    if code not in MESSAGES:
        code, ok = "internal_error", False
    return Reply(ok=ok, code=code, message=MESSAGES[code], **fields).model_dump(
        exclude_none=True
    )


def metric_values(values: dict) -> Metrics:
    return Metrics(
        **{key: float(values[key]) for key in Metrics.model_fields if key in values}
    )
