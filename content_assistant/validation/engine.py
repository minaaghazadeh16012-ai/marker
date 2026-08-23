"""Runs the rules and reports what they found.

The engine is deliberately dumb: it owns no checks of its own, only the order
they run in and how the result is summarised. Adding a check means adding a
:class:`~content_assistant.validation.rules.Rule`, never editing this file.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from content_assistant.validation.rules import (
    ALL_RULES,
    Finding,
    Rule,
    Stage,
    ValidationContext,
)

#: Stages in the order they become checkable as the pipeline progresses.
STAGE_ORDER: Sequence[Stage] = ("structure", "semantic", "final")


class ValidationReport(BaseModel):
    findings: List[Finding] = Field(default_factory=list)
    stages_run: List[Stage] = Field(default_factory=list)

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def reviews(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "review"]

    @property
    def ok(self) -> bool:
        """True when nothing blocking was found. Warnings do not block."""
        return not self.errors

    def counts(self) -> Dict[str, int]:
        return dict(Counter(f.severity for f in self.findings))

    def by_code(self) -> Dict[str, int]:
        return dict(Counter(f.code for f in self.findings))

    def summary(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "stages_run": list(self.stages_run),
            "counts": self.counts(),
            "by_code": self.by_code(),
            "total": len(self.findings),
        }


def run_validation(
    ctx: ValidationContext,
    stages: Optional[Iterable[Stage]] = None,
    rules: Sequence[Rule] = ALL_RULES,
) -> ValidationReport:
    """Run every rule whose stage was requested.

    Calling this with ``stages=["structure"]`` is the pre-model gate: it works
    on the deterministic skeleton alone, so structure can be proved correct
    before a single token is spent.
    """
    selected = list(stages) if stages is not None else list(STAGE_ORDER)
    findings: List[Finding] = []
    for stage in STAGE_ORDER:
        if stage not in selected:
            continue
        for rule in rules:
            if rule.stage != stage:
                continue
            findings.extend(rule.check(ctx))
    return ValidationReport(findings=findings, stages_run=selected)


def render_review_markdown(report: ValidationReport, title: str = "گزارش بررسی") -> str:
    """Human-readable review file.

    Ordered by severity because a reviewer's time goes to errors first, and
    grouped by rule code so a systemic problem reads as one entry rather than
    forty.
    """
    lines = [f"# {title}", ""]
    counts = report.counts()
    lines.append(
        f"- خطا: {counts.get('error', 0)}  "
        f"- هشدار: {counts.get('warning', 0)}  "
        f"- نیازمند بررسی: {counts.get('review', 0)}"
    )
    lines.append("")
    for severity, heading in (
        ("error", "خطاها"),
        ("review", "نیازمند بررسی انسانی"),
        ("warning", "هشدارها"),
    ):
        items = [f for f in report.findings if f.severity == severity]
        if not items:
            continue
        lines += [f"## {heading} ({len(items)})", ""]
        for finding in items:
            target = f" `{finding.entity_id}`" if finding.entity_id else ""
            lines.append(f"- **{finding.code}**{target} — {finding.message}")
        lines.append("")
    if not report.findings:
        lines.append("موردی یافت نشد.")
    return "\n".join(lines)
