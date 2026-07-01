"""Budget-aware cooling intervention allocation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import pandas as pd

from src.scenario_engine.interventions import (
    SUPPORTED_INTERVENTIONS,
    ScenarioResult,
    simulate_intervention,
)


DEFAULT_INTERVENTION_COSTS_INR_PER_KM2 = {
    "Urban Greening": 420_000_000.0,
    "Cool Roofs": 110_000_000.0,
    "Reflective / High-Albedo Surfaces": 80_000_000.0,
    "Blue-Green Infrastructure": 520_000_000.0,
    "Water Body Restoration": 700_000_000.0,
}


@dataclass(frozen=True)
class AllocationCandidate:
    scenario: ScenarioResult
    unit_cost_inr_per_km2: float
    estimated_cost_inr: float
    cooling_efficiency: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        data = self.scenario.to_dict()
        data.update(
            {
                "unit_cost_inr_per_km2": self.unit_cost_inr_per_km2,
                "estimated_cost_inr": self.estimated_cost_inr,
                "cooling_efficiency": self.cooling_efficiency,
                "rationale": self.rationale,
            }
        )
        return data


@dataclass(frozen=True)
class OptimizationResult:
    ranked_plan: pd.DataFrame
    candidate_table: pd.DataFrame
    total_budget_inr: float
    allocated_budget_inr: float
    remaining_budget_inr: float
    estimated_total_cooling_c: float
    strategy: str
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ranked_plan": self.ranked_plan.to_dict(orient="records"),
            "candidate_table": self.candidate_table.to_dict(orient="records"),
            "total_budget_inr": self.total_budget_inr,
            "allocated_budget_inr": self.allocated_budget_inr,
            "remaining_budget_inr": self.remaining_budget_inr,
            "estimated_total_cooling_c": self.estimated_total_cooling_c,
            "strategy": self.strategy,
            "explanation": self.explanation,
        }


class OptimizationStrategy(Protocol):
    def optimize(
        self,
        ward_records: pd.DataFrame,
        total_budget: float,
        intervention_costs: Mapping[str, float],
        target_wards: list[str] | None = None,
        shap_summary: pd.DataFrame | None = None,
    ) -> OptimizationResult:
        ...


@dataclass
class GreedyCoolingOptimizer:
    intensity: float = 70.0
    coverage_pct: float = 20.0
    interventions: tuple[str, ...] = SUPPORTED_INTERVENTIONS

    def optimize(
        self,
        ward_records: pd.DataFrame,
        total_budget: float,
        intervention_costs: Mapping[str, float] | None = None,
        target_wards: list[str] | None = None,
        shap_summary: pd.DataFrame | None = None,
    ) -> OptimizationResult:
        costs = {**DEFAULT_INTERVENTION_COSTS_INR_PER_KM2, **(intervention_costs or {})}
        wards = _filter_target_wards(ward_records, target_wards)
        candidates = build_allocation_candidates(
            wards,
            costs,
            self.interventions,
            self.intensity,
            self.coverage_pct,
            shap_summary,
        )
        candidate_table = pd.DataFrame([candidate.to_dict() for candidate in candidates])
        if candidate_table.empty:
            return OptimizationResult(
                ranked_plan=pd.DataFrame(),
                candidate_table=candidate_table,
                total_budget_inr=float(total_budget),
                allocated_budget_inr=0.0,
                remaining_budget_inr=float(total_budget),
                estimated_total_cooling_c=0.0,
                strategy="greedy_baseline",
                explanation="No feasible candidates were generated from the selected wards.",
            )

        ranked_candidates = sorted(candidates, key=lambda item: item.cooling_efficiency, reverse=True)
        selected: list[AllocationCandidate] = []
        selected_wards: set[str] = set()
        allocated = 0.0

        for candidate in ranked_candidates:
            ward_key = candidate.scenario.ward_id
            if ward_key in selected_wards:
                continue
            if candidate.estimated_cost_inr <= 0:
                continue
            if allocated + candidate.estimated_cost_inr <= total_budget:
                selected.append(candidate)
                selected_wards.add(ward_key)
                allocated += candidate.estimated_cost_inr

        plan = pd.DataFrame([candidate.to_dict() for candidate in selected])
        if not plan.empty:
            plan = plan.sort_values("cooling_efficiency", ascending=False).reset_index(drop=True)
            plan.insert(0, "rank", range(1, len(plan) + 1))

        return OptimizationResult(
            ranked_plan=plan,
            candidate_table=candidate_table.sort_values("cooling_efficiency", ascending=False).reset_index(drop=True),
            total_budget_inr=float(total_budget),
            allocated_budget_inr=round(allocated, 2),
            remaining_budget_inr=round(float(total_budget) - allocated, 2),
            estimated_total_cooling_c=round(float(plan["estimated_cooling_c"].sum()) if not plan.empty else 0.0, 2),
            strategy="greedy_baseline",
            explanation=(
                "Candidates are ranked by estimated ward-scale cooling per INR. "
                "The greedy baseline selects the best remaining intervention for each ward until the budget is exhausted."
            ),
        )


def optimize_cooling_plan(
    ward_records: pd.DataFrame,
    total_budget: float,
    intervention_costs: Mapping[str, float] | None = None,
    target_wards: list[str] | None = None,
    shap_summary: pd.DataFrame | None = None,
    strategy: OptimizationStrategy | None = None,
) -> OptimizationResult:
    optimizer = strategy or GreedyCoolingOptimizer()
    return optimizer.optimize(ward_records, total_budget, intervention_costs or {}, target_wards, shap_summary)


def build_allocation_candidates(
    ward_records: pd.DataFrame,
    intervention_costs: Mapping[str, float],
    interventions: tuple[str, ...],
    intensity: float,
    coverage_pct: float,
    shap_summary: pd.DataFrame | None = None,
) -> list[AllocationCandidate]:
    candidates: list[AllocationCandidate] = []
    if ward_records.empty:
        return candidates

    for ward in ward_records.itertuples(index=False):
        ward_record = ward._asdict()
        for intervention in interventions:
            scenario = simulate_intervention(ward_record, intervention, intensity, coverage_pct, shap_summary)
            unit_cost = float(intervention_costs.get(intervention, DEFAULT_INTERVENTION_COSTS_INR_PER_KM2[intervention]))
            estimated_cost = max(0.0, scenario.affected_area_km2 * unit_cost)
            efficiency = scenario.estimated_cooling_c / estimated_cost if estimated_cost > 0 else 0.0
            candidates.append(
                AllocationCandidate(
                    scenario=scenario,
                    unit_cost_inr_per_km2=unit_cost,
                    estimated_cost_inr=round(estimated_cost, 2),
                    cooling_efficiency=efficiency,
                    rationale=_candidate_rationale(scenario, estimated_cost),
                )
            )
    return candidates


def _filter_target_wards(ward_records: pd.DataFrame, target_wards: list[str] | None) -> pd.DataFrame:
    if ward_records.empty or not target_wards:
        return ward_records
    targets = {str(ward) for ward in target_wards}
    columns = [column for column in ("ward_id", "ward_name", "WARD") if column in ward_records.columns]
    if not columns:
        return ward_records
    mask = False
    for column in columns:
        mask = mask | ward_records[column].astype(str).isin(targets)
    return ward_records[mask].copy()


def _candidate_rationale(scenario: ScenarioResult, estimated_cost: float) -> str:
    return (
        f"{scenario.intervention_type} in {scenario.ward_name} is expected to cool about "
        f"{scenario.estimated_cooling_c:.2f} C across {scenario.affected_area_km2:.2f} km2 "
        f"with {scenario.confidence.lower()} confidence at an estimated cost of INR {estimated_cost:,.0f}."
    )
