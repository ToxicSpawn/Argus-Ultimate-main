from __future__ import annotations
# pyright: reportMissingImports=false

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CausalGraph:
    """Lightweight causal graph representation."""

    nodes: List[str]
    edges: List[Tuple[str, str]]
    edge_strengths: Dict[Tuple[str, str], float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class CausalEffect:
    """Container for one estimated causal effect."""

    treatment: str
    outcome: str
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str


class CausalInferenceEngine:
    """Causal discovery and effect estimation for trading features."""

    _VALID_ATE_METHODS = {"propensity_score", "iv", "diff_in_diff", "do_pearl"}

    def __init__(
        self,
        significance_level: float = 0.05,
        max_conditioning_size: int = 1,
        min_samples: int = 30,
    ) -> None:
        self.significance_level = float(significance_level)
        self.max_conditioning_size = max(0, int(max_conditioning_size))
        self.min_samples = max(10, int(min_samples))
        self.learned_graphs: Dict[str, CausalGraph] = {}
        self.estimated_effects: Dict[str, CausalEffect] = {}
        logger.info("CausalInferenceEngine initialised")

    def learn_causal_graph(self, data: pd.DataFrame, method: str = "pc") -> CausalGraph:
        """Learn a DAG using a simplified PC-style causal discovery pipeline."""
        if method != "pc":
            raise ValueError("Only 'pc' causal discovery is currently supported")

        numeric = self._prepare_dataframe(data)
        columns = list(numeric.columns)
        n_cols = len(columns)

        adjacency: Dict[str, set[str]] = {column: set() for column in columns}
        edge_strengths: Dict[Tuple[str, str], float] = {}

        for left, right in combinations(columns, 2):
            corr, p_value = self._safe_pearsonr(numeric[left].to_numpy(), numeric[right].to_numpy())
            if p_value < self.significance_level:
                adjacency[left].add(right)
                adjacency[right].add(left)
                edge_strengths[(left, right)] = abs(corr)
                edge_strengths[(right, left)] = abs(corr)

        for cond_size in range(1, self.max_conditioning_size + 1):
            to_remove: List[Tuple[str, str]] = []
            for left, right in combinations(columns, 2):
                if right not in adjacency[left]:
                    continue
                candidates = list((adjacency[left] | adjacency[right]) - {left, right})
                if len(candidates) < cond_size:
                    continue
                independent = False
                for conditioning in combinations(candidates, cond_size):
                    p_value, strength = self._partial_independence_test(numeric, left, right, list(conditioning))
                    if p_value >= self.significance_level:
                        independent = True
                        edge_strengths[(left, right)] = strength
                        edge_strengths[(right, left)] = strength
                        break
                if independent:
                    to_remove.append((left, right))
            for left, right in to_remove:
                adjacency[left].discard(right)
                adjacency[right].discard(left)

        oriented_edges: List[Tuple[str, str]] = []
        oriented_strengths: Dict[Tuple[str, str], float] = {}
        node_confidence: Dict[str, float] = {}

        for left, right in combinations(columns, 2):
            if right not in adjacency[left]:
                continue
            source, target, strength = self._orient_edge(numeric, left, right)
            if self._creates_cycle(oriented_edges, source, target):
                source, target = target, source
            if self._creates_cycle(oriented_edges, source, target):
                source, target = sorted((left, right))
                if self._creates_cycle(oriented_edges, source, target):
                    continue
            oriented_edges.append((source, target))
            oriented_strengths[(source, target)] = strength

        for node in columns:
            incident = [value for (src, dst), value in oriented_strengths.items() if src == node or dst == node]
            node_confidence[node] = float(np.clip(np.mean(incident) if incident else 0.0, 0.0, 1.0))

        graph = CausalGraph(
            nodes=columns,
            edges=oriented_edges,
            edge_strengths=oriented_strengths,
            confidence_scores=node_confidence,
        )
        self.learned_graphs["latest"] = graph
        logger.info("learn_causal_graph: discovered %d nodes and %d edges", len(graph.nodes), len(graph.edges))
        return graph

    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        data: pd.DataFrame,
        method: str = "propensity_score",
    ) -> CausalEffect:
        """Estimate an average treatment effect using one of the supported estimators."""
        numeric = self._prepare_dataframe(data)
        self._validate_columns(numeric, [treatment, outcome])
        if method not in self._VALID_ATE_METHODS:
            raise ValueError(f"Unsupported method '{method}'")

        if method == "propensity_score":
            effect = self._estimate_propensity_score(treatment, outcome, numeric)
        elif method == "iv":
            effect = self._estimate_iv(treatment, outcome, numeric)
        elif method == "diff_in_diff":
            effect = self._estimate_diff_in_diff(treatment, outcome, numeric)
        else:
            graph = self.learn_causal_graph(numeric)
            effect = self.estimate_causal_effect(graph, treatment, outcome, numeric)

        self.estimated_effects[f"{method}:{treatment}->{outcome}"] = effect
        logger.info(
            "estimate_ate(%s -> %s, method=%s): effect=%.6f p=%.6f",
            treatment,
            outcome,
            method,
            effect.effect_size,
            effect.p_value,
        )
        return effect

    def estimate_causal_effect(
        self,
        graph: CausalGraph,
        treatment: str,
        outcome: str,
        data: pd.DataFrame,
    ) -> CausalEffect:
        """Estimate a graph-adjusted causal effect using a do-calculus style backdoor adjustment."""
        numeric = self._prepare_dataframe(data)
        self._validate_columns(numeric, [treatment, outcome])

        confounders = self.detect_confounders(graph, treatment, outcome)
        y = numeric[outcome].to_numpy(dtype=float)
        X_columns = confounders + [treatment]
        X = numeric[X_columns].to_numpy(dtype=float)
        beta, se, p_value, ci = self._ols_effect(X, y, treatment_index=len(X_columns) - 1)

        effect = CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=float(beta),
            confidence_interval=ci,
            p_value=float(p_value),
            method="do_pearl",
        )
        self.estimated_effects[f"do_pearl:{treatment}->{outcome}"] = effect
        return effect

    def do_intervention(self, graph: CausalGraph, interventions: Dict[str, float]) -> CausalGraph:
        """Apply do-calculus intervention by severing incoming edges into intervened nodes."""
        intervened_nodes = set(interventions.keys())
        new_edges = [(src, dst) for src, dst in graph.edges if dst not in intervened_nodes]
        new_strengths = {
            (src, dst): strength
            for (src, dst), strength in graph.edge_strengths.items()
            if (src, dst) in new_edges
        }
        new_confidence = dict(graph.confidence_scores)
        for node in intervened_nodes:
            if node in new_confidence:
                new_confidence[node] = 1.0
        logger.info("do_intervention: applied interventions to %s", sorted(intervened_nodes))
        return CausalGraph(
            nodes=list(graph.nodes),
            edges=new_edges,
            edge_strengths=new_strengths,
            confidence_scores=new_confidence,
        )

    def calculate_feature_importance_causal(self, data: pd.DataFrame, target: str) -> Dict[str, float]:
        """Estimate causal feature importance by absolute graph-adjusted effect size."""
        numeric = self._prepare_dataframe(data)
        self._validate_columns(numeric, [target])
        graph = self.learn_causal_graph(numeric)
        scores: Dict[str, float] = {}

        for column in numeric.columns:
            if column == target:
                continue
            try:
                effect = self.estimate_causal_effect(graph, column, target, numeric)
                scores[column] = abs(effect.effect_size)
            except Exception as exc:
                logger.debug("calculate_feature_importance_causal: skipping %s due to %s", column, exc)
                scores[column] = 0.0

        total = sum(scores.values())
        if total > 0:
            scores = {name: value / total for name, value in scores.items()}
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def test_mediation(self, treatment: str, mediator: str, outcome: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Test mediation using linear path coefficients and Sobel significance."""
        numeric = self._prepare_dataframe(data)
        self._validate_columns(numeric, [treatment, mediator, outcome])

        x = numeric[[treatment]].to_numpy(dtype=float)
        m = numeric[mediator].to_numpy(dtype=float)
        y = numeric[outcome].to_numpy(dtype=float)

        a_coef, a_se, a_p_value, _ = self._ols_effect(x, m, treatment_index=0)
        X_by = numeric[[treatment, mediator]].to_numpy(dtype=float)
        b_coef, b_se, b_p_value, _ = self._ols_effect(X_by, y, treatment_index=1)
        c_prime, _, c_prime_p_value, _ = self._ols_effect(X_by, y, treatment_index=0)
        c_total, _, c_total_p_value, _ = self._ols_effect(x, y, treatment_index=0)

        indirect_effect = a_coef * b_coef
        sobel_se = float(np.sqrt((b_coef ** 2) * (a_se ** 2) + (a_coef ** 2) * (b_se ** 2)))
        if sobel_se <= 0:
            sobel_z = 0.0
            sobel_p = 1.0
        else:
            sobel_z = float(indirect_effect / sobel_se)
            sobel_p = float(2 * (1 - stats.norm.cdf(abs(sobel_z))))

        proportion = float(indirect_effect / c_total) if abs(c_total) > 1e-12 else 0.0
        return {
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
            "a_path": float(a_coef),
            "b_path": float(b_coef),
            "direct_effect": float(c_prime),
            "total_effect": float(c_total),
            "indirect_effect": float(indirect_effect),
            "sobel_z": sobel_z,
            "p_value": sobel_p,
            "proportion_mediated": proportion,
            "significant": sobel_p < self.significance_level,
            "path_p_values": {
                "a_path": float(a_p_value),
                "b_path": float(b_p_value),
                "direct_effect": float(c_prime_p_value),
                "total_effect": float(c_total_p_value),
            },
        }

    def generate_counterfactuals(self, data: pd.DataFrame, interventions: List[Dict[str, float]]) -> pd.DataFrame:
        """Generate simple linear-structural counterfactual scenarios for each intervention."""
        numeric = self._prepare_dataframe(data)
        graph = self.learn_causal_graph(numeric)
        topological_order = self._topological_sort(graph)
        models = self._fit_structural_models(graph, numeric)
        scenarios: List[pd.DataFrame] = []

        for index, intervention in enumerate(interventions):
            scenario = numeric.copy()
            for variable, value in intervention.items():
                if variable not in scenario.columns:
                    raise KeyError(f"Intervention variable '{variable}' not found in data")
                scenario.loc[:, variable] = float(value)

            for node in topological_order:
                if node in intervention:
                    continue
                model = models.get(node)
                if model is None:
                    continue
                parents = model["parents"]
                if not parents:
                    continue
                X = scenario[parents].to_numpy(dtype=float)
                scenario.loc[:, node] = model["intercept"] + X @ model["coefficients"]

            scenario = scenario.copy()
            scenario["scenario_id"] = index
            scenario["intervention"] = str(intervention)
            scenarios.append(scenario)

        if not scenarios:
            return numeric.copy()
        return pd.concat(scenarios, ignore_index=True)

    def export_dag_visualization_data(self, graph: CausalGraph) -> Dict[str, Any]:
        """Export DAG data suitable for front-end visualisation."""
        return {
            "nodes": [
                {
                    "id": node,
                    "label": node,
                    "confidence": graph.confidence_scores.get(node, 0.0),
                }
                for node in graph.nodes
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "strength": graph.edge_strengths.get((source, target), 0.0),
                }
                for source, target in graph.edges
            ],
        }

    def analyze_causal_paths(self, graph: CausalGraph, treatment: str, outcome: str) -> List[Dict[str, Any]]:
        """Return causal paths with multiplicative path strengths."""
        paths = self._find_paths(graph, treatment, outcome)
        analysed: List[Dict[str, Any]] = []
        for path in paths:
            strength = 1.0
            for src, dst in zip(path[:-1], path[1:]):
                strength *= graph.edge_strengths.get((src, dst), 0.0)
            analysed.append({"path": path, "strength": float(strength), "length": len(path) - 1})
        return sorted(analysed, key=lambda item: item["strength"], reverse=True)

    def detect_confounders(self, graph: CausalGraph, treatment: str, outcome: str) -> List[str]:
        """Detect likely confounders from the graph topology."""
        parents_treatment = {src for src, dst in graph.edges if dst == treatment}
        ancestors_outcome = self._ancestors(graph, outcome)
        confounders = sorted(parents_treatment & ancestors_outcome - {treatment, outcome})
        return confounders

    def sensitivity_analysis(
        self,
        treatment: str,
        outcome: str,
        data: pd.DataFrame,
        gamma_values: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """Rosenbaum-style sensitivity analysis for unobserved confounding."""
        base_effect = self.estimate_ate(treatment, outcome, data, method="propensity_score")
        gamma_values = list(gamma_values or [1.0, 1.25, 1.5, 2.0])
        attenuated: Dict[str, float] = {}
        for gamma in gamma_values:
            adjusted = base_effect.effect_size / float(max(gamma, 1.0))
            attenuated[f"gamma_{gamma}"] = float(adjusted)
        attenuated["base_effect"] = float(base_effect.effect_size)
        return attenuated

    def get_causal_summary(self) -> Dict[str, Any]:
        """Return lightweight summary for the advanced features orchestrator."""
        return {
            "graphs": len(self.learned_graphs),
            "estimated_effects": len(self.estimated_effects),
            "latest_graph_nodes": self.learned_graphs.get("latest", CausalGraph([], [])).nodes,
            "effects": {
                name: {
                    "treatment": effect.treatment,
                    "outcome": effect.outcome,
                    "effect_size": effect.effect_size,
                    "confidence_interval": effect.confidence_interval,
                    "p_value": effect.p_value,
                    "method": effect.method,
                }
                for name, effect in self.estimated_effects.items()
            },
        }

    def _estimate_propensity_score(self, treatment: str, outcome: str, data: pd.DataFrame) -> CausalEffect:
        treatment_series = self._binary_treatment(data[treatment])
        outcome_values = data[outcome].to_numpy(dtype=float)
        covariates = [column for column in data.columns if column not in {treatment, outcome}]
        X = data[covariates].to_numpy(dtype=float) if covariates else np.empty((len(data), 0), dtype=float)
        scores = self._propensity_scores(X, treatment_series)

        treated_indices = np.where(treatment_series == 1)[0]
        control_indices = np.where(treatment_series == 0)[0]
        if len(treated_indices) == 0 or len(control_indices) == 0:
            raise ValueError("Propensity score matching requires both treated and control observations")

        matched_differences: List[float] = []
        for treated_index in treated_indices:
            distances = np.abs(scores[control_indices] - scores[treated_index])
            best_control = control_indices[int(np.argmin(distances))]
            matched_differences.append(float(outcome_values[treated_index] - outcome_values[best_control]))

        return self._effect_from_differences(treatment, outcome, matched_differences, "propensity_score")

    def _estimate_iv(self, treatment: str, outcome: str, data: pd.DataFrame) -> CausalEffect:
        candidates = [column for column in data.columns if column not in {treatment, outcome}]
        if not candidates:
            raise ValueError("Instrumental variables estimation requires at least one candidate instrument")

        instrument = self._select_instrument(data, treatment, outcome, candidates)
        z = data[[instrument]].to_numpy(dtype=float)
        x = data[treatment].to_numpy(dtype=float)
        y = data[outcome].to_numpy(dtype=float)

        first_stage = self._design_matrix(z)
        first_beta = np.linalg.lstsq(first_stage, x, rcond=None)[0]
        treatment_hat = first_stage @ first_beta
        second_X = self._design_matrix(treatment_hat.reshape(-1, 1))
        beta = np.linalg.lstsq(second_X, y, rcond=None)[0]
        residuals = y - second_X @ beta
        dof = max(len(y) - second_X.shape[1], 1)
        sigma2 = float(np.sum(residuals ** 2) / dof)
        cov = sigma2 * np.linalg.pinv(second_X.T @ second_X)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
        coef = float(beta[1])
        p_value, ci = self._wald_inference(coef, se, dof)

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=coef,
            confidence_interval=ci,
            p_value=p_value,
            method="iv",
        )

    def _estimate_diff_in_diff(self, treatment: str, outcome: str, data: pd.DataFrame) -> CausalEffect:
        if "post" not in data.columns:
            raise ValueError("Difference-in-differences requires a 'post' column")

        treat = self._binary_treatment(data[treatment])
        post = self._binary_treatment(data["post"])
        interaction = treat * post
        y = data[outcome].to_numpy(dtype=float)
        X = np.column_stack([treat, post, interaction])
        coef, se, p_value, ci = self._ols_effect(X, y, treatment_index=2)
        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=float(coef),
            confidence_interval=ci,
            p_value=float(p_value),
            method="diff_in_diff",
        )

    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        numeric = data.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError("data must contain numeric columns")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if len(numeric) < self.min_samples:
            raise ValueError(f"At least {self.min_samples} complete observations are required")
        return numeric

    def _validate_columns(self, data: pd.DataFrame, required: List[str]) -> None:
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def _safe_pearsonr(self, left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            return 0.0, 1.0
        corr, p_value = stats.pearsonr(left, right)
        return float(0.0 if np.isnan(corr) else corr), float(1.0 if np.isnan(p_value) else p_value)

    def _partial_independence_test(
        self,
        data: pd.DataFrame,
        left: str,
        right: str,
        conditioning: List[str],
    ) -> Tuple[float, float]:
        left_values = data[left].to_numpy(dtype=float)
        right_values = data[right].to_numpy(dtype=float)
        z = data[conditioning].to_numpy(dtype=float)
        left_residual = self._residualise(left_values, z)
        right_residual = self._residualise(right_values, z)
        corr, p_value = self._safe_pearsonr(left_residual, right_residual)
        return p_value, abs(corr)

    def _residualise(self, target: np.ndarray, features: np.ndarray) -> np.ndarray:
        X = self._design_matrix(features)
        beta = np.linalg.lstsq(X, target, rcond=None)[0]
        return target - X @ beta

    def _orient_edge(self, data: pd.DataFrame, left: str, right: str) -> Tuple[str, str, float]:
        lead_left = abs(self._lagged_association(data[left].to_numpy(dtype=float), data[right].to_numpy(dtype=float)))
        lead_right = abs(self._lagged_association(data[right].to_numpy(dtype=float), data[left].to_numpy(dtype=float)))
        strength = max(lead_left, lead_right)
        if lead_left > lead_right + 1e-6:
            return left, right, float(np.clip(strength, 0.0, 1.0))
        if lead_right > lead_left + 1e-6:
            return right, left, float(np.clip(strength, 0.0, 1.0))
        left_var = float(np.var(data[left].to_numpy(dtype=float)))
        right_var = float(np.var(data[right].to_numpy(dtype=float)))
        if left_var <= right_var:
            return left, right, float(np.clip(max(strength, abs(data[left].corr(data[right]))), 0.0, 1.0))
        return right, left, float(np.clip(max(strength, abs(data[left].corr(data[right]))), 0.0, 1.0))

    def _lagged_association(self, cause: np.ndarray, effect: np.ndarray) -> float:
        if len(cause) < 4 or len(effect) < 4:
            corr, _ = self._safe_pearsonr(cause, effect)
            return corr
        corr, _ = self._safe_pearsonr(cause[:-1], effect[1:])
        return corr

    def _creates_cycle(self, edges: List[Tuple[str, str]], source: str, target: str) -> bool:
        adjacency: Dict[str, List[str]] = {}
        for src, dst in edges + [(source, target)]:
            adjacency.setdefault(src, []).append(dst)
        stack = [target]
        visited = set()
        while stack:
            node = stack.pop()
            if node == source:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency.get(node, []))
        return False

    def _ols_effect(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment_index: int,
    ) -> Tuple[float, float, float, Tuple[float, float]]:
        design = self._design_matrix(X)
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residuals = y - design @ beta
        dof = max(len(y) - design.shape[1], 1)
        sigma2 = float(np.sum(residuals ** 2) / dof)
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        index = treatment_index + 1
        coef = float(beta[index])
        se = float(np.sqrt(max(cov[index, index], 0.0)))
        p_value, ci = self._wald_inference(coef, se, dof)
        return coef, se, p_value, ci

    def _wald_inference(self, coefficient: float, standard_error: float, dof: int) -> Tuple[float, Tuple[float, float]]:
        if standard_error <= 0:
            return 1.0, (coefficient, coefficient)
        critical = float(stats.t.ppf(0.975, dof)) if dof > 0 else 1.96
        statistic = coefficient / standard_error
        p_value = float(2 * (1 - stats.t.cdf(abs(statistic), dof))) if dof > 0 else float(2 * (1 - stats.norm.cdf(abs(statistic))))
        ci = (float(coefficient - critical * standard_error), float(coefficient + critical * standard_error))
        return p_value, ci

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        return np.column_stack([np.ones(len(features)), features])

    def _binary_treatment(self, series: pd.Series) -> np.ndarray:
        values = series.to_numpy(dtype=float)
        unique = set(np.unique(values))
        if unique <= {0.0, 1.0}:
            return values.astype(int)
        threshold = float(np.median(values))
        logger.debug("Binarising treatment '%s' at median %.6f", series.name, threshold)
        return (values >= threshold).astype(int)

    def _propensity_scores(self, X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        if X.size == 0:
            base_rate = float(np.clip(np.mean(treatment), 1e-4, 1 - 1e-4))
            return np.full(len(treatment), base_rate, dtype=float)

        design = self._design_matrix(X)
        beta = np.zeros(design.shape[1], dtype=float)
        ridge = 1e-6 * np.eye(design.shape[1])
        ridge[0, 0] = 0.0

        for _ in range(50):
            linear = design @ beta
            probs = 1.0 / (1.0 + np.exp(-np.clip(linear, -30, 30)))
            weights = np.clip(probs * (1.0 - probs), 1e-6, None)
            z = linear + (treatment - probs) / weights
            WX = design * weights[:, None]
            lhs = design.T @ WX + ridge
            rhs = design.T @ (weights * z)
            beta_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            if np.max(np.abs(beta_new - beta)) < 1e-6:
                beta = beta_new
                break
            beta = beta_new

        scores = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
        return np.clip(scores, 1e-4, 1 - 1e-4)

    def _effect_from_differences(
        self,
        treatment: str,
        outcome: str,
        differences: List[float],
        method: str,
    ) -> CausalEffect:
        diff = np.asarray(differences, dtype=float)
        if diff.size == 0:
            raise ValueError("No matched differences available for effect estimation")
        effect = float(np.mean(diff))
        se = float(stats.sem(diff)) if diff.size > 1 else 0.0
        if diff.size > 1 and se > 0:
            critical = float(stats.t.ppf(0.975, diff.size - 1))
            p_value = float(stats.ttest_1samp(diff, 0.0).pvalue)
            ci = (effect - critical * se, effect + critical * se)
        else:
            p_value = 1.0
            ci = (effect, effect)
        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=effect,
            confidence_interval=(float(ci[0]), float(ci[1])),
            p_value=p_value,
            method=method,
        )

    def _select_instrument(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        candidates: List[str],
    ) -> str:
        best_name: Optional[str] = None
        best_score = float("-inf")
        for candidate in candidates:
            corr_tx, _ = self._safe_pearsonr(data[candidate].to_numpy(dtype=float), data[treatment].to_numpy(dtype=float))
            corr_out, _ = self._safe_pearsonr(data[candidate].to_numpy(dtype=float), data[outcome].to_numpy(dtype=float))
            score = abs(corr_tx) - abs(corr_out)
            if score > best_score and abs(corr_tx) > 0.05:
                best_score = score
                best_name = candidate
        if best_name is None:
            raise ValueError("No valid instrument candidate found")
        logger.info("_select_instrument: selected '%s' for %s -> %s", best_name, treatment, outcome)
        return best_name

    def _fit_structural_models(self, graph: CausalGraph, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        models: Dict[str, Dict[str, Any]] = {}
        for node in graph.nodes:
            parents = [src for src, dst in graph.edges if dst == node]
            if not parents:
                continue
            X = data[parents].to_numpy(dtype=float)
            y = data[node].to_numpy(dtype=float)
            design = self._design_matrix(X)
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            models[node] = {
                "parents": parents,
                "intercept": float(beta[0]),
                "coefficients": beta[1:].astype(float),
            }
        return models

    def _topological_sort(self, graph: CausalGraph) -> List[str]:
        indegree = {node: 0 for node in graph.nodes}
        adjacency: Dict[str, List[str]] = {node: [] for node in graph.nodes}
        for source, target in graph.edges:
            indegree[target] += 1
            adjacency[source].append(target)

        queue = [node for node, degree in indegree.items() if degree == 0]
        ordered: List[str] = []
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for neighbour in adjacency[node]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)
        return ordered if len(ordered) == len(graph.nodes) else list(graph.nodes)

    def _find_paths(self, graph: CausalGraph, source: str, target: str) -> List[List[str]]:
        adjacency: Dict[str, List[str]] = {node: [] for node in graph.nodes}
        for start, end in graph.edges:
            adjacency.setdefault(start, []).append(end)

        results: List[List[str]] = []

        def dfs(node: str, path: List[str]) -> None:
            if node == target:
                results.append(path.copy())
                return
            for neighbour in adjacency.get(node, []):
                if neighbour in path:
                    continue
                path.append(neighbour)
                dfs(neighbour, path)
                path.pop()

        dfs(source, [source])
        return results

    def _ancestors(self, graph: CausalGraph, node: str) -> set[str]:
        parents = {dst: [] for dst in graph.nodes}
        for source, target in graph.edges:
            parents.setdefault(target, []).append(source)
        seen: set[str] = set()
        stack = list(parents.get(node, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(parents.get(current, []))
        return seen


__all__ = ["CausalGraph", "CausalEffect", "CausalInferenceEngine"]



