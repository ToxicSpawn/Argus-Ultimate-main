from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import json
import logging

'''
Cost Optimizer - Quantum Compute Cost Management

Tracks spending, optimizes vendor selection based on budget,
implements intelligent job batching, and provides alerts.
'''

logger = logging.getLogger(__name__)


@dataclass
class CostAlert:
    '''Cost alert notification'''

    level: str  # 'warning', 'critical'
    message: str
    timestamp: datetime
    current_spend: float
    budget_limit: float


@dataclass
class SpendingReport:
    '''Monthly spending report'''

    month: str
    total_spend: float
    budget: float
    utilization: float
    vendor_breakdown: dict[str, float] = field(default_factory=dict)
    problem_breakdown: dict[str, float] = field(default_factory=dict)
    alerts: list[CostAlert] = field(default_factory=list)


class CostOptimizer:
    '''
    Quantum Computing Cost Optimizer

    Manages budget, tracks spending, optimizes vendor selection,
    and implements intelligent batching to reduce overhead.
    '''

    def __init__(
        self,
        monthly_budget: float = 1000.0,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
    ):
        self.monthly_budget = monthly_budget
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # Spending tracking
        self.monthly_spend: dict[str, float] = defaultdict(float)
        self.vendor_spend: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.problem_spend: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Alerts
        self.alerts: list[CostAlert] = []

        # Pricing database (cost per 1000 shots)
        self.vendor_pricing = {
            "ionq": 10.0,
            "ibm": 5.0,
            "google": 8.0,
            "rigetti": 6.0,
            "dwave": 3.0,
            "simulator": 0.0,
        }

        logger.info(f"Cost Optimizer initialized with ${monthly_budget} monthly budget")

    def get_current_month(self) -> str:
        '''Get current month key'''
        return datetime.now().strftime("%Y-%m")

    def record_job_cost(self, vendor: str, problem_type: str, cost: float, shots: int) -> None:
        '''
        Record cost of a quantum job

        Args:
            vendor: Vendor name
            problem_type: Type of problem
            cost: Job cost
            shots: Number of shots
        '''
        month = self.get_current_month()

        # Update totals
        self.monthly_spend[month] += cost
        self.vendor_spend[month][vendor] += cost
        self.problem_spend[month][problem_type] += cost

        # Check for alerts
        current_spend = self.monthly_spend[month]
        utilization = current_spend / self.monthly_budget

        if utilization >= self.critical_threshold and not self._has_recent_alert("critical"):
            alert = CostAlert(
                level="critical",
                message=f"CRITICAL: {utilization * 100:.1f}% of monthly budget used (${current_spend:.2f}/${self.monthly_budget})",
                timestamp=datetime.now(),
                current_spend=current_spend,
                budget_limit=self.monthly_budget,
            )
            self.alerts.append(alert)
            logger.critical(alert.message)

        elif utilization >= self.warning_threshold and not self._has_recent_alert("warning"):
            alert = CostAlert(
                level="warning",
                message=f"WARNING: {utilization * 100:.1f}% of monthly budget used (${current_spend:.2f}/${self.monthly_budget})",
                timestamp=datetime.now(),
                current_spend=current_spend,
                budget_limit=self.monthly_budget,
            )
            self.alerts.append(alert)
            logger.warning(alert.message)

        logger.debug(f"Recorded {vendor} job: ${cost:.2f} ({shots} shots, {problem_type})")

    def _has_recent_alert(self, level: str, window_minutes: int = 60) -> bool:
        '''Check if alert of this level was raised recently'''
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff and a.level == level]
        return len(recent_alerts) > 0

    def can_afford(self, estimated_cost: float) -> bool:
        '''
        Check if we can afford a job within budget

        Args:
            estimated_cost: Estimated job cost

        Returns:
            True if job is affordable
        '''
        month = self.get_current_month()
        current_spend = self.monthly_spend[month]
        projected_spend = current_spend + estimated_cost

        return projected_spend <= self.monthly_budget

    def estimate_job_cost(self, vendor: str, shots: int) -> float:
        '''
        Estimate cost of a quantum job

        Args:
            vendor: Vendor name
            shots: Number of shots

        Returns:
            Estimated cost in dollars
        '''
        price_per_1k = self.vendor_pricing.get(vendor, 5.0)
        cost = (shots / 1000.0) * price_per_1k
        return cost

    def recommend_vendor(self, problem_type: str, shots: int, candidates: list[str]) -> str:
        '''
        Recommend most cost-effective vendor

        Args:
            problem_type: Type of problem
            shots: Number of shots needed
            candidates: list of candidate vendors

        Returns:
            Recommended vendor name
        '''
        month = self.get_current_month()
        current_spend = self.monthly_spend[month]

        # If budget is tight, prioritize cheapest
        utilization = current_spend / self.monthly_budget
        if utilization > self.critical_threshold:
            # Find cheapest vendor
            costs = [(v, self.estimate_job_cost(v, shots)) for v in candidates]
            recommended = min(costs, key=lambda x: x[1])[0]
            logger.info(f"Budget critical, recommending cheapest vendor: {recommended}")
            return recommended

        # Otherwise, balance cost and quality
        scores = []
        for vendor in candidates:
            cost = self.estimate_job_cost(vendor, shots)

            # Score factors:
            # - Lower cost is better
            # - Avoid simulator if we have budget
            cost_score = 1.0 / (cost + 0.1)
            quality_score = 0.0 if vendor == "simulator" else 1.0

            total_score = cost_score + quality_score
            scores.append((vendor, total_score))

        recommended = max(scores, key=lambda x: x[1])[0]
        logger.info(f"Recommended vendor: {recommended} (balanced cost/quality)")
        return recommended

    def batch_jobs(self, jobs: list[dict[str, Any]], max_batch_size: int = 10) -> list[list[dict[str, Any]]]:
        '''
        Batch similar jobs together to reduce overhead

        Args:
            jobs: list of job specifications
            max_batch_size: Maximum jobs per batch

        Returns:
            list of job batches
        '''
        # Group jobs by vendor and problem type
        groups = defaultdict(list)
        for job in jobs:
            key = (job.get("vendor"), job.get("problem_type"))
            groups[key].append(job)

        # Create batches
        batches = []
        for group_jobs in groups.values():
            # Split into batches of max_batch_size
            for i in range(0, len(group_jobs), max_batch_size):
                batch = group_jobs[i: i + max_batch_size]
                batches.append(batch)

        savings = len(jobs) - len(batches)
        logger.info(
            f"Batched {len(jobs)} jobs into {len(batches)} batches (saved {savings} overhead calls)"
        )

        return batches

    def get_spending_report(self, month: str = None) -> SpendingReport:
        '''
        Get spending report for a month

        Args:
            month: Month in YYYY-MM format (default: current month)

        Returns:
            Spending report
        '''
        if month is None:
            month = self.get_current_month()

        total_spend = self.monthly_spend.get(month, 0.0)
        utilization = total_spend / self.monthly_budget if self.monthly_budget > 0 else 0.0

        # Get vendor breakdown
        vendor_breakdown = dict(self.vendor_spend.get(month, {}))

        # Get problem breakdown
        problem_breakdown = dict(self.problem_spend.get(month, {}))

        # Recent alerts
        recent_alerts = [a for a in self.alerts if a.timestamp.strftime("%Y-%m") == month]

        report = SpendingReport(
            month=month,
            total_spend=total_spend,
            budget=self.monthly_budget,
            utilization=utilization,
            vendor_breakdown=vendor_breakdown,
            problem_breakdown=problem_breakdown,
            alerts=recent_alerts,
        )

        return report

    def get_cost_summary(self) -> dict[str, Any]:
        '''Get current cost summary'''
        month = self.get_current_month()
        total_spend = self.monthly_spend.get(month, 0.0)
        utilization = total_spend / self.monthly_budget if self.monthly_budget > 0 else 0.0

        return {
            "month": month,
            "total_spend": total_spend,
            "budget": self.monthly_budget,
            "remaining": self.monthly_budget - total_spend,
            "utilization": utilization,
            "alert_count": len(self.alerts),
            "status": self._get_budget_status(utilization),
        }

    def _get_budget_status(self, utilization: float) -> str:
        '''Get budget status string'''
        if utilization >= self.critical_threshold:
            return "critical"
        elif utilization >= self.warning_threshold:
            return "warning"
        else:
            return "healthy"

    def reset_monthly_budget(self) -> None:
        '''Reset budget tracking for new month (call at month start)'''
        month = self.get_current_month()
        if month not in self.monthly_spend:
            logger.info(f"No spending in {month} yet")
        else:
            logger.info(f"Month {month} tracking already active")

    def export_report(self, filepath: str) -> None:
        '''Export spending report to file'''

        report = self.get_spending_report()

        data = {
            "month": report.month,
            "total_spend": report.total_spend,
            "budget": report.budget,
            "utilization": report.utilization,
            "vendor_breakdown": report.vendor_breakdown,
            "problem_breakdown": report.problem_breakdown,
            "alerts": [
                {"level": a.level, "message": a.message, "timestamp": a.timestamp.isoformat()} for a in report.alerts
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Cost report exported to {filepath}")