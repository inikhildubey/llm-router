import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("llm-router.telemetry")

@dataclass
class RequestMetrics:
    prompt: str
    triage_latency: float
    execution_latency: float
    selected_model: str
    status: str = "success"
    error_message: Optional[str] = None

class Telemetry:
    """
    Production observability layer.
    Tracks performance and cost metrics for the routing system.
    """
    def __init__(self):
        self.metrics_log = []

    def record_request(self, metrics: RequestMetrics):
        self.metrics_log.append(metrics)
        logger.info(
            f"Metric recorded | Model: {metrics.selected_model} | "
            f"Triage: {metrics.triage_latency:.3f}s | "
            f"Exec: {metrics.execution_latency:.3f}s | "
            f"Total: {metrics.triage_latency + metrics.execution_latency:.3f}s"
        )

    def get_report(self):
        if not self.metrics_log:
            return "No metrics recorded."

        total_reqs = len(self.metrics_log)
        avg_triage = sum(m.triage_latency for m in self.metrics_log) / total_reqs
        avg_exec = sum(m.execution_latency for m in self.metrics_log) / total_reqs

        model_counts = {}
        for m in self.metrics_log:
            model_counts[m.selected_model] = model_counts.get(m.selected_model, 0) + 1

        report = [
            "\n" + "="*30,
            " PRODUCTION PERFORMANCE REPORT ",
            "="*30,
            f"Total Requests: {total_reqs}",
            f"Avg Triage Latency: {avg_triage:.3f}s",
            f"Avg Execution Latency: {avg_exec:.3f}s",
            f"Overall Avg Latency: {avg_triage + avg_exec:.3f}s",
            "\nModel Distribution:",
        ]
        for model, count in model_counts.items():
            report.append(f" - {model}: {count} requests")

        return "\n".join(report)
