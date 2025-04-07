import time
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RequestLatencyTracker:
    """Tracks latency between different stages of request processing"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.perf_counter()
        self.timestamps: Dict[str, float] = {"request_received": self.start_time}
        self._logged_events = set()  # Track which events have been logged

    def mark(self, event: str, force_log: bool = False):
        """Record timestamp for a specific event

        Args:
            event (str): Name of the event to mark
            force_log (bool): Whether to log this event even if it's been logged before
        """
        self.timestamps[event] = time.perf_counter()

        # Only log first occurrence of each event unless forced
        if event not in self._logged_events or force_log:
            self._log_latest_latency(event)
            self._logged_events.add(event)

    def _log_latest_latency(self, current_event: str):
        """Log the latency from the previous event to the current one"""
        events = sorted(self.timestamps.items(), key=lambda x: x[1])
        if len(events) < 2:
            return

        # Get the previous and current events
        prev_event = None
        for i, (event, _) in enumerate(events):
            if event == current_event and i > 0:
                prev_event = events[i - 1]
                break

        if prev_event:
            latency_ms = (self.timestamps[current_event] - prev_event[1]) * 1000
            # Only log significant transitions
            if current_event in {
                "first_token_generated",
                "streaming_complete",
                "tool_calls_processed",
            }:
                logger.info(
                    f"Request {self.request_id} - Latency from {prev_event[0]} to {current_event}: {latency_ms:.2f}ms"
                )

    def get_latency_breakdown(self) -> Dict[str, float]:
        """Calculate latencies between all events in milliseconds"""
        latencies = {}
        events = sorted(self.timestamps.items(), key=lambda x: x[1])

        for i in range(len(events) - 1):
            current, next_event = events[i], events[i + 1]
            latency_ms = (next_event[1] - current[1]) * 1000
            latencies[f"{current[0]}_to_{next_event[0]}"] = round(latency_ms, 2)

        # Add total latency
        total_ms = (events[-1][1] - events[0][1]) * 1000
        latencies["total_ms"] = round(total_ms, 2)
        return latencies
