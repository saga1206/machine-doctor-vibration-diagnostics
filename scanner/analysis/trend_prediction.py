"""
History + Trend Prediction Module (Step 9)

Kid explanation:
    One check-up tells you how the machine looks TODAY. But is it
    getting better, staying the same, or getting worse? That needs
    MULTIPLE check-ups compared over time -- like checking a kid's
    height every month to see if they're growing normally or too fast.

    This module looks at a machine's last few scans and answers:
    "compared to a few scans ago, is the vibration amplitude going up,
    and if so, by how much?" A big, fast increase is worth flagging
    even if today's single reading still looks "healthy" on its own --
    catching a problem while it's still trending toward trouble, not
    after it's already broken.
"""
from dataclasses import dataclass


@dataclass
class TrendResult:
    direction: str            # "increasing", "decreasing", "stable", "insufficient_data"
    percent_change: float     # over the compared window
    n_scans_compared: int
    warning_level: str        # "none", "watch", "urgent"
    message: str


# Heuristic thresholds for how much amplitude increase is worth flagging.
# Like the amplitude-level thresholds in Step 7, these are reasonable
# demo defaults on an uncalibrated scale, easy to retune once real-world
# data is available.
WATCH_THRESHOLD_PERCENT = 15.0
URGENT_THRESHOLD_PERCENT = 30.0


def analyze_trend(amplitude_history, max_window=3) -> TrendResult:
    """
    Main entry point for Step 9.

    `amplitude_history` is a list of amplitude values in CHRONOLOGICAL
    order (oldest first, matching how scans naturally come out of the
    database ordered by date). `max_window` controls how many of the
    most recent scans to compare, matching the original spec's example
    ("vibration amplitude increased 30% over last 3 scans").
    """
    valid_history = [a for a in amplitude_history if a is not None]

    if len(valid_history) < 2:
        return TrendResult(
            direction="insufficient_data", percent_change=0, n_scans_compared=len(valid_history),
            warning_level="none",
            message="Not enough scan history yet to detect a trend (need at least 2 scans).",
        )

    window = valid_history[-max_window:]
    first, last = window[0], window[-1]

    if first == 0:
        # Avoid divide-by-zero; treat any increase from exactly zero as
        # a large jump rather than an undefined percentage.
        percent_change = 100.0 if last > 0 else 0.0
    else:
        percent_change = ((last - first) / abs(first)) * 100.0

    if abs(percent_change) < 5.0:
        direction = "stable"
    elif percent_change > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    if direction == "increasing" and percent_change >= URGENT_THRESHOLD_PERCENT:
        warning_level = "urgent"
        message = (
            f"Vibration amplitude increased {percent_change:.0f}% over the last "
            f"{len(window)} scans. Recommend inspection within 2 weeks."
        )
    elif direction == "increasing" and percent_change >= WATCH_THRESHOLD_PERCENT:
        warning_level = "watch"
        message = (
            f"Vibration amplitude increased {percent_change:.0f}% over the last "
            f"{len(window)} scans. Worth monitoring closely at the next scan."
        )
    elif direction == "decreasing":
        warning_level = "none"
        message = (
            f"Vibration amplitude decreased {abs(percent_change):.0f}% over the last "
            f"{len(window)} scans. Trending in a good direction."
        )
    elif direction == "stable":
        warning_level = "none"
        message = f"Vibration amplitude has stayed roughly stable over the last {len(window)} scans."
    else:
        # Increasing, but below the "watch" threshold
        warning_level = "none"
        message = (
            f"Vibration amplitude increased slightly ({percent_change:.0f}%) over the last "
            f"{len(window)} scans -- not yet a concern, but worth keeping an eye on."
        )

    return TrendResult(
        direction=direction,
        percent_change=round(percent_change, 1),
        n_scans_compared=len(window),
        warning_level=warning_level,
        message=message,
    )