from __future__ import annotations

from argus_live.proving.day_review import DayReview


def go_no_go(review: DayReview) -> str:
    return review.promotion_decision
