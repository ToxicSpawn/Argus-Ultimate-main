from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdvancedOrderTemplate:
    order_family: str
    tif: str
    post_only: bool
    reduce_only: bool
    reason: str


def build_post_only_template() -> AdvancedOrderTemplate:
    return AdvancedOrderTemplate(order_family="limit", tif="GTC", post_only=True, reduce_only=False, reason="maker-preserving advanced order template")


def build_reduce_only_template() -> AdvancedOrderTemplate:
    return AdvancedOrderTemplate(order_family="limit", tif="IOC", post_only=False, reduce_only=True, reason="risk-reducing advanced order template")
