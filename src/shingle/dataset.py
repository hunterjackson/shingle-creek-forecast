"""Assemble the model input dict (see features.py for the keys) from fetched history."""

from __future__ import annotations

import pandas as pd


def build_data(history: dict, rain: pd.DataFrame | None = None, pet: pd.DataFrame | None = None) -> dict:
    """Align everything on the Campbell gauge's daily index.

    rain/pet override the gridMET per-cell frames (e.g. MRMS radar cells).
    PET stays per gridMET cell; when rain cells differ from PET cells,
    water-balance features should use gridMET rain (pr_gm) explicitly.
    """
    gauge = history["gauge"]
    idx = gauge.index
    lakes = history["lakes"].reindex(idx)
    d = {
        "stage": gauge["stage_ft"],
        "q": gauge["discharge_cfs"],
        "q_up": history["upstream"]["discharge_cfs"].reindex(idx),
        "toho": lakes["toho_stage"],
        "s61_q": lakes["s61_q"],
        "s59_q": lakes["s59_q"],
        "etoho": lakes["etoho_stage"],
        "pr": (rain if rain is not None else history["pr"]).reindex(idx),
        "pet": (pet if pet is not None else history["pet"]).reindex(idx),
        "pr_gm": history["pr"].reindex(idx),
    }
    return d
