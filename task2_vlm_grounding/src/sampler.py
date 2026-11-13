"""Seeded, resumable curriculum sampler (pure python / numpy; no torch).

Stage 1: a stream of two passes over the caption examples (per-epoch seeded permutations).
Stages 2/3: 8 random *draws* per update; every draw yields 2 examples (a complete complementary
pair, or two caption examples for replay), so one update = 16 examples = 2 microbatches of 8
and a microbatch always holds whole draws.
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np

DRAWS_PER_UPDATE = 8
EXAMPLES_PER_UPDATE = 16
COMPONENTS = ("presence", "counting", "vqa", "caption")


def caption_to_example(e: dict) -> dict:
    return {
        "example_id": e["example_id"], "pair_id": None, "image_id": int(e["image_id"]),
        "question": e["question"], "answer": e["answer"], "component": "caption", "qtype": "caption",
        "target": "caption", "canonical": e["answer"], "all_answers": None,
    }


def pair_to_examples(p: dict) -> list[dict]:
    comp = "vqa" if p["suite"] == "natural_vqa" else p["kind"]
    out = []
    for it in p["items"]:
        out.append({
            "example_id": it["example_id"], "pair_id": p["pair_id"], "image_id": int(it["image_id"]),
            "question": p["question"], "answer": it["answer"], "component": comp, "qtype": p["kind"],
            "target": "short", "canonical": it["canonical"], "all_answers": it.get("all_answers"),
        })
    return out


class CurriculumSampler:
    def __init__(self, manifest: dict, seed: int, stage_mixes: Sequence[dict]) -> None:
        self.seed = int(seed)
        self.captions = list(manifest["caption_train"])
        self.pools = {
            "presence": list(manifest["grounding_train_presence"]),
            "counting": list(manifest["grounding_train_counting"]),
            "vqa": list(manifest["vqa_train"]),
        }
        self.stage_mixes = [dict(m) for m in stage_mixes]
        for comp in ("presence", "counting", "vqa"):
            if not self.pools[comp]:
                raise ValueError(f"empty pool: {comp}")
        if len(self.captions) < 2:
            raise ValueError("need at least 2 caption examples")
        self.epoch = 0
        self.pos = 0
        self._perm: np.ndarray | None = None
        self.rngs = [None,
                     random.Random(f"{self.seed}/stage2_draws"),
                     random.Random(f"{self.seed}/stage3_draws")]

    # ---------------- stage 1 ----------------
    def _perm_for(self, epoch: int) -> np.ndarray:
        return np.random.RandomState(self.seed * 1000 + epoch).permutation(len(self.captions))

    def _stage1_update(self) -> tuple[list[dict], dict]:
        out = []
        while len(out) < EXAMPLES_PER_UPDATE:
            if self._perm is None:
                self._perm = self._perm_for(self.epoch)
            out.append(caption_to_example(self.captions[int(self._perm[self.pos])]))
            self.pos += 1
            if self.pos >= len(self.captions):
                self.epoch += 1
                self.pos = 0
                self._perm = None
        return out, {"caption": DRAWS_PER_UPDATE}

    # ---------------- stages 2 / 3 ----------------
    def _draw_component(self, rng: random.Random, mix: dict) -> str:
        r, acc = rng.random(), 0.0
        order = [c for c in COMPONENTS if c in mix]
        for comp in order:
            acc += mix[comp]
            if r < acc:
                return comp
        return order[-1]

    def _mixed_update(self, stage_index: int) -> tuple[list[dict], dict]:
        rng, mix = self.rngs[stage_index], self.stage_mixes[stage_index]
        out, counts = [], {c: 0 for c in mix}
        for _ in range(DRAWS_PER_UPDATE):
            comp = self._draw_component(rng, mix)
            counts[comp] += 1
            if comp == "caption":
                i, j = rng.sample(range(len(self.captions)), 2)
                out.extend([caption_to_example(self.captions[i]), caption_to_example(self.captions[j])])
            else:
                pool = self.pools[comp]
                out.extend(pair_to_examples(pool[rng.randrange(len(pool))]))
        return out, counts

    def next_update(self, stage_index: int) -> tuple[list[dict], dict]:
        """Returns (16 examples, {component: number of draws}); microbatch k = examples[8k:8k+8]."""
        out, counts = self._stage1_update() if stage_index == 0 else self._mixed_update(stage_index)
        assert len(out) == EXAMPLES_PER_UPDATE
        return out, counts

    # ---------------- resume ----------------
    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos,
                "rng_stage2": self.rngs[1].getstate(), "rng_stage3": self.rngs[2].getstate()}

    def load_state_dict(self, s: dict) -> None:
        self.epoch, self.pos, self._perm = int(s["epoch"]), int(s["pos"]), None
        self.rngs[1].setstate(s["rng_stage2"])
        self.rngs[2].setstate(s["rng_stage3"])
