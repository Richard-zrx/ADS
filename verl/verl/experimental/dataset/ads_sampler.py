# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Paper-faithful Adaptive Data Scheduling (ADS) sampler.

Implements the dual-level data schedule from "Learning at the Right Pace:
Adaptive Data Scheduling Improves LLM Reinforcement Learning" (ADS):

1. Inter-cluster distribution (paper Eq. 4 + Appendix D):
   Each cluster maintains a running success rate ``r_k`` (fraction of correct
   rollouts among all rollouts generated from that cluster the last time it was
   sampled; carried forward otherwise). The target probability is
   ``p_hat_k = r_k / sum_j r_j`` and the actual sampling distribution is updated
   with exponential smoothing ``p_k <- (1 - alpha) * p_k + alpha * p_hat_k`` with
   ``alpha = 0.3``.

2. Intra-cluster scheduling / boundary mini-cluster (paper Eq. 5, 6, 7):
   Each cluster keeps a boundary mini-cluster ``M_k`` of ``B`` samples, indexed
   by *position* in the cluster's offline difficulty-sorted order (easy -> hard,
   ascending NLL of the reference solution). ``M_k`` is initialised with the
   easiest ``B`` samples. After every step, for each sampled candidate with
   empirical success rate ``rho``:

       rho in [0.5 - eps, 0.5 + eps]  -> keep        (policy-boundary)
       rho < 0.5 - eps                -> too hard     -> move to easier neighbour
       rho > 0.5 + eps                -> too easy     -> move to harder neighbour

   In a difficulty-sorted array the nearest easier / harder sample is simply the
   adjacent position (-1 / +1), with collision/boundary handling so the
   mini-cluster keeps exactly ``B`` distinct samples.

3. Per-step batch: ``active_clusters`` clusters are drawn (without replacement)
   according to the inter-cluster distribution and **all** ``B`` samples of each
   selected cluster's mini-cluster are used. With the paper's defaults this gives
   ``4 * 32 = 128`` policy-boundary samples per step.

4. Epoch reset (Appendix D): the inter-cluster distribution and every boundary
   mini-cluster are reinitialised at each epoch boundary (``|D| / batch_size``
   steps) so that stale rollout statistics from earlier policy states do not
   dominate later scheduling decisions.

This class is intentionally self-contained (it does not subclass the
window/frontier sampler) so the two scheduling strategies can be compared
side by side.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sized
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


@dataclass
class ADSClusterState:
    cluster_id: int
    cluster_size: int
    # Running cluster-level success rate r_k (carried forward when not sampled).
    success_rate: float = 0.5
    # Inter-cluster sampling probability p_k.
    prob: float = 0.0
    # Boundary mini-cluster: set of positions into the difficulty-sorted array.
    mini_positions: set[int] = field(default_factory=set)


class ADSMiniClusterSampler(AbstractCurriculumSampler):
    """Paper-faithful ADS sampler (boundary mini-cluster + adaptive inter-cluster).

    Config is read from ``data_config.ads_sampler``. Required dataset columns:
    ``cluster_id``, ``sample_id``, ``rank_in_cluster``, ``cluster_size`` (the
    same offline products produced by ``sort_clusters_by_difficulty.py``).
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        cfg = data_config.get("ads_sampler", {})

        self.cluster_key = str(cfg.get("cluster_key", "cluster_id"))
        self.sample_id_key = str(cfg.get("sample_id_key", "sample_id"))
        self.rank_key = str(cfg.get("rank_key", "rank_in_cluster"))
        self.cluster_size_key = str(cfg.get("cluster_size_key", "cluster_size"))
        self.seed = int(cfg.get("seed", 42))
        self.prob_snapshot_log_interval = max(1, int(cfg.get("prob_snapshot_log_interval", 25)))

        # Core ADS hyper-parameters (paper defaults).
        self.active_clusters = int(cfg.get("active_clusters", 4))
        self.mini_cluster_size = int(cfg.get("mini_cluster_size", 32))
        self.boundary_eps = float(cfg.get("boundary_eps", 0.17))
        self.alpha = float(cfg.get("alpha", 0.3))
        # Initial / fallback running success rate so the initial target probability
        # is uniform (r_init equal for all clusters -> p_hat_k = 1/K) before any
        # rollout feedback is available.
        self.r_init = float(cfg.get("r_init", 0.5))

        self.band_low = 0.5 - self.boundary_eps
        self.band_high = 0.5 + self.boundary_eps

        self.batch_size = self.active_clusters * self.mini_cluster_size

        self.rng = np.random.default_rng(self.seed)
        self.length = len(data_source)

        # Epoch reset cadence. ``epoch_reset_steps <= 0`` disables the reset.
        # Default to one pass over the dataset (|D| / batch_size), matching the
        # paper's epoch-boundary reset.
        default_epoch_steps = max(1, self.length // max(1, self.batch_size))
        self.epoch_reset_steps = int(cfg.get("epoch_reset_steps", default_epoch_steps))

        # Built by _initialize_from_dataset.
        self.cluster_to_sorted_indices: dict[int, np.ndarray] = {}
        self.row_idx_to_position: dict[int, int] = {}
        self.sample_id_to_cluster: dict[Any, int] = {}
        self.sample_id_to_row_idx: dict[Any, int] = {}
        self.index_to_cluster: dict[int, int] = {}
        self.cluster_states: dict[int, ADSClusterState] = {}
        self.cluster_ids: list[int] = []

        self._global_update_steps = 0
        self._steps_in_epoch = 0
        self._last_batch_band_counts: dict[str, int] = {"too_hard": 0, "keep": 0, "too_easy": 0}

        self._initialize_from_dataset()
        self._init_schedule_state()

        print(
            f"[ADSMiniClusterSampler] init n={self.length} clusters={len(self.cluster_ids)} "
            f"active={self.active_clusters} B={self.mini_cluster_size} "
            f"eps={self.boundary_eps} band=[{self.band_low:.2f},{self.band_high:.2f}] "
            f"alpha={self.alpha} batch={self.batch_size} epoch_reset_steps={self.epoch_reset_steps}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_from_dataset(self) -> None:
        if not hasattr(self.data_source, "dataframe"):
            raise AttributeError("ADSMiniClusterSampler expects the dataset to expose `.dataframe`.")

        dataframe = self.data_source.dataframe
        cluster_to_rank_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)

        for row_idx in range(len(dataframe)):
            row = dataframe[row_idx]
            if self.cluster_key not in row:
                raise KeyError(f"Dataset row is missing required cluster key: {self.cluster_key}")

            cluster_id = int(row[self.cluster_key])
            rank = int(row[self.rank_key])

            sample_id = row.get(self.sample_id_key)
            if sample_id is None and isinstance(row.get("extra_info"), dict):
                sample_id = row["extra_info"].get("index")
            if sample_id is None:
                sample_id = row_idx

            cluster_to_rank_rows[cluster_id].append((rank, row_idx))
            self.index_to_cluster[row_idx] = cluster_id
            self.sample_id_to_cluster[sample_id] = cluster_id
            self.sample_id_to_row_idx[sample_id] = row_idx

        if not cluster_to_rank_rows:
            raise ValueError("No cluster ids found in dataset for ADSMiniClusterSampler")

        self.cluster_ids = sorted(cluster_to_rank_rows)
        for cluster_id in self.cluster_ids:
            rank_rows = cluster_to_rank_rows[cluster_id]
            rank_rows.sort(key=lambda x: x[0])  # easy -> hard by rank_in_cluster
            indices = np.array([r[1] for r in rank_rows], dtype=np.int64)
            self.cluster_to_sorted_indices[cluster_id] = indices
            for position, row_idx in enumerate(indices.tolist()):
                self.row_idx_to_position[int(row_idx)] = position
            self.cluster_states[cluster_id] = ADSClusterState(
                cluster_id=cluster_id,
                cluster_size=len(indices),
            )

        if len(self.cluster_ids) < self.active_clusters:
            raise ValueError(
                f"active_clusters={self.active_clusters} exceeds number of clusters "
                f"{len(self.cluster_ids)}."
            )

    def _init_schedule_state(self) -> None:
        """Reset inter-cluster distribution (uniform) and mini-clusters (easiest B)."""
        k = len(self.cluster_ids)
        uniform = 1.0 / float(k)
        for cluster_id in self.cluster_ids:
            state = self.cluster_states[cluster_id]
            state.success_rate = self.r_init
            state.prob = uniform
            b = min(self.mini_cluster_size, state.cluster_size)
            state.mini_positions = set(range(b))

    def __len__(self) -> int:
        return self.length

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[int]:
        while True:
            probs = np.array(
                [self.cluster_states[cid].prob for cid in self.cluster_ids], dtype=np.float64
            )
            total = probs.sum()
            if not np.isfinite(total) or total <= 0.0:
                probs = np.full(len(self.cluster_ids), 1.0 / len(self.cluster_ids))
            else:
                probs = probs / total

            chosen_positions = self.rng.choice(
                len(self.cluster_ids), size=self.active_clusters, replace=False, p=probs
            )
            active_cluster_ids = [self.cluster_ids[int(i)] for i in chosen_positions]

            batch_indices: list[int] = []
            for cid in active_cluster_ids:
                state = self.cluster_states[cid]
                sorted_indices = self.cluster_to_sorted_indices[cid]
                positions = sorted(state.mini_positions)
                for pos in positions:
                    batch_indices.append(int(sorted_indices[pos]))

            for idx in batch_indices:
                yield idx

    def blacklist_samples(self, sample_ids: list) -> int:
        """No-op for ADS.

        The trainer's zero-accuracy group filter calls this for prompts whose
        rollouts are all incorrect. ADS does not blacklist such prompts: the
        intra-cluster schedule already responds to a too-hard sample by moving
        the mini-cluster toward an easier neighbour, so the sample must stay
        in the cluster pool. (All-wrong groups contribute zero GRPO advantage,
        so excluding them from the gradient step is harmless either way.)
        """
        return 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, batch: DataProto) -> None:
        self.update_main(batch=batch, global_step=self._global_update_steps + 1)

    def update_main(self, batch: DataProto, global_step: int) -> None:
        """Update boundary mini-clusters (per-sample) and inter-cluster distribution."""
        sample_ids = self._resolve_sample_ids(batch)
        correctness = self._resolve_correctness(batch)
        uids = self._to_numpy(batch.non_tensor_batch["uid"])

        # Group rollout rows by prompt (uid) -> per-prompt success rate rho.
        groups: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids.tolist()):
            groups[uid].append(idx)

        # cluster_id -> per-prompt rho list (for the running success rate r_k)
        cluster_rhos: dict[int, list[float]] = defaultdict(list)
        # cluster_id -> {position: direction} where direction in {-1, 0, +1}
        cluster_directions: dict[int, dict[int, int]] = defaultdict(dict)

        band_counts = {"too_hard": 0, "keep": 0, "too_easy": 0}

        for positions in groups.values():
            pos_arr = np.array(positions)
            sample_id = sample_ids[positions[0]]
            row_idx = self.sample_id_to_row_idx.get(sample_id)
            if row_idx is None:
                # Replay / out-of-pool sample: contributes to the gradient only.
                continue
            cluster_id = self.index_to_cluster[row_idx]
            position = self.row_idx_to_position[row_idx]

            rho = float(np.mean(correctness[pos_arr]))
            cluster_rhos[cluster_id].append(rho)

            # Only adapt positions that are actually in the current mini-cluster.
            if position not in self.cluster_states[cluster_id].mini_positions:
                continue

            if rho < self.band_low:
                cluster_directions[cluster_id][position] = -1  # too hard -> easier
                band_counts["too_hard"] += 1
            elif rho > self.band_high:
                cluster_directions[cluster_id][position] = +1  # too easy -> harder
                band_counts["too_easy"] += 1
            else:
                cluster_directions[cluster_id][position] = 0  # policy-boundary -> keep
                band_counts["keep"] += 1

        # 1) Update running cluster-level success rates for sampled clusters.
        for cluster_id, rhos in cluster_rhos.items():
            if cluster_id in self.cluster_states and rhos:
                self.cluster_states[cluster_id].success_rate = float(np.mean(rhos))

        # 2) Adapt each sampled cluster's boundary mini-cluster (per sample).
        for cluster_id, directions in cluster_directions.items():
            self._update_mini_positions(cluster_id, directions)

        # 3) Update the inter-cluster distribution (Eq. 4 + exponential smoothing).
        self._update_inter_cluster_distribution()

        self._last_batch_band_counts = band_counts
        self._global_update_steps = int(global_step)

        # 4) Epoch-boundary reset of the whole schedule state.
        self._steps_in_epoch += 1
        if self.epoch_reset_steps > 0 and self._steps_in_epoch >= self.epoch_reset_steps:
            self._init_schedule_state()
            self._steps_in_epoch = 0

    def _update_inter_cluster_distribution(self) -> None:
        rates = np.array(
            [self.cluster_states[cid].success_rate for cid in self.cluster_ids], dtype=np.float64
        )
        rate_sum = float(rates.sum())
        if rate_sum <= 0.0:
            target = np.full(len(self.cluster_ids), 1.0 / len(self.cluster_ids))
        else:
            target = rates / rate_sum

        for cid, tgt in zip(self.cluster_ids, target.tolist(), strict=True):
            state = self.cluster_states[cid]
            state.prob = (1.0 - self.alpha) * state.prob + self.alpha * float(tgt)

        # Renormalise for numerical safety (smoothing two distributions keeps the
        # sum at 1 in exact arithmetic; this guards against float drift).
        prob_sum = sum(self.cluster_states[cid].prob for cid in self.cluster_ids)
        if prob_sum > 0.0:
            for cid in self.cluster_ids:
                self.cluster_states[cid].prob /= prob_sum

    def _update_mini_positions(self, cluster_id: int, directions: dict[int, int]) -> None:
        """Apply the per-sample replacement rule (paper Eq. 6/7) to one mini-cluster.

        ``directions`` maps a current position to -1 (move to easier neighbour),
        0 (keep), or +1 (move to harder neighbour). Positions not present in
        ``directions`` are treated as "keep". Collisions and array bounds are
        resolved by taking the nearest free position (preferring the requested
        direction) so the mini-cluster keeps exactly ``B`` distinct samples.
        """
        state = self.cluster_states[cluster_id]
        size = state.cluster_size
        old_positions = sorted(state.mini_positions)

        # When the mini-cluster already covers the whole cluster there is nowhere
        # to move; leave it unchanged.
        if size <= len(old_positions):
            return

        new_positions: set[int] = set()
        movers: list[tuple[int, int]] = []

        # Stayers (and any positions with no recorded feedback) hold their slot first.
        for pos in old_positions:
            d = directions.get(pos, 0)
            if d == 0:
                new_positions.add(pos)
            else:
                movers.append((pos, d))

        # Resolve easier-movers from the easy end and harder-movers from the hard
        # end so adjacent movers cascade deterministically instead of fighting for
        # the same slot.
        easier = sorted(p for p, d in movers if d < 0)
        harder = sorted((p for p, d in movers if d > 0), reverse=True)

        for pos in easier:
            chosen = self._nearest_free(pos - 1, -1, new_positions, size, fallback=pos)
            new_positions.add(chosen)
        for pos in harder:
            chosen = self._nearest_free(pos + 1, +1, new_positions, size, fallback=pos)
            new_positions.add(chosen)

        state.mini_positions = new_positions

    @staticmethod
    def _nearest_free(
        desired: int, direction: int, occupied: set[int], size: int, fallback: int
    ) -> int:
        """Nearest position to ``desired`` not in ``occupied``, within [0, size).

        Search prefers ``direction`` (the requested move) and expands outward.
        ``fallback`` (the sample's own current position) is returned only if the
        cluster is fully occupied, which the caller already guards against.
        """
        d = min(max(desired, 0), size - 1)
        if d not in occupied:
            return d
        for radius in range(1, size):
            for cand in (d + direction * radius, d - direction * radius):
                if 0 <= cand < size and cand not in occupied:
                    return cand
        return fallback

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def get_logging_metrics(self, global_step: int | None = None) -> dict[str, Any]:
        step = self._global_update_steps if global_step is None else int(global_step)
        probs = np.array([self.cluster_states[cid].prob for cid in self.cluster_ids], dtype=np.float64)
        rates = np.array(
            [self.cluster_states[cid].success_rate for cid in self.cluster_ids], dtype=np.float64
        )
        mean_pos_frac = float(
            np.mean(
                [
                    (np.mean(sorted(s.mini_positions)) / max(s.cluster_size - 1, 1))
                    for s in self.cluster_states.values()
                ]
            )
        )

        metrics: dict[str, Any] = {
            "ads/prob_max": float(probs.max()) if probs.size else 0.0,
            "ads/prob_min": float(probs.min()) if probs.size else 0.0,
            "ads/prob_entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
            "ads/success_rate_mean": float(rates.mean()) if rates.size else 0.0,
            "ads/mini_position_frac_mean": mean_pos_frac,
            "ads/batch_too_hard_count": float(self._last_batch_band_counts.get("too_hard", 0)),
            "ads/batch_keep_count": float(self._last_batch_band_counts.get("keep", 0)),
            "ads/batch_too_easy_count": float(self._last_batch_band_counts.get("too_easy", 0)),
            "ads/steps_in_epoch": float(self._steps_in_epoch),
        }

        if self._should_log_prob_snapshot(step):
            metrics["cluster/prob_snapshot_step"] = float(step)
            for cid in self.cluster_ids:
                metrics[f"cluster/prob_snapshot/cluster_{int(cid)}"] = float(self.cluster_states[cid].prob)

        return metrics

    def _should_log_prob_snapshot(self, global_step: int) -> bool:
        if global_step <= 0:
            return True
        return global_step % self.prob_snapshot_log_interval == 0

    def get_prob_snapshot_payload(self, global_step: int) -> dict[str, Any] | None:
        step = int(global_step)
        if not self._should_log_prob_snapshot(step):
            return None
        points = [
            {"cluster_id": int(cid), "prob": float(self.cluster_states[cid].prob)}
            for cid in self.cluster_ids
        ]
        return {"step": step, "points": points}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        serialized = {
            cid: {
                "cluster_id": s.cluster_id,
                "cluster_size": s.cluster_size,
                "success_rate": s.success_rate,
                "prob": s.prob,
                "mini_positions": sorted(s.mini_positions),
            }
            for cid, s in self.cluster_states.items()
        }
        return {
            "cluster_states": serialized,
            "rng_state": self.rng.bit_generator.state,
            "global_update_steps": int(self._global_update_steps),
            "steps_in_epoch": int(self._steps_in_epoch),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        loaded = state_dict.get("cluster_states", {})
        for cluster_id in self.cluster_ids:
            entry = loaded.get(cluster_id, loaded.get(str(cluster_id)))
            if entry is None:
                continue
            state = self.cluster_states[cluster_id]
            state.success_rate = float(entry["success_rate"])
            state.prob = float(entry["prob"])
            state.mini_positions = set(int(p) for p in entry["mini_positions"])

        rng_state = state_dict.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

        self._global_update_steps = int(state_dict.get("global_update_steps", self._global_update_steps))
        self._steps_in_epoch = int(state_dict.get("steps_in_epoch", self._steps_in_epoch))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(values: Any) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values
        if hasattr(values, "detach"):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    def _resolve_sample_ids(self, batch: DataProto) -> np.ndarray:
        if self.sample_id_key in batch.non_tensor_batch:
            return self._to_numpy(batch.non_tensor_batch[self.sample_id_key])
        if "index" in batch.non_tensor_batch:
            return self._to_numpy(batch.non_tensor_batch["index"])

        extra_info_values = batch.non_tensor_batch.get("extra_info")
        if extra_info_values is not None:
            extra_info_array = self._to_numpy(extra_info_values)
            extracted: list[Any] = []
            ok = True
            for item in extra_info_array.tolist():
                if not isinstance(item, dict):
                    ok = False
                    break
                value = item.get(self.sample_id_key, item.get("index"))
                if value is None:
                    ok = False
                    break
                extracted.append(value)
            if ok and len(extracted) == len(extra_info_array):
                return np.asarray(extracted, dtype=object)

        raise KeyError(
            f"Batch is missing `{self.sample_id_key}` and no `index`/`extra_info` fallback is available."
        )

    def _resolve_correctness(self, batch: DataProto) -> np.ndarray:
        preferred_binary_keys = ("acc", "is_correct", "correct")
        for key in preferred_binary_keys:
            if key in batch.non_tensor_batch:
                values = self._to_numpy(batch.non_tensor_batch[key]).astype(np.float32, copy=False)
                values = np.nan_to_num(values, nan=0.0)
                return np.clip(values, 0.0, 1.0)

        reward_like_keys = ("correctness_score", "score", "reward")
        for key in reward_like_keys:
            if key in batch.non_tensor_batch:
                values = self._to_numpy(batch.non_tensor_batch[key]).astype(np.float32, copy=False)
                values = np.nan_to_num(values, nan=0.0)
                return (values > 0.0).astype(np.float32)

        tensor_keys = ("token_level_scores", "token_level_rewards", "rm_scores")
        for key in tensor_keys:
            if key in batch.batch:
                values = self._to_numpy(batch.batch[key]).astype(np.float32, copy=False)
                if values.ndim > 1:
                    reduce_axes = tuple(range(1, values.ndim))
                    values = values.sum(axis=reduce_axes)
                values = np.nan_to_num(values, nan=0.0)
                return (values > 0.0).astype(np.float32)

        raise KeyError(
            "Unable to derive correctness signal: missing `acc` and no reward-like fallback keys were found."
        )
