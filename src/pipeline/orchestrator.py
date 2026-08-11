"""
IEMS – Pipeline Orchestrator (background demo pipeline)
============================================================
Runs the full 9-stage capstone demo pipeline in the background,
respecting real dependencies between stages, so the frontend can poll
per-stage status independently of which tab a viewer happens to be on.

Why this exists (not just "run everything on click")
----------------------------------------------------------
Most stages aren't standalone — they consume an earlier stage's output.
Sizing (grid_search, ga) needs the load + solar curves. Every controller
(rule_based, mpc_normal, mpc_weather, dqn) needs a SPECIFIC panel/battery
size, which only exists once sizing has produced a recommendation. The
final outage-proof stage needs all four controllers' results. Clicking a
downstream tab before its dependencies exist has nothing to compute.

Dependency graph
--------------------
    synthetic_load ─┐
                     ├─► grid_search ─┐
    solar_generation┤                 ├─► [pick recommended candidate]
                     └─► ga ──────────┘
                                              │
                    ┌─────────────┬───────────┼───────────┬─────────┐
                    ▼             ▼           ▼           ▼         │
              rule_based    mpc_normal   mpc_weather     dqn        │
                    └─────────────┴───────────┴───────────┴─────────┘
                                              ▼
                                       outage_proof

Design
------
In-memory job store (dict), fine for a single-process demo — no Redis/
Celery needed at this scale. Each job runs as an asyncio background task
that repeatedly scans for stages whose dependencies are all "done" and
launches them concurrently. A stage's failure marks every stage that
(transitively) depends on it as "blocked" rather than leaving them
pending forever.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable, Optional

from loguru import logger


STAGE_DEPENDENCIES: dict[str, object] = {
    "synthetic_load":   [],
    "solar_generation": [],
    "grid_search":      ["synthetic_load", "solar_generation"],
    "ga":               ["synthetic_load", "solar_generation"],
    # Each controller tab now has an EXPLICIT, single sizing source (no
    # any_of/fallback) — this is a deliberate change from the earlier
    # design. The three controller tabs the demo shows are specifically
    # "Grid Search + Rule-Based", "GA + Rule-Based", and "GA + DQN", so
    # each depends on exactly the one sizing stage it's supposed to use.
    # If that sizing stage fails, the controller tab correctly shows
    # "blocked" rather than silently falling back to the other sizer.
    "rule_based_gs":     ["grid_search"],
    "rule_based_ga":     ["ga"],
    "dqn":               ["ga"],
    # MPC — Normal and MPC — Weather stages are commented out (see
    # stage_functions.py) rather than deleted — not run by the pipeline
    # right now, so no dependency entries for them either.
    # "mpc_normal":      {"any_of": ["grid_search", "ga"]},
    # "mpc_weather":     {"any_of": ["grid_search", "ga"]},
    #
    # outage_proof (displayed as "Final Report" in the UI) needs at least
    # one of the three controller comparisons to exist — not necessarily
    # all three (e.g. DQN legitimately has no trained model on a fresh
    # machine and will fail; the report should still show whichever
    # combos succeeded).
    "outage_proof":     {"any_of": ["rule_based_gs", "rule_based_ga", "dqn"]},
}


VALID_STATUSES = {"pending", "running", "done", "failed", "blocked"}


def _dep_names(dep_spec: object) -> list[str]:
    """Extract the plain list of stage names a dependency spec references,
    regardless of whether it's a plain AND-list or an {'any_of': [...]} group."""
    if isinstance(dep_spec, dict) and "any_of" in dep_spec:
        return list(dep_spec["any_of"])
    return list(dep_spec)


def _dep_satisfied(dep_spec: object, completed: set, failed_or_blocked: set) -> bool:
    """
    True once this stage is READY TO RUN.
    - Plain list (AND semantics): every named stage must be in `completed`.
    - {'any_of': [...]} : every named stage must have reached a TERMINAL
      state (done, failed, or blocked — i.e. we wait for all of them to
      finish trying), AND at least one must be in `completed` (succeeded).
    """
    if isinstance(dep_spec, dict) and "any_of" in dep_spec:
        names = dep_spec["any_of"]
        all_terminal = all(n in completed or n in failed_or_blocked for n in names)
        any_succeeded = any(n in completed for n in names)
        return all_terminal and any_succeeded
    return all(n in completed for n in dep_spec)


def _dep_impossible(dep_spec: object, completed: set, failed_or_blocked: set) -> bool:
    """
    True once this stage can NEVER become ready (should be marked blocked
    immediately rather than the scheduler waiting on it forever).
    - Plain list: any named dependency failed/blocked.
    - {'any_of': [...]} : every named stage has reached a terminal state,
      and NONE of them succeeded.
    """
    if isinstance(dep_spec, dict) and "any_of" in dep_spec:
        names = dep_spec["any_of"]
        all_terminal = all(n in completed or n in failed_or_blocked for n in names)
        any_succeeded = any(n in completed for n in names)
        return all_terminal and not any_succeeded
    return any(n in failed_or_blocked for n in dep_spec)

# In-memory job store: {job_id: {"stages": {...}, "created_at": float, "params": {...}}}
_JOBS: dict[str, dict] = {}


def _transitive_dependents(stage_name: str) -> set[str]:
    """Every stage that depends on `stage_name`, directly or indirectly —
    used to propagate a failure/block downstream instead of leaving
    dependents stuck at 'pending' forever."""
    dependents = set()
    changed = True
    while changed:
        changed = False
        for name, deps in STAGE_DEPENDENCIES.items():
            if name in dependents:
                continue
            if any(d == stage_name or d in dependents for d in _dep_names(deps)):
                dependents.add(name)
                changed = True
    return dependents


def create_job(params: dict) -> str:
    """Register a new job and return its ID. Does NOT start execution —
    call run_pipeline_job() (typically via asyncio.create_task) separately."""
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "stages": {
            name: {"status": "pending", "result": None, "error": None, "started_at": None, "finished_at": None}
            for name in STAGE_DEPENDENCIES
        },
        "created_at": time.time(),
        "params": params,
    }
    return job_id


def get_job_status(job_id: str) -> Optional[dict]:
    """Returns the full status dict for a job, or None if unknown."""
    job = _JOBS.get(job_id)
    if job is None:
        return None
    return {
        "job_id": job_id,
        "created_at": job["created_at"],
        "stages": {
            name: {k: v for k, v in stage.items() if k != "result"} | {"has_result": stage["result"] is not None}
            for name, stage in job["stages"].items()
        },
    }


def get_stage_result(job_id: str, stage_name: str) -> Optional[dict]:
    """Returns just one stage's result payload (the actual chart/table
    data), or None if the job/stage doesn't exist or hasn't produced a
    result yet."""
    job = _JOBS.get(job_id)
    if job is None or stage_name not in job["stages"]:
        return None
    return job["stages"][stage_name]["result"]


async def run_pipeline_job(job_id: str, stage_functions: dict[str, Callable], poll_interval_sec: float = 0.5) -> None:
    """
    The background orchestrator loop. Launches each stage as soon as all
    of its dependencies are 'done', running independent stages
    concurrently. Call via `asyncio.create_task(run_pipeline_job(...))`
    right after `create_job()` so it doesn't block the HTTP response that
    returns the job_id.

    Parameters
    ----------
    stage_functions : dict[str, Callable]
        Maps stage name -> an async callable `(params, get_upstream) -> dict`.
        `get_upstream(name)` lets a stage read an already-completed
        upstream stage's result (e.g. rule_based reads grid_search's
        chosen candidate).
    """
    job = _JOBS[job_id]
    stages = job["stages"]
    params = job["params"]

    def get_upstream(name: str):
        return stages[name]["result"]

    running_tasks: dict[str, asyncio.Task] = {}
    completed: set[str] = set()
    failed_or_blocked: set[str] = set()

    async def run_one_stage(name: str):
        stages[name]["status"] = "running"
        stages[name]["started_at"] = time.time()
        try:
            fn = stage_functions[name]
            result = await fn(params, get_upstream)
            stages[name]["result"] = result
            stages[name]["status"] = "done"
        except Exception as e:
            logger.warning(f"[pipeline {job_id}] stage '{name}' failed: {e}")
            stages[name]["status"] = "failed"
            stages[name]["error"] = str(e)
        finally:
            stages[name]["finished_at"] = time.time()

    total_stages = len(STAGE_DEPENDENCIES)
    while len(completed) + len(failed_or_blocked) < total_stages:
        # Launch any stage whose dependencies are satisfied and hasn't started
        for name, deps in STAGE_DEPENDENCIES.items():
            if name in running_tasks or name in completed or name in failed_or_blocked:
                continue
            if _dep_satisfied(deps, completed, failed_or_blocked):
                running_tasks[name] = asyncio.create_task(run_one_stage(name))
            elif _dep_impossible(deps, completed, failed_or_blocked):
                # This stage can never run — mark it blocked immediately
                # rather than waiting forever for a dependency that will
                # never complete (or, for {'any_of': [...]}, waiting for a
                # group where every member has already failed).
                stages[name]["status"] = "blocked"
                stages[name]["error"] = f"Blocked: dependency requirement not satisfiable ({deps})"
                failed_or_blocked.add(name)

        # Reap finished tasks
        for name, task in list(running_tasks.items()):
            if task.done():
                del running_tasks[name]
                if stages[name]["status"] == "done":
                    completed.add(name)
                else:
                    failed_or_blocked.add(name)

        await asyncio.sleep(poll_interval_sec)

    logger.info(
        f"[pipeline {job_id}] complete: {len(completed)} done, "
        f"{len(failed_or_blocked)} failed/blocked"
    )