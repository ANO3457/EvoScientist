"""A project's crons, read and changed inside the server through the
runtime's own store.

A cron belongs to the project of the workspace assistant it runs on. User
crons (scheduled tasks) are the user's; the AutoSkills and proactive crons are
the system's, recreated by their reconciles when the project is used again.
"""

from __future__ import annotations

from typing import Any

from ..cron.schedule import SCHEDULED_RUN_KIND, belongs_to, owned
from ..memory.autoskills.schedule import AUTOSKILL_RUN_KIND
from ..paths import Workspace
from ..proactive.cron import PROACTIVE_RUN_KIND

SYSTEM_RUN_KINDS = frozenset({AUTOSKILL_RUN_KIND, PROACTIVE_RUN_KIND})

# Every cron the store holds fits in one search.
_SEARCH_LIMIT = 100_000


def _run_kind(cron: dict[str, Any]) -> object:
    return (cron.get("metadata") or {}).get("run_kind")


async def project_crons(workspace: Workspace) -> list[dict[str, Any]]:
    """The crons that run on one of *workspace*'s assistants."""
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Crons

    async with connect() as conn:
        rows, _count = await Crons.search(
            conn,
            assistant_id=None,
            thread_id=None,
            enabled=None,
            limit=_SEARCH_LIMIT,
            offset=0,
        )
        return owned([cron async for cron in rows], workspace)


def enabled_user_crons(crons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The enabled scheduled tasks among *crons*."""
    return [
        cron
        for cron in crons
        if cron.get("enabled") and _run_kind(cron) == SCHEDULED_RUN_KIND
    ]


def system_crons(crons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The AutoSkills and proactive crons among *crons*."""
    return [cron for cron in crons if _run_kind(cron) in SYSTEM_RUN_KINDS]


async def _delete(conn: Any, cron_id: Any) -> None:
    from langgraph_runtime.ops import Crons

    async for _deleted in await Crons.delete(conn, cron_id):
        pass


async def delete_crons(cron_ids: list[Any]) -> None:
    from langgraph_runtime.database import connect

    async with connect() as conn:
        for cron_id in cron_ids:
            await _delete(conn, cron_id)


async def drop_system_cron(cron_id: str, workspace: Workspace) -> None:
    """Delete *cron_id* if it is one of *workspace*'s system crons, which fired
    into its missing folder; its reconcile recreates it.

    User crons are left as they are: each fire is refused until the folder is
    back, and leaves nothing in the store (they run on threads of their own,
    deleted after the run).
    """
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Crons

    async with connect() as conn:
        found = [cron async for cron in await Crons.get(conn, cron_id)]
        if not found:
            return
        (cron,) = found
        if belongs_to(cron, workspace) and _run_kind(cron) in SYSTEM_RUN_KINDS:
            await _delete(conn, cron["cron_id"])
