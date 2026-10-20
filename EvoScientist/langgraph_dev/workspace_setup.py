"""What the server needs for a project before clients use it."""

from __future__ import annotations

import asyncio

from ..paths import SessionDirs
from . import assistants
from .factories import prebuild


async def set_up_workspace(dirs: SessionDirs) -> None:
    """Create *dirs*' workspace assistants and build its graphs.

    Runs inside the server process. The assistants are lost with the
    runtime's store, so this runs at every start; the graphs are built now
    rather than at a client's first run or read. Without either, most runs
    would fail with errors that hide the cause, so a failure is raised.
    """
    try:
        await assistants.ensure_workspace_assistants(dirs.workspace)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create the workspace assistants for "
            f"{dirs.workspace.root}: {exc}"
        ) from exc
    await asyncio.to_thread(prebuild, dirs)
