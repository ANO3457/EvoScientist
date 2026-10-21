"""What the server needs for a project before clients use it: its registration.

Registration (``POST /api/workspaces``) is a setup barrier: the project's
threads are restored and its assistants created, and only then is it
recorded as registered, which is what the factories serve while its folder
exists. Every
step can run again, so registering a registered project repeats them. The
server's start sets every registered project up again: a fresh runtime store
has none of their assistants. Graphs are built when first needed; the chat
graph, which reads use, starts building in the background. Setup runs inside
the server, through the runtime's own ops.
"""

from __future__ import annotations

import asyncio
import logging

from ..paths import SessionDirs, Workspace
from ..sessions import AGENT_NAME, restore_workspace_threads
from . import assistants, registry
from .factories import GRAPHS

logger = logging.getLogger(__name__)

_lock: asyncio.Lock | None = None
# Background builds, kept until done so they are not garbage-collected, and
# run one at a time: builds take turns anyway (``GraphCache``), and one
# waiting here holds no worker thread.
_warm_ups: set[asyncio.Task] = set()
_warm_lock: asyncio.Lock | None = None


class FolderMissingError(RuntimeError):
    """A project's folder does not exist."""


def registration_lock() -> asyncio.Lock:
    """Held while a project is registered or unregistered, one at a time."""
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def _set_up(workspace: Workspace, *, restore: bool) -> None:
    if not await asyncio.to_thread(workspace.root.is_dir):
        raise FolderMissingError(f"The folder {workspace.root} does not exist.")
    if restore and not await restore_workspace_threads(workspace.key):
        raise RuntimeError(f"Could not restore the threads of {workspace.root}.")
    try:
        await assistants.ensure_workspace_assistants(workspace)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create the workspace assistants for {workspace.root}: {exc}"
        ) from exc
    await asyncio.to_thread(registry.REGISTRY.set_registered, workspace.key, True)


def _warmed(task: asyncio.Task) -> None:
    _warm_ups.discard(task)
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("Could not build the chat graph ahead of use: %s", exc)


async def _warm(workspace: Workspace) -> None:
    global _warm_lock
    if _warm_lock is None:
        _warm_lock = asyncio.Lock()
    async with _warm_lock:
        await asyncio.to_thread(GRAPHS.get, AGENT_NAME, SessionDirs(workspace))


def warm_up(workspace: Workspace) -> None:
    """Start building *workspace*'s chat graph, the one reads use, in the
    background, so a client's first read does not wait for it. A failure is
    logged; the read builds it again."""
    task = asyncio.create_task(_warm(workspace))
    _warm_ups.add(task)
    task.add_done_callback(_warmed)


async def register_workspace(dirs: SessionDirs) -> None:
    """Set *dirs*' project up and record it as registered; the caller holds
    :func:`registration_lock`.

    Raises :class:`FolderMissingError` when the project's folder does not
    exist, and ``RuntimeError`` when its threads cannot be restored or its
    assistants created; a project that was not registered then stays so.
    """
    await _set_up(dirs.workspace, restore=True)
    warm_up(dirs.workspace)


async def set_up_workspace(dirs: SessionDirs) -> None:
    """At the server's start, before it serves requests: set every registered
    project up again, then register *dirs*, the folder it was started in.

    The checkpointer has restored every known project's threads by then. A
    project that cannot be set up is logged and stays registered; *dirs* must
    be served, so its failure is raised.
    """
    for key in await asyncio.to_thread(registry.REGISTRY.registered):
        if key == dirs.workspace.key:
            continue
        try:
            other = await asyncio.to_thread(Workspace, key)
            await _set_up(other, restore=False)
        except FolderMissingError:
            logger.warning("Not serving %s: its folder is missing.", key)
        except Exception:
            logger.exception("Could not set %s up.", key)
    await _set_up(dirs.workspace, restore=False)
    warm_up(dirs.workspace)
