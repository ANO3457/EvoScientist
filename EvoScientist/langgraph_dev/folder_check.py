"""Refuse runs meant for other folders than the ones this server serves.

Until the server serves several workspaces, it is pinned to one workspace and,
for a ``--mode=run`` session, one run folder. Child runs forward the folders
they belong to in ``configurable``; each graph's factory (``factories``)
compares them with the server's before the run executes, so a run meant for
another folder fails instead of working in this one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import SessionDirs, Workspace, process_session_dirs


class RunFolderMismatchError(RuntimeError):
    """A run forwarded folders this server does not serve."""


def _describe(dirs: SessionDirs, *, with_run_dir: bool) -> str:
    text = f"workspace {dirs.workspace.root}"
    if with_run_dir and dirs.run_dir is not None:
        text += f" (run folder {dirs.run_dir})"
    return text


def _forwarded_dirs(configurable: dict[str, Any]) -> SessionDirs | None:
    """The folders a run forwards, or ``None`` when it forwards none.

    Read strictly: a value that is present but cannot be read is refused
    rather than treated as not forwarded.
    """
    workspace_dir = configurable.get("workspace_dir")
    if not workspace_dir:
        return None
    run_dir = configurable.get("run_dir")
    try:
        return SessionDirs(Workspace(workspace_dir), Path(run_dir) if run_dir else None)
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise RunFolderMismatchError(
            f"The run forwards folders this server cannot read "
            f"({workspace_dir!r}, {run_dir!r}): {exc}"
        ) from exc


def check_run_folders(config: dict[str, Any], *, works_in_folder: bool) -> None:
    """Raise if the run's forwarded folders differ from the server's.

    A run that forwards no ``workspace_dir`` is not checked. For a graph that
    does not work in a folder, only the workspace counts. Resolves paths, so
    call it off the event loop.
    """
    requested = _forwarded_dirs(config.get("configurable") or {})
    if requested is None:
        return
    served = process_session_dirs()
    if requested.workspace == served.workspace and (
        not works_in_folder or requested.run_dir == served.run_dir
    ):
        return
    raise RunFolderMismatchError(
        f"This EvoScientist server serves "
        f"{_describe(served, with_run_dir=works_in_folder)}, but the run is for "
        f"{_describe(requested, with_run_dir=works_in_folder)}: the server did "
        f"not follow the session that started this run (another EvoSci session "
        f"moved it, or moving it failed). Restart that session; if another "
        f"session holds the server, stop that one first."
    )
