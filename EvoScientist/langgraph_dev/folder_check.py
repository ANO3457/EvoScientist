"""Which folders each access to a served graph is for: a thread's binding.

A thread is bound to the folders its first run worked in: the workspace and,
for a ``--mode=run`` session, the run folder. The server stamps them on every
checkpoint it writes, and the newest stamped checkpoint is the binding
(``sessions.thread_binding``). Only the server writes the stamp: the graph
factory decides the folders here and hands them to the checkpointer, which
ignores whatever folders the request carried.

- A run is for the project its workspace assistant stands for; a run on a
  graph's own assistant, for the project it forwards in ``configurable``. On a
  bound thread it works in the binding, and is refused when it names another
  project or run folder; on a new thread it works in the run folder it
  forwards, or the project root.
- A state update works in its thread's binding and binds nothing.
- A run or update is refused when its project is not the one this server
  serves. Reads are served for any thread and write nothing.

Graphs outside ``FOLDER_GRAPH_IDS`` work in the workspace root, so only the
workspace counts for them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PATH_ERRORS, SessionDirs, Workspace, normalize_path
from ..sessions import FOLDER_GRAPH_IDS
from .assistants import workspace_assistant_id


class RunFolderMismatchError(RuntimeError):
    """An access for folders other than its thread's or this server's."""


def _describe(dirs: SessionDirs) -> str:
    text = f"workspace {dirs.workspace.root}"
    if dirs.run_dir is not None:
        text += f" (run folder {dirs.run_dir})"
    return text


def _read_forwarded(value: Any, what: str) -> Path:
    """A forwarded folder, normalised. Read strictly: a value that is not an
    absolute path (``~`` expanded) or cannot be read is refused."""
    try:
        if not Path(value).expanduser().is_absolute():
            raise ValueError("not an absolute path")
        return normalize_path(value)
    except PATH_ERRORS as exc:
        raise RunFolderMismatchError(
            f"The run forwards a {what} this server cannot read ({value!r}): {exc}"
        ) from exc


def _forwarded_workspace(configurable: dict[str, Any]) -> Workspace | None:
    """The project a run forwards in ``workspace_dir``, or ``None``."""
    workspace_dir = configurable.get("workspace_dir")
    if not workspace_dir:
        return None
    return Workspace(_read_forwarded(workspace_dir, "workspace"))


def _forwarded_run_dir(configurable: dict[str, Any], project: Workspace) -> Path | None:
    """The run folder of *project* a run forwards in ``run_dir``, or ``None``
    when it forwards none or the project root. A folder that is not one of
    the project's run folders is refused.
    """
    run_dir = configurable.get("run_dir")
    if not run_dir:
        return None
    path = _read_forwarded(run_dir, "run folder")
    if path == project.root:
        return None
    if not project.is_run_folder(path):
        raise RunFolderMismatchError(
            f"The run's folder {run_dir} (resolved: {path}) is not a run folder "
            f"of workspace {project.root}: run folders are "
            f"{project.runs_dir}/<name>, inside the workspace."
        )
    return path


def _project_of(
    graph_id: str, configurable: dict[str, Any], served: SessionDirs
) -> Workspace:
    """The project a run is for.

    A workspace assistant stands for its project, whatever ``workspace_dir``
    the run's ``configurable`` holds: the runtime merges into it the stored
    config of the thread, which keeps the first workspace assistant the
    thread ever ran on. A run on a graph's own assistant is for the project
    it forwards, or the served one when it forwards none.
    """
    if configurable.get("assistant_id") == workspace_assistant_id(
        graph_id, served.workspace
    ):
        return served.workspace
    return _forwarded_workspace(configurable) or served.workspace


def resolve_folders(
    graph_id: str,
    access: str,
    configurable: dict[str, Any],
    served: SessionDirs,
    bound: SessionDirs | None,
) -> SessionDirs | None:
    """The folders an access to *graph_id* works in and stamps on its writes;
    ``None`` for an access that writes no stamp, which works in *served*.

    *access* is langgraph-api's access context; *bound* is the thread's
    binding; *served* is the project this server serves (any run folder of
    it). Raises :class:`RunFolderMismatchError` for an access this server
    must not serve.
    """
    if access not in ("threads.create_run", "threads.update"):
        return None
    if bound is not None and bound.workspace != served.workspace:
        raise RunFolderMismatchError(
            f"This conversation works in {_describe(bound)}, which this "
            f"EvoScientist server does not serve."
        )
    if access == "threads.update":
        return bound

    project = _project_of(graph_id, configurable, served)
    if bound is not None and bound.workspace != project:
        raise RunFolderMismatchError(
            f"This conversation works in {_describe(bound)}; it cannot "
            f"continue in workspace {project.root}."
        )
    if project != served.workspace:
        raise RunFolderMismatchError(
            f"This EvoScientist server serves workspace {served.workspace.root}, "
            f"but the run is for workspace {project.root}: the server did not "
            f"follow the session that started this run (another EvoSci session "
            f"moved it, or moving it failed). Restart that session; if another "
            f"session holds the server, stop that one first."
        )
    # Read for every graph, so each refuses a broken value alike; only the
    # graphs that work in a run folder use it.
    forwarded_run_dir = _forwarded_run_dir(configurable, project)
    run_dir = forwarded_run_dir if graph_id in FOLDER_GRAPH_IDS else None
    requested = SessionDirs(project, run_dir)
    if bound is not None and run_dir not in (None, bound.run_dir):
        raise RunFolderMismatchError(
            f"This conversation works in {_describe(bound)}; it cannot "
            f"continue in {_describe(requested)}."
        )
    return bound or requested
