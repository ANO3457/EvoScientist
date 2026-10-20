"""Graph factories: each served graph is built per project and run folder.

langgraph-api calls a registered factory on every access to a graph: runs,
state and history reads, state updates and schema reads. The factory picks
the folders the access is for and returns the graph built for them from a
cache; a graph is built once per (graph, project, run folder) and kept.

Graphs in ``FOLDER_GRAPH_IDS`` work in the session's folder, so their key
includes the run folder of a ``--mode=run`` session; every other graph works
in the project root and is built once per project.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph_sdk.runtime import ServerRuntime

from ..paths import SessionDirs, process_session_dirs
from ..sessions import FOLDER_GRAPH_IDS
from .folder_check import check_run_folders


def _main(dirs: SessionDirs) -> Any:
    from ..EvoScientist import build_main_graph

    return build_main_graph(dirs.workspace, work_dir=dirs.work_dir)


def _async_subagent(name: str) -> Callable[[SessionDirs], Any]:
    def build(dirs: SessionDirs) -> Any:
        from ..subagents._factory import build_async_subagent_graph

        if name in FOLDER_GRAPH_IDS:
            return build_async_subagent_graph(
                name, workspace=dirs.workspace, work_dir=dirs.work_dir
            )
        # Scheduled tasks belong to the workspace, not to one run.
        return build_async_subagent_graph(name, workspace=dirs.workspace)

    return build


def _expert_container(dirs: SessionDirs) -> Any:
    # One graph for every installed expert skill, parameterised per invocation
    # by the ``skill_name`` payload; the loader middleware resolves the skill
    # body at model-call time.
    from ..subagents.expert_container_async import build_expert_container_async_graph

    return build_expert_container_async_graph(
        workspace=dirs.workspace, work_dir=dirs.work_dir
    )


def _memory_worker(source: str) -> Callable[[SessionDirs], Any]:
    def build(dirs: SessionDirs) -> Any:
        from ..memory.agents import build_memory_worker_graph
        from ..memory.types import MemorySourceType

        return build_memory_worker_graph(
            MemorySourceType[source], workspace=dirs.workspace
        )

    return build


def _observation_linker(dirs: SessionDirs) -> Any:
    from ..memory.agents import build_observation_linker_graph

    return build_observation_linker_graph(workspace=dirs.workspace)


def _autoskills(dirs: SessionDirs) -> Any:
    from ..memory.agents import build_autoskills_graph

    return build_autoskills_graph(workspace=dirs.workspace)


def _proactive(dirs: SessionDirs) -> Any:
    # Proactive pushes (#263): one tick of the workspace's proactive cron.
    from ..proactive.graph import build_proactive_graph

    return build_proactive_graph(workspace=dirs.workspace)


# How each graph in ``langgraph.json`` is built. Blocking: builds load MCP and
# touch the disk.
BUILDERS: dict[str, Callable[[SessionDirs], Any]] = {
    "EvoScientist": _main,
    "writing-agent": _async_subagent("writing-agent"),
    "data-analysis-agent": _async_subagent("data-analysis-agent"),
    "scheduler": _async_subagent("scheduler"),
    "expert-container-async": _expert_container,
    "evomemory-subagent-worker": _memory_worker("SUBAGENT"),
    "evomemory-turn-worker": _memory_worker("TURN"),
    "evomemory-observation-linker": _observation_linker,
    "evomemory-autoskills": _autoskills,
    "proactive": _proactive,
}


def _build(graph_id: str, dirs: SessionDirs) -> Any:
    """Build *graph_id* for *dirs*, its checkpoint reads filtered."""
    from langgraph.graph.state import CompiledStateGraph

    from .main_graph import _EvoFilteredGraph

    try:
        graph = BUILDERS[graph_id](dirs)
    except Exception as exc:
        raise RuntimeError(
            f"Graph {graph_id!r} failed to build for {dirs.workspace.root}: {exc}"
        ) from exc
    # Strip private state fields from checkpoint reads (see ``main_graph``).
    if not isinstance(graph, CompiledStateGraph):
        raise TypeError(f"Graph {graph_id!r} is not a compiled graph: {graph!r}")
    graph.__class__ = _EvoFilteredGraph
    return graph


class GraphCache:
    """Built graphs by (graph, project, run folder).

    Builds share process-wide state (the active config, chat model and MCP
    caches, ``os.environ``), so they run one at a time; a graph already built
    is returned without waiting for another's build.
    """

    def __init__(self) -> None:
        self._build_lock = threading.Lock()
        self._graphs: dict[tuple[str, str, str | None], Any] = {}

    def get(self, graph_id: str, dirs: SessionDirs) -> Any:
        """The graph for *dirs*, built on a miss. Blocking."""
        if graph_id not in FOLDER_GRAPH_IDS:
            dirs = SessionDirs(dirs.workspace)
        run_dir = dirs.run_dir.as_posix() if dirs.run_dir is not None else None
        key = (graph_id, dirs.workspace.key, run_dir)
        if (graph := self._graphs.get(key)) is not None:
            return graph
        with self._build_lock:
            if (graph := self._graphs.get(key)) is None:
                graph = self._graphs[key] = _build(graph_id, dirs)
            return graph


GRAPHS = GraphCache()


def prebuild(dirs: SessionDirs) -> None:
    """Build every served graph for *dirs*, so first accesses don't wait. Blocking."""
    for graph_id in BUILDERS:
        GRAPHS.get(graph_id, dirs)


def _graph_for(graph_id: str, config: dict[str, Any], running: bool) -> Any:
    if running:
        check_run_folders(config, works_in_folder=graph_id in FOLDER_GRAPH_IDS)
    return GRAPHS.get(graph_id, process_session_dirs())


def served_graph(graph_id: str) -> Callable[..., Awaitable[Any]]:
    """The factory registered for *graph_id* in ``langgraph.json``.

    Serves the folders this server serves. A run that forwards other folders
    is refused before it executes (``check_run_folders``).
    """

    async def factory(config: dict[str, Any], runtime: ServerRuntime) -> Any:
        # Resolving folders and building touch the disk, which blockbuster
        # refuses on the event loop.
        running = runtime.execution_runtime is not None
        return await asyncio.to_thread(_graph_for, graph_id, config, running)

    return factory
