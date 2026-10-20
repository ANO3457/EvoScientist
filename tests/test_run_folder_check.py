"""Child runs carry their folders, and the server serves each thread in its own."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.langgraph_dev import factories
from EvoScientist.langgraph_dev.assistants import workspace_assistant_id
from EvoScientist.langgraph_dev.factories import served_graph
from EvoScientist.langgraph_dev.folder_check import (
    RunFolderMismatchError,
    resolve_folders,
)
from EvoScientist.paths import SessionDirs, Workspace
from EvoScientist.sessions import FOLDER_GRAPH_IDS, RESOLVED_FOLDERS

RUN = "20260930_120000"
RUN_ACCESS = "threads.create_run"


@pytest.fixture
def served(workspace) -> SessionDirs:
    """A server pinned to a run folder of ``workspace``."""
    return SessionDirs(workspace, workspace.runs_dir / RUN)


def _config(dirs: SessionDirs | None) -> dict:
    return {"configurable": dirs.metadata() if dirs is not None else {}}


def _run(served, configurable, *, bound=None, graph_id="writing-agent"):
    return resolve_folders(graph_id, RUN_ACCESS, configurable, served, bound)


# ---------------------------------------------------------------------------
# The server's check
# ---------------------------------------------------------------------------


def test_runs_without_folders_work_in_project_root(served):
    root = SessionDirs(served.workspace)
    assert _run(served, {}) == root


def test_runs_for_served_folders_run(served, tmp_path):
    assert _run(served, served.metadata()) == served

    # The same folders written another way: a symlink and a trailing slash.
    link = tmp_path / "link"
    link.symlink_to(served.workspace.root, target_is_directory=True)
    assert (
        _run(served, {"workspace_dir": f"{link}/", "run_dir": str(link / "runs" / RUN)})
        == served
    )


def test_runs_for_other_workspace_are_refused(served, tmp_path):
    other = SessionDirs(Workspace(tmp_path / "other"))
    for graph_id in ("writing-agent", "scheduler"):
        with pytest.raises(RunFolderMismatchError) as exc:
            _run(served, other.metadata(), graph_id=graph_id)
        assert str(served.workspace.root) in str(exc.value)
        assert str(other.workspace.root) in str(exc.value)


def test_resolve_folders_refuses_unreadable_folders(served):
    """Present but unreadable is a mismatch, not a run without folders."""
    for workspace_dir in ("/tmp/evil\x00x", 123, "relative/ws"):
        with pytest.raises(RunFolderMismatchError, match="cannot read"):
            _run(served, {"workspace_dir": workspace_dir})
    # Every graph reads the run folder, including those that work in the root.
    for graph_id in ("writing-agent", "scheduler"):
        for run_dir in ("/tmp/evil\x00x", 123, f"runs/{RUN}"):
            configurable = {"workspace_dir": served.workspace.key, "run_dir": run_dir}
            with pytest.raises(RunFolderMismatchError, match="cannot read"):
                _run(served, configurable, graph_id=graph_id)


@pytest.mark.parametrize("graph_id", ["writing-agent", "scheduler"])
@pytest.mark.parametrize(
    "run_dir",
    ["elsewhere/runs/" + RUN, "ws/memories", "ws/runs/" + RUN + "/nested"],
)
def test_resolve_folders_refuses_run_dir_outside_workspace_runs(
    served, tmp_path, run_dir, graph_id
):
    with pytest.raises(RunFolderMismatchError, match="not a run folder"):
        _run(
            served,
            {"workspace_dir": served.workspace.key, "run_dir": str(tmp_path / run_dir)},
            graph_id=graph_id,
        )


def test_resolve_folders_refuses_runs_folder_linked_outside_workspace(
    workspace, tmp_path
):
    """Not supported, as stored bindings would lose the run folder."""
    disk = tmp_path / "disk"
    disk.mkdir()
    workspace.runs_dir.symlink_to(disk, target_is_directory=True)
    configurable = {
        "workspace_dir": workspace.key,
        "run_dir": str(workspace.runs_dir / RUN),
    }
    with pytest.raises(RunFolderMismatchError, match="not a run folder"):
        _run(SessionDirs(workspace), configurable)


def test_resolve_folders_refuses_relative_run_dir(served):
    configurable = {"workspace_dir": served.workspace.key, "run_dir": f"runs/{RUN}"}
    with pytest.raises(RunFolderMismatchError, match="not an absolute path"):
        _run(served, configurable)


def test_resolve_folders_reads_project_root_as_no_run_folder(served):
    configurable = {
        "workspace_dir": served.workspace.key,
        "run_dir": served.workspace.key,
    }
    assert _run(served, configurable) == SessionDirs(served.workspace)


def test_runs_work_in_any_run_folder_of_served_project(served):
    other_run = SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    assert _run(served, other_run.metadata()) == other_run
    # Scheduled tasks and memory workers work in the root, whatever run
    # folder the run forwards.
    root = SessionDirs(served.workspace)
    assert _run(served, other_run.metadata(), graph_id="scheduler") == root


# ---------------------------------------------------------------------------
# A thread's binding
# ---------------------------------------------------------------------------


def test_runs_on_workspace_assistant_continue_in_bound_run_folder(served):
    """A bare resume names its assistant, not the run folder."""
    assistant = workspace_assistant_id("writing-agent", served.workspace)
    configurable = {"assistant_id": assistant, "workspace_dir": served.workspace.key}

    assert _run(served, configurable, bound=served) == served


def test_runs_on_workspace_assistant_ignore_merged_workspace(served, tmp_path):
    """The runtime merges the thread's stored config into the run's: a thread
    whose first run was on another project's assistant keeps that project's
    ``workspace_dir``, which must not pin the thread there."""
    assistant = workspace_assistant_id("writing-agent", served.workspace)
    other = SessionDirs(Workspace(tmp_path / "other"))
    configurable = {"assistant_id": assistant, **other.metadata()}
    assert _run(served, configurable) == SessionDirs(served.workspace)
    # A run folder the request forwards is read against the assistant's
    # project, and an unreadable merged workspace is ignored too.
    for workspace_dir in (other.workspace.key, "/tmp/evil\x00x"):
        configurable = {
            "assistant_id": assistant,
            "workspace_dir": workspace_dir,
            "run_dir": str(served.run_dir),
        }
        assert _run(served, configurable) == served


def test_runs_for_other_run_folder_than_binding_are_refused(served):
    bound = SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    with pytest.raises(RunFolderMismatchError) as exc:
        _run(served, served.metadata(), bound=bound)
    assert "This conversation works in" in str(exc.value)
    assert str(bound.run_dir) in str(exc.value)


@pytest.mark.parametrize("graph_id", ["writing-agent", "scheduler"])
def test_runs_without_folders_continue_in_binding(served, graph_id):
    """A bare resume names no folder; a graph that works in the root keeps
    the binding's run folder in its stamp."""
    bound = SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    assert _run(served, {}, bound=bound, graph_id=graph_id) == bound


def test_runs_on_thread_bound_to_unserved_workspace_are_refused(served, tmp_path):
    bound = SessionDirs(Workspace(tmp_path / "other"))
    with pytest.raises(RunFolderMismatchError, match="does not serve"):
        _run(served, bound.metadata(), bound=bound)
    with pytest.raises(RunFolderMismatchError, match="This conversation works in"):
        _run(served, {}, bound=bound)


def test_updates_keep_binding(served, tmp_path):
    """Folders in the request cannot re-bind a thread."""
    elsewhere = SessionDirs(Workspace(tmp_path / "other")).metadata()

    assert (
        resolve_folders("writing-agent", "threads.update", elsewhere, served, served)
        == served
    )
    assert (
        resolve_folders("writing-agent", "threads.update", elsewhere, served, None)
        is None
    )


@pytest.mark.parametrize("graph_id", ["writing-agent", "scheduler"])
def test_updates_work_in_bound_run_folder(served, graph_id):
    bound = SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    assert resolve_folders(graph_id, "threads.update", {}, served, bound) == bound


def test_updates_of_threads_bound_to_unserved_workspace_are_refused(served, tmp_path):
    bound = SessionDirs(Workspace(tmp_path / "other"))
    with pytest.raises(RunFolderMismatchError, match="does not serve"):
        resolve_folders("writing-agent", "threads.update", {}, served, bound)


@pytest.mark.parametrize("access", ["threads.read", "assistants.read"])
def test_reads_stamp_nothing(served, access):
    assert resolve_folders("writing-agent", access, {}, served, served) is None


# ---------------------------------------------------------------------------
# The registered factories
# ---------------------------------------------------------------------------


def test_served_graph_factory_matches_server_signature():
    """langgraph-api classifies the factory by its ``runtime`` annotation."""
    import inspect
    import typing

    from langgraph_sdk.runtime import ServerRuntime

    factory = served_graph("writing-agent")
    assert list(inspect.signature(factory).parameters) == ["config", "runtime"]
    assert typing.get_type_hints(factory)["runtime"] is ServerRuntime


class _Built:
    """A graph cache that returns one graph for every key."""

    def __init__(self, graph):
        self.graph = graph

    def get(self, _graph_id, _dirs):
        return self.graph


@pytest.fixture
def factory_env(served, monkeypatch):
    """The served folders, one graph for every key, and no binding."""
    graph = object()
    monkeypatch.setattr(factories, "GRAPHS", _Built(graph))
    monkeypatch.setattr(factories, "process_session_dirs", lambda: served)
    monkeypatch.setattr(factories, "thread_binding", lambda _thread_id: None)
    return graph


def test_factory_hands_folders_to_checkpointer(served, factory_env, tmp_path):
    factory = served_graph("writing-agent")

    async def access(config, context):
        graph = await factory(config, SimpleNamespace(access_context=context))
        return graph, RESOLVED_FOLDERS.get()

    assert asyncio.run(access(_config(served), RUN_ACCESS)) == (factory_env, served)
    assert asyncio.run(access(_config(served), "threads.read")) == (factory_env, None)
    with pytest.raises(RunFolderMismatchError):
        asyncio.run(
            access(_config(SessionDirs(Workspace(tmp_path / "other"))), RUN_ACCESS)
        )


async def test_checkpoint_writes_see_factory_folders(tiny_graph, served, monkeypatch):
    """langgraph-api awaits the factory in the run's task, and the Pregel loop
    writes checkpoints from tasks of its own: they inherit the folders the
    factory set (a runtime behaviour this design depends on: recheck it after
    every runtime upgrade)."""
    from langgraph.checkpoint.memory import InMemorySaver

    seen = []

    class Recording(InMemorySaver):
        async def aput(self, config, checkpoint, metadata, new_versions):
            seen.append(RESOLVED_FOLDERS.get(None))
            return await super().aput(config, checkpoint, metadata, new_versions)

    graph = tiny_graph()
    graph.checkpointer = Recording()
    monkeypatch.setattr(factories, "GRAPHS", _Built(graph))
    monkeypatch.setattr(factories, "process_session_dirs", lambda: served)
    monkeypatch.setattr(factories, "thread_binding", lambda _thread_id: None)
    config = {"configurable": {"thread_id": "t", **served.metadata()}}

    built = await served_graph("writing-agent")(
        config, SimpleNamespace(access_context=RUN_ACCESS)
    )
    await built.ainvoke({"x": 1}, {"configurable": {"thread_id": "t"}})

    assert seen
    assert set(seen) == {served}


def test_refused_updates_answer_409(served, factory_env, monkeypatch, tmp_path):
    from starlette.exceptions import HTTPException

    bound = SessionDirs(Workspace(tmp_path / "other"))
    monkeypatch.setattr(factories, "thread_binding", lambda _thread_id: bound)
    update = SimpleNamespace(access_context="threads.update")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            served_graph("writing-agent")({"configurable": {"thread_id": "t"}}, update)
        )
    assert exc.value.status_code == 409
    assert "does not serve" in exc.value.detail


def test_updates_with_unreadable_binding_answer_503(served, factory_env, monkeypatch):
    from starlette.exceptions import HTTPException

    from EvoScientist.sessions import ThreadBindingError

    def unreadable(thread_id):
        raise ThreadBindingError(
            f"Could not read the folders of conversation {thread_id}"
        )

    monkeypatch.setattr(factories, "thread_binding", unreadable)
    update = SimpleNamespace(access_context="threads.update")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            served_graph("writing-agent")({"configurable": {"thread_id": "t"}}, update)
        )
    assert exc.value.status_code == 503
    assert "conversation t" in exc.value.detail


def test_reads_skip_binding_lookup(served, factory_env, monkeypatch):
    def unexpected(_thread_id):
        raise AssertionError("a read looked up the binding")

    monkeypatch.setattr(factories, "thread_binding", unexpected)
    read = SimpleNamespace(access_context="threads.read")
    asyncio.run(
        served_graph("writing-agent")({"configurable": {"thread_id": "t"}}, read)
    )


# ---------------------------------------------------------------------------
# Runs as the runtime creates them
# ---------------------------------------------------------------------------


async def _runtime_run(assistant_id: str, thread_id: str, configurable=None) -> dict:
    """The config langgraph-api hands the factory for a run it created."""
    from langgraph_runtime_inmem.database import connect
    from langgraph_runtime_inmem.ops import Runs

    async with connect() as conn:
        runs = await Runs.put(
            conn,
            assistant_id,
            {"config": {"configurable": configurable or {}}},
            thread_id=thread_id,
            metadata={},
            prevent_insert_if_inflight=False,
            if_not_exists="create",
        )
        # The new run first, then the thread's runs still in flight.
        run = await anext(runs)
    return run["kwargs"]["config"]


async def _create_run(config: dict, graph_id: str = "EvoScientist"):
    await served_graph(graph_id)(config, SimpleNamespace(access_context=RUN_ACCESS))
    return RESOLVED_FOLDERS.get()


async def test_refused_run_does_not_pin_thread_to_other_project(
    runtime_store, served, factory_env, tmp_path
):
    """The runtime stores the first workspace assistant's config on the thread
    and merges it into every later run, even when that first run was refused."""
    from EvoScientist.langgraph_dev.assistants import ensure_workspace_assistants

    other = Workspace(tmp_path / "other")
    await ensure_workspace_assistants(served.workspace)
    await ensure_workspace_assistants(other)
    thread_id = "11111111-1111-1111-1111-111111111111"

    refused = await _runtime_run(
        workspace_assistant_id("EvoScientist", other), thread_id
    )
    with pytest.raises(RunFolderMismatchError):
        await _create_run(refused)
    # The refused run errors, which frees the thread for the next run.
    for record in (*runtime_store["threads"], *runtime_store["runs"]):
        record["status"] = "error"
    retried = await _runtime_run(
        workspace_assistant_id("EvoScientist", served.workspace), thread_id
    )

    assert retried["configurable"]["workspace_dir"] == other.key
    assert await _create_run(retried) == SessionDirs(served.workspace)


async def test_runtime_bare_resume_continues_in_binding(
    runtime_store, served, factory_env, monkeypatch
):
    from EvoScientist.langgraph_dev.assistants import ensure_workspace_assistants

    await ensure_workspace_assistants(served.workspace)
    bound = SessionDirs(served.workspace, served.workspace.runs_dir / "earlier")
    monkeypatch.setattr(factories, "thread_binding", lambda _thread_id: bound)
    config = await _runtime_run(
        workspace_assistant_id("EvoScientist", served.workspace),
        "22222222-2222-2222-2222-222222222222",
    )

    assert await _create_run(config) == bound


# ---------------------------------------------------------------------------
# Child runs forward their folders
# ---------------------------------------------------------------------------


def _launch(workspace: Workspace, work_dir=None) -> dict:
    """Start an async sub-agent and return the ``configurable`` it was sent."""
    from EvoScientist.middleware.expert_async_subagent import (
        EvoAsyncSubAgentMiddleware,
    )

    middleware = EvoAsyncSubAgentMiddleware(
        workspace=workspace,
        work_dir=work_dir,
        async_subagents=[
            {
                "name": "writing-agent",
                "description": "writes",
                "graph_id": "writing-agent",
                "url": "http://127.0.0.1:6174",
            }
        ],
    )
    start = next(t for t in middleware.tools if t.name == "start_async_task")
    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "t1"}
    client.runs.create.return_value = {"run_id": "r1"}
    with patch(
        "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
        return_value=client,
    ):
        start.func(
            description="draft the paper",
            subagent_type="writing-agent",
            runtime=SimpleNamespace(tool_call_id="tc1", config={}),
        )
    return client.runs.create.call_args.kwargs["config"]["configurable"]


def test_async_tasks_forward_the_session_folders(workspace):
    run = workspace.runs_dir / RUN
    configurable = _launch(workspace, work_dir=run)
    assert configurable["workspace_dir"] == workspace.key
    assert configurable["run_dir"] == run.as_posix()


def test_daemon_async_tasks_forward_the_root_only(workspace):
    configurable = _launch(workspace)
    assert configurable["workspace_dir"] == workspace.key
    assert "run_dir" not in configurable


def test_memory_workers_and_the_linker_forward_the_root(workspace, tmp_path):
    from EvoScientist.memory.launch import (
        _memory_worker_run_payload,
        _observation_linker_run_payload,
    )
    from tests.test_observation_memory import _linker_context, _memory_source_context

    worker = _memory_worker_run_payload(
        context=_memory_source_context(
            memory_dir=tmp_path / "memories", workspace_dir=workspace.root
        ),
        thread_id="t1",
    )
    linker = _observation_linker_run_payload(
        context=_linker_context(
            memory_dir=tmp_path / "memories",
            workspace_dir=workspace.root,
            observation_ids=("o1",),
        ),
        thread_id="t2",
    )
    for payload in (worker, linker):
        assert payload["config"]["configurable"]["workspace_dir"] == workspace.key


def test_manual_autoskills_runs_forward_the_root(workspace, monkeypatch):
    import EvoScientist.memory.autoskills.schedule as schedule

    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "t1"}
    client.runs.create.return_value = {"run_id": "r1"}
    monkeypatch.setattr(schedule, "get_langgraph_sync_client", lambda **_k: client)

    config = SimpleNamespace(
        memory_skill_synthesis_mode=SimpleNamespace(value="propose"),
        memory_skill_synthesis_cadence=SimpleNamespace(value="daily"),
        memory_skill_synthesis_time="03:00",
    )
    monkeypatch.setattr(schedule, "langgraph_dev_url", lambda _c: "http://x")
    schedule.run_autoskill_now(config, workspace_dir=workspace.root)

    run = client.runs.create.call_args.kwargs
    assert run["config"]["configurable"]["workspace_dir"] == workspace.key
    assert run["assistant_id"] == workspace_assistant_id(
        schedule.AUTOSKILL_GRAPH_ID, workspace
    )


def test_registered_graphs_refuse_other_workspace(served, factory_env):
    """Every registered graph refuses another workspace and runs in any run
    folder of the served one."""
    import importlib
    import json
    from pathlib import Path

    registry = json.loads(
        (Path(factories.__file__).parent / "langgraph.json").read_text()
    )["graphs"]
    other_run = _config(
        SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    )
    other_workspace = _config(SessionDirs(Workspace(served.workspace.root / "other")))
    running = SimpleNamespace(access_context=RUN_ACCESS)
    assert FOLDER_GRAPH_IDS <= registry.keys()

    for target in registry.values():
        module_path, attr = target.rsplit(":", 1)
        factory = getattr(importlib.import_module(module_path), attr)
        with pytest.raises(RunFolderMismatchError):
            asyncio.run(factory(other_workspace, running))
        asyncio.run(factory(other_run, running))


def test_async_task_check_reports_why_a_run_was_refused(workspace):
    """langgraph-api records a failed run's error on its thread; the check
    shows it to the agent instead of a generic failure."""
    import json

    from EvoScientist.middleware.expert_async_subagent import (
        EvoAsyncSubAgentMiddleware,
    )

    middleware = EvoAsyncSubAgentMiddleware(
        workspace=workspace,
        async_subagents=[
            {
                "name": "writing-agent",
                "description": "writes",
                "graph_id": "writing-agent",
                "url": "http://127.0.0.1:6174",
            }
        ],
    )
    check = next(t for t in middleware.tools if t.name == "check_async_task")
    client = MagicMock()
    client.runs.get.return_value = {
        "status": "error",
        "thread_id": "t1",
        "run_id": "r1",
    }
    client.threads.get.return_value = {
        "error": {
            "error": "RunFolderMismatchError",
            "message": "This EvoScientist server serves workspace /a",
        }
    }
    task = {
        "task_id": "t1",
        "agent_name": "writing-agent",
        "thread_id": "t1",
        "run_id": "r1",
        "status": "running",
        "created_at": "2026-10-01T00:00:00Z",
        "last_checked_at": "2026-10-01T00:00:00Z",
        "last_updated_at": "2026-10-01T00:00:00Z",
    }
    runtime = SimpleNamespace(state={"async_tasks": {"t1": task}}, tool_call_id="tc1")
    with patch(
        "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
        return_value=client,
    ):
        first = check.func(task_id="t1", runtime=runtime)
        second = check.func(task_id="t1", runtime=runtime)

    for command in (first, second):
        result = json.loads(command.update["messages"][0].content)
        assert result["status"] == "error"
        assert "serves workspace /a" in result["error"]
    # A failed run's reason does not change; it is looked up once.
    client.threads.get.assert_called_once_with("t1")


def test_new_run_folders_never_reuse_an_existing_one(workspace, monkeypatch):
    """Two sessions starting in the same second get different run folders."""
    from datetime import datetime

    import EvoScientist.cli.agent as agent

    class _SameSecond(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 1, 12, 0, 0)

    monkeypatch.setattr(agent, "datetime", _SameSecond)
    first = agent._create_run_dir(workspace)
    second = agent._create_run_dir(workspace)

    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_create_run_dir_skips_dangling_symlink(workspace):
    import EvoScientist.cli.agent as agent

    workspace.runs_dir.mkdir(parents=True)
    (workspace.runs_dir / "exp").symlink_to(workspace.root / "does-not-exist")

    assert agent._create_run_dir(workspace, "exp") == workspace.runs_dir / "exp_1"


# ---------------------------------------------------------------------------
# Failed child runs report why
# ---------------------------------------------------------------------------


def test_run_errors_keep_exception_type():
    """langgraph-api stores a placeholder text for most exception types."""
    from EvoScientist.llm.patches import _RunErrors

    failed = {"status": "error", "thread_id": "t1", "run_id": "r1"}
    attached = _RunErrors().attach(
        failed,
        {"error": {"error": "RateLimitError", "message": "An internal error occurred"}},
    )
    assert "RateLimitError" in attached["error"]


def test_run_errors_return_run_without_id_unchanged():
    from EvoScientist.llm.patches import _RunErrors

    run = {"status": "error", "thread_id": "t1"}
    thread = {"error": {"error": "RuntimeError", "message": "boom"}}
    assert _RunErrors().attach(run, thread) == run


async def test_acheck_async_task_reports_failure_reason(workspace):
    """``acheck_async_task`` (langgraph dev, TUI) reads the thread's error too."""
    import json
    from unittest.mock import AsyncMock

    from EvoScientist.middleware.expert_async_subagent import (
        EvoAsyncSubAgentMiddleware,
    )

    middleware = EvoAsyncSubAgentMiddleware(
        workspace=workspace,
        async_subagents=[
            {
                "name": "writing-agent",
                "description": "writes",
                "graph_id": "writing-agent",
                "url": "http://127.0.0.1:6174",
            }
        ],
    )
    check = next(t for t in middleware.tools if t.name == "check_async_task")

    def _task(task_id: str) -> dict:
        return {
            "task_id": task_id,
            "agent_name": "writing-agent",
            "thread_id": task_id,
            "run_id": f"r-{task_id}",
            "status": "running",
            "created_at": "2026-10-01T00:00:00Z",
            "last_checked_at": "2026-10-01T00:00:00Z",
            "last_updated_at": "2026-10-01T00:00:00Z",
        }

    def _thread(thread_id: str, **_k) -> dict:
        if thread_id != "t1":
            raise RuntimeError("thread unreadable")
        return {"error": {"error": "RunFolderMismatchError", "message": "no"}}

    client = MagicMock()
    client.runs.get = AsyncMock(
        side_effect=lambda thread_id, run_id, **_k: {
            "status": "error",
            "thread_id": thread_id,
            "run_id": run_id,
        }
    )
    client.threads.get = AsyncMock(side_effect=_thread)
    runtime = SimpleNamespace(
        state={"async_tasks": {"t1": _task("t1"), "t2": _task("t2")}},
        tool_call_id="tc1",
    )
    with patch(
        "EvoScientist.middleware.expert_async_subagent._ClientCache.get_async",
        return_value=client,
    ):
        refused = await check.coroutine(task_id="t1", runtime=runtime)
        unreadable = await check.coroutine(task_id="t2", runtime=runtime)

    result = json.loads(refused.update["messages"][0].content)
    assert "RunFolderMismatchError" in result["error"]
    # When the thread cannot be read, the run is reported as it is.
    result = json.loads(unreadable.update["messages"][0].content)
    assert result["status"] == "error"


def test_merge_runs_config_replaces_caller_folders():
    """The launching graph's folders win over a caller's stale ones."""
    from EvoScientist.llm.patches import _merge_runs_config_kwargs

    caller = {"workspace_dir": "/c", "run_dir": "/c/runs/r", "thread_id": "t1"}
    with patch("EvoScientist.llm.patches._read_cfg_configurable", lambda: {}):
        merged = _merge_runs_config_kwargs(
            {"config": {"configurable": caller}}, {"workspace_dir": "/ws"}
        )
    assert merged["config"]["configurable"] == {
        "workspace_dir": "/ws",
        "thread_id": "t1",
    }


def test_merge_runs_config_runs_graph_on_workspace_assistant():
    """A run named by its graph goes to that graph's assistant for the
    launching graph's workspace; an assistant ID is left as it is."""
    from EvoScientist.llm.patches import _merge_runs_config_kwargs

    folders = {"workspace_dir": "/ws"}
    with patch("EvoScientist.llm.patches._read_cfg_configurable", lambda: {}):
        named = _merge_runs_config_kwargs({"assistant_id": "writing-agent"}, folders)
        kept = _merge_runs_config_kwargs({"assistant_id": "custom-id"}, folders)
        unscoped = _merge_runs_config_kwargs({"assistant_id": "writing-agent"})

    assert named["assistant_id"] == workspace_assistant_id("writing-agent", "/ws")
    assert kept["assistant_id"] == "custom-id"
    assert unscoped["assistant_id"] == "writing-agent"


def test_server_gateway_run_config_forwards_target_folders(monkeypatch):
    """The gateway's own main-agent runs carry their folders like child runs,
    over any a caller put in ``configurable_extra``."""
    import EvoScientist.EvoScientist as evo
    from EvoScientist.gateway.server import LangGraphServerGateway
    from EvoScientist.gateway.types import GraphTarget

    monkeypatch.setattr(
        evo,
        "_ensure_config",
        lambda: SimpleNamespace(model="m", provider="p", recursion_limit=None),
    )
    monkeypatch.setattr(
        "EvoScientist.backends.hitl_suppressed_for_run", lambda _cfg: False
    )
    gateway = object.__new__(LangGraphServerGateway)
    stale = {"workspace_dir": "/b", "run_dir": "/b/runs/y", "active_teams": ["t"]}

    run_mode = gateway._resolve_run_config(
        "t1", stale, target=GraphTarget(workspace_dir="/ws", run_dir="/ws/runs/x")
    )["configurable"]
    daemon = gateway._resolve_run_config(
        "t1", stale, target=GraphTarget(workspace_dir="/ws")
    )["configurable"]
    bare = gateway._resolve_run_config("t1", None)["configurable"]

    assert (run_mode["workspace_dir"], run_mode["run_dir"]) == ("/ws", "/ws/runs/x")
    assert run_mode["model"] == "m"
    assert run_mode["active_teams"] == ["t"]
    assert daemon["workspace_dir"] == "/ws"
    assert "run_dir" not in daemon
    assert "workspace_dir" not in bare


# ---------------------------------------------------------------------------
# The server moves with the session
# ---------------------------------------------------------------------------


async def _sync_order(workspace, *, pinned_elsewhere: bool) -> list[str]:
    """Sync the server to ``workspace`` and return what ran, in order."""
    from EvoScientist.cli import commands

    order: list[str] = []
    with (
        patch.object(
            commands, "_wait_for_memory_workers", lambda **_k: order.append("memory")
        ),
        patch(
            "EvoScientist.langgraph_dev.manager.owned_server_pinned_elsewhere",
            lambda *a, **k: pinned_elsewhere,
        ),
        patch(
            "EvoScientist.langgraph_dev.manager.ensure_langgraph_dev",
            lambda *a, **k: order.append("move"),
        ),
        patch.object(commands, "_reconcile_autoskill_schedule", lambda *a, **k: None),
        patch.object(commands, "_adopt_stale_scheduled_tasks", lambda **_k: None),
    ):
        await commands._sync_background_agent_server_workspace(
            MagicMock(), dirs=SessionDirs(workspace)
        )
    return order


async def test_server_move_waits_for_memory_work(workspace):
    """Moving the pinned server stops runs in flight, so the previous
    session's memory work finishes before it moves."""
    assert await _sync_order(workspace, pinned_elsewhere=True) == ["memory", "move"]


async def test_server_sync_to_served_folders_skips_memory_wait(workspace):
    """A daemon-mode /resume syncs to the folders the server already serves."""
    assert await _sync_order(workspace, pinned_elsewhere=False) == ["move"]


def test_owned_server_pinned_elsewhere_compares_folder_pair(workspace, monkeypatch):
    from EvoScientist.langgraph_dev import manager

    alive = SimpleNamespace(poll=lambda: None)
    run = workspace.runs_dir / RUN
    monkeypatch.setattr(manager, "_PROCESS", alive)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", workspace.root)
    monkeypatch.setattr(manager, "_PROCESS_RUN_DIR", run)
    assert manager.owned_server_pinned_elsewhere(workspace.root, run) is False
    assert manager.owned_server_pinned_elsewhere(workspace.root, None) is True
    assert manager.owned_server_pinned_elsewhere(workspace.root / "other", run) is True
    monkeypatch.setattr(manager, "_PROCESS", None)
    assert manager.owned_server_pinned_elsewhere(workspace.root, None) is False


def test_ensure_async_subagent_server_refused_removes_run_dir(workspace):
    """A session refused by another one's server leaves no empty run folder."""
    import typer

    from EvoScientist.cli import commands
    from EvoScientist.langgraph_dev.manager import WorkspaceMismatchError

    run = workspace.runs_dir / RUN
    run.mkdir(parents=True)

    def _refuse(*_a, **_k):
        raise WorkspaceMismatchError("held by another session")

    with (
        patch("EvoScientist.langgraph_dev.manager.ensure_langgraph_dev", _refuse),
        pytest.raises(typer.Exit),
    ):
        commands._ensure_async_subagent_server(
            MagicMock(), dirs=SessionDirs(workspace, run)
        )

    assert not run.exists()
    assert workspace.runs_dir.is_dir()


async def test_restore_thread_dirs_removes_unused_run_dir(workspace, tmp_path):
    """A ``--mode=run`` resume works in the thread's folders, so the run
    folder made at startup goes."""
    from unittest.mock import AsyncMock

    from EvoScientist.cli import commands

    started = SessionDirs(workspace, workspace.runs_dir / RUN)
    started.run_dir.mkdir(parents=True)
    stored = SessionDirs(Workspace(tmp_path / "other"))
    gateway = SimpleNamespace(
        get_thread_metadata=AsyncMock(return_value=stored.metadata())
    )

    with patch.object(commands, "_sync_background_agent_server_workspace", AsyncMock()):
        restored = await commands._restore_thread_dirs(
            "t1",
            dirs=started,
            served=started,
            graph_gateway=gateway,
            config=MagicMock(),
        )

    assert restored == stored
    assert not started.run_dir.exists()


async def _new_session(workspace, sync) -> SessionDirs:
    """Run the Rich CLI ``/new`` folder switch of a ``--mode=run`` session."""
    from EvoScientist.cli import commands
    from EvoScientist.cli.interactive import _new_session_dirs

    with patch.object(commands, "_sync_background_agent_server_workspace", sync):
        return await _new_session_dirs(
            SessionDirs(workspace),
            config=MagicMock(),
            backend=None,
            mode="run",
            run_name="exp",
        )


async def test_new_session_dirs_move_server_to_new_run_dir(workspace):
    from unittest.mock import AsyncMock

    sync = AsyncMock()
    new_dirs = await _new_session(workspace, sync)

    assert new_dirs == SessionDirs(workspace, workspace.runs_dir / "exp")
    assert new_dirs.run_dir.is_dir()
    assert sync.await_args.kwargs["dirs"] == new_dirs


async def test_new_session_dirs_refused_removes_run_dir(workspace):
    """Another session holds the server: ``/new`` fails and the session
    keeps its folders, with no empty run folder left behind."""
    from unittest.mock import AsyncMock

    from EvoScientist.langgraph_dev.manager import WorkspaceMismatchError

    sync = AsyncMock(side_effect=WorkspaceMismatchError("held by another session"))
    with pytest.raises(WorkspaceMismatchError):
        await _new_session(workspace, sync)

    assert not (workspace.runs_dir / "exp").exists()


async def test_new_session_dirs_survive_failed_sync(workspace):
    """Any other sync failure starts the new session without background work."""
    from unittest.mock import AsyncMock

    new_dirs = await _new_session(workspace, AsyncMock(side_effect=OSError("down")))

    assert new_dirs == SessionDirs(workspace, workspace.runs_dir / "exp")
    assert new_dirs.run_dir.is_dir()
