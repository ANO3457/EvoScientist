"""Workspace assistants: the IDs and fields the WebUI shares, and their upkeep."""

from __future__ import annotations

import uuid

import pytest

from EvoScientist.langgraph_dev.assistants import (
    ensure_workspace_assistants,
    folder_name,
    served_graph_ids,
    workspace_assistant_fields,
    workspace_assistant_id,
)
from EvoScientist.paths import Workspace


def test_workspace_assistant_id_matches_the_webui():
    # Computed with the WebUI's own ``uuid`` package (``workspaceAssistantId``).
    assert (
        workspace_assistant_id("EvoScientist", "/home/you/projA")
        == "39dcbcc1-957e-51c1-8e52-c294d4031162"
    )
    assert (
        workspace_assistant_id("scheduler", "/tmp/projÄ ü")
        == "17766371-eab6-54fd-940c-f639d8dc3d97"
    )


def test_workspace_assistant_id_uses_the_workspace_key(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace_assistant_id("EvoScientist", workspace) == workspace_assistant_id(
        "EvoScientist", workspace.key
    )


@pytest.mark.parametrize(
    ("key", "name"),
    [("/home/you/projA", "projA"), ("C:/Users/you/proj", "proj"), ("/", "/")],
)
def test_folder_name_takes_the_last_segment(key, name):
    assert folder_name(key) == name


def test_workspace_assistant_fields_match_the_webui():
    assert workspace_assistant_fields("scheduler", "/home/you/projA") == {
        "graph_id": "scheduler",
        "config": {"configurable": {"workspace_dir": "/home/you/projA"}},
        "metadata": {"workspace_dir": "/home/you/projA"},
        "name": "scheduler · projA",
    }


@pytest.fixture
def runtime_store(monkeypatch):
    """The in-memory runtime's store, empty, with every served graph registered."""
    # Read when langgraph-api's config is first imported, as in the server.
    for name, value in {
        "LANGGRAPH_RUNTIME_EDITION": "inmem",
        "REDIS_URI": "fake",
        "DATABASE_URI": ":memory:",
        "MIGRATIONS_PATH": "__inmem",
    }.items():
        monkeypatch.setenv(name, value)
    import langgraph_api.graph as lg_graph
    from langgraph_runtime_inmem.database import GLOBAL_STORE

    for graph_id in served_graph_ids():
        monkeypatch.setitem(lg_graph.GRAPHS, graph_id, object())
    saved = {
        key: list(GLOBAL_STORE.get(key, []))
        for key in ("assistants", "assistant_versions")
    }
    GLOBAL_STORE["assistants"] = []
    GLOBAL_STORE["assistant_versions"] = []
    yield GLOBAL_STORE
    GLOBAL_STORE.update(saved)


def _assistant(store, assistant_id):
    (found,) = [
        a for a in store["assistants"] if a["assistant_id"] == uuid.UUID(assistant_id)
    ]
    return found


async def test_ensure_workspace_assistants_creates_one_per_graph(
    runtime_store, tmp_path
):
    workspace = Workspace(tmp_path)
    await ensure_workspace_assistants(workspace)
    await ensure_workspace_assistants(workspace)

    assert len(runtime_store["assistants"]) == len(served_graph_ids())
    main = _assistant(runtime_store, workspace_assistant_id("EvoScientist", workspace))
    assert main["config"] == {"configurable": {"workspace_dir": workspace.key}}
    assert main["metadata"] == {"workspace_dir": workspace.key}


@pytest.mark.parametrize("folder", ["/elsewhere", None])
async def test_ensure_workspace_assistants_repairs_other_fields(
    runtime_store, tmp_path, folder
):
    """Another folder, or the right folder with a stale name and metadata."""
    workspace = Workspace(tmp_path)
    folder = folder or workspace.key
    assistant_id = workspace_assistant_id("EvoScientist", workspace)
    runtime_store["assistants"].append(
        {
            "assistant_id": uuid.UUID(assistant_id),
            "graph_id": "EvoScientist",
            "config": {"configurable": {"workspace_dir": folder}},
            "context": {"workspace_dir": folder},
            "metadata": {},
            "name": "planted",
            "version": 1,
            "description": None,
        }
    )
    runtime_store["assistant_versions"].append(
        {"assistant_id": uuid.UUID(assistant_id), "version": 1}
    )

    await ensure_workspace_assistants(workspace)

    repaired = _assistant(runtime_store, assistant_id)
    assert repaired["config"] == {"configurable": {"workspace_dir": workspace.key}}
    assert repaired["metadata"] == {"workspace_dir": workspace.key}
    assert repaired["name"] == f"EvoScientist · {tmp_path.name}"


def test_server_start_sets_the_served_workspace_up(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev import workspace_setup
    from EvoScientist.langgraph_dev.http import app

    set_up = []

    async def record(dirs):
        set_up.append(dirs)

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(workspace_setup, "set_up_workspace", record)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)):
        pass
    assert [d.workspace for d in set_up] == [Workspace(tmp_path)]


def test_server_start_fails_when_the_assistants_cannot_be_created(
    monkeypatch, tmp_path
):
    """Without them most runs would fail with a 404 that hides the cause."""
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev import assistants
    from EvoScientist.langgraph_dev.http import app

    async def fail(_workspace):
        raise ValueError("store unavailable")

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(assistants, "ensure_workspace_assistants", fail)
    with pytest.raises(
        RuntimeError, match=r"workspace assistants for .*store unavailable"
    ):
        with TestClient(app, base_url="http://127.0.0.1"):
            pass
