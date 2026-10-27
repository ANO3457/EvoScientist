"""The server outlives the sessions that use it: ``EvoSci server start``, the
offer after onboarding, and the setup stage after an upgrade."""

from __future__ import annotations

import json
import time

import pytest
import yaml
from typer.testing import CliRunner

from EvoScientist import paths
from EvoScientist.cli._app import app
from EvoScientist.config.settings import get_config_path
from EvoScientist.langgraph_dev import manager
from EvoScientist.langgraph_dev.registration import OlderServerError
from EvoScientist.paths import Workspace
from EvoScientist.setup import server as server_stage


class _Started:
    """The server process this process started."""

    pid = 4242

    def poll(self):
        return None


@pytest.fixture
def started(monkeypatch) -> list:
    """Fake the server start (this process holds the server it starts);
    return the projects it was started for."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    calls: list = []

    def ensure(config, *, workspace_dir, backend):
        calls.append((workspace_dir, backend))
        if not manager.is_langgraph_dev_running():
            monkeypatch.setattr(manager, "_PROCESS", _Started())
        return True

    monkeypatch.setattr(manager, "ensure_langgraph_dev", ensure)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda config, *, lasting: None
    )
    return calls


def _emit():
    events: list[dict] = []
    return events, events.append


def _saved(**values) -> None:
    get_config_path().parent.mkdir(parents=True, exist_ok=True)
    get_config_path().write_text(yaml.safe_dump(values))


# ---------------------------------------------------------------------------
# EvoSci server start
# ---------------------------------------------------------------------------


def test_server_start_starts_server_for_project(started, tmp_path):
    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert started == [(tmp_path, "langgraph_server")]
    assert "EvoSci server stop" in " ".join(result.output.split())


def test_server_start_running_server_serves_project_too(started, monkeypatch, tmp_path):
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "already running" in " ".join(result.output.split())


def test_server_start_refusal_exits(started, monkeypatch, tmp_path):
    def older(*_a, **_k):
        raise OlderServerError()

    monkeypatch.setattr(manager, "ensure_langgraph_dev", older)

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 1
    assert "older EvoScientist" in " ".join(result.output.split())


# ---------------------------------------------------------------------------
# After onboarding
# ---------------------------------------------------------------------------


def _answer(monkeypatch, answer):
    import questionary

    asked = []

    def confirm(question, **_k):
        asked.append(question)
        return type("Q", (), {"ask": lambda self: answer})()

    monkeypatch.setattr(questionary, "confirm", confirm)
    return asked


@pytest.mark.parametrize("answer", [True, False])
def test_offer_service_start_names_project(monkeypatch, tmp_path, answer):
    from EvoScientist.cli import commands, server_cmd

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    asked = _answer(monkeypatch, answer)
    starts = []
    monkeypatch.setattr(
        server_cmd,
        "start_background",
        lambda workdir, *, with_ccproxy: starts.append(with_ccproxy),
    )

    commands._offer_server_start()

    assert str(tmp_path) in asked[0]
    assert starts == ([True] if answer else [])


def test_offer_without_default_workdir_starts_nothing(monkeypatch, tmp_path):
    """The folder onboarding ran in is not the project the user works in."""
    from EvoScientist.cli import commands, server_cmd

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVOSCIENTIST_WORKSPACE_DIR", raising=False)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    asked = _answer(monkeypatch, True)
    starts = []
    monkeypatch.setattr(
        server_cmd,
        "start_background",
        lambda workdir, *, with_ccproxy: starts.append(with_ccproxy),
    )

    commands._offer_server_start()

    assert asked == []
    assert starts == []


def _running_service(monkeypatch, sidecar, *, jobs=(), busy=False):
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    monkeypatch.setattr(manager, "_read_workspace_sidecar", lambda: sidecar)
    monkeypatch.setattr(manager, "_pid_serves_port", lambda *_a: True)
    monkeypatch.setattr(
        "EvoScientist.deploy.busy.running_bg_process_names", lambda url: list(jobs)
    )
    monkeypatch.setattr("EvoScientist.deploy.busy.has_busy_thread", lambda url: busy)


def _record_steps(monkeypatch) -> list:
    from EvoScientist.cli import server_cmd, service_cmd

    steps: list = []
    monkeypatch.setattr(service_cmd, "service_stop", lambda: steps.append("stop"))
    monkeypatch.setattr(
        server_cmd,
        "start_background",
        lambda workdir, *, with_ccproxy: steps.append("start"),
    )
    return steps


@pytest.mark.parametrize(
    "sidecar",
    [
        None,
        {"pid": 1, "workspace": "/a", "lasting": False},
        {"pid": 1, "lasting": True},
    ],
)
def test_offer_leaves_running_service_alone(monkeypatch, sidecar):
    """It says how to apply the new settings; it restarts nothing."""
    from EvoScientist.cli import commands

    _running_service(monkeypatch, sidecar)
    asked = _answer(monkeypatch, True)
    steps = _record_steps(monkeypatch)

    with commands.console.capture() as captured:
        commands._offer_server_start()

    assert asked == []
    assert steps == []
    lasting = bool(sidecar and sidecar.get("lasting"))
    hint = "EvoSci service stop" if lasting else "not the background service"
    assert hint in " ".join(captured.get().split())


def test_offer_service_start_failure_is_reported(monkeypatch, tmp_path):
    """The configuration is saved; no traceback."""
    from EvoScientist.cli import commands, server_cmd

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    _answer(monkeypatch, True)

    def unwritable(workdir, *, with_ccproxy):
        raise PermissionError("read-only")

    monkeypatch.setattr(server_cmd, "start_background", unwritable)

    with commands.console.capture() as captured:
        commands._offer_server_start()

    assert "next EvoSci session" in captured.get()


def test_onboard_without_terminal_offers_nothing(monkeypatch):
    from EvoScientist.cli import commands

    monkeypatch.setattr(commands, "_run_onboard_cli", lambda **_k: True)
    offered = []
    monkeypatch.setattr(commands, "_offer_server_start", lambda: offered.append(1))

    CliRunner().invoke(app, ["onboard"])

    assert offered == []


# ---------------------------------------------------------------------------
# The setup stage after an upgrade
# ---------------------------------------------------------------------------


def test_stage_announces_lasting_server_once_after_upgrade(started):
    _saved(model="m")
    _, emit = _emit()

    first = server_stage.run_stage(emit, "default")
    again = server_stage.run_stage(emit, "default")

    assert first.status == "done"
    assert first.message == server_stage.LASTING_MESSAGE
    assert again.status == "skipped"
    assert started == []


def test_stage_on_new_install_announces_nothing(started):
    result = server_stage.run_stage(_emit()[1], "default")

    assert result.status == "skipped"
    assert not get_config_path().exists()


def _updated(monkeypatch) -> None:
    """The update replaced the installed version since the record."""
    monkeypatch.setattr(server_stage, "_installed_version", lambda: "new")


def test_stage_restarts_server_update_stopped(started, monkeypatch, tmp_path):
    project = Workspace(tmp_path / "proj")
    project.root.mkdir()
    server_stage.remember_for_restart(str(project.root))
    _updated(monkeypatch)
    events, emit = _emit()

    result = server_stage.run_stage(emit, "default")

    assert started == [(project.root, "langgraph_server")]
    assert result.detail["restarted"] is True
    assert str(project.root) in result.message
    assert not (paths.SERVER_HOME / "restart-after-update.json").exists()
    assert [e["status"] for e in events] == ["running"]


def test_stage_ignores_record_of_failed_update(started, tmp_path):
    """An update that failed after it stopped the server left it."""
    paths.SERVER_HOME.mkdir(parents=True, exist_ok=True)
    (paths.SERVER_HOME / "restart-after-update.json").write_text(
        json.dumps({"workspace": str(tmp_path), "stopped_at": time.time() - 7200})
    )

    result = server_stage.run_stage(_emit()[1], "default")

    assert started == []
    assert result.detail["restarted"] is False
    assert not (paths.SERVER_HOME / "restart-after-update.json").exists()


def test_stage_restart_failure_is_reported_not_raised(started, monkeypatch, tmp_path):
    def older(*_a, **_k):
        raise OlderServerError()

    monkeypatch.setattr(manager, "ensure_langgraph_dev", older)
    server_stage.remember_for_restart(str(tmp_path))
    _updated(monkeypatch)

    result = server_stage.run_stage(_emit()[1], "default")

    assert result.detail["restarted"] is False
    assert "Could not start" in result.message


def test_stage_ignores_record_when_update_did_not_install(started, tmp_path):
    """The update failed after it stopped the server: same version."""
    server_stage.remember_for_restart(str(tmp_path))

    result = server_stage.run_stage(_emit()[1], "default")

    assert started == []
    assert result.detail["restarted"] is False


def test_stage_unreadable_restart_record_is_dropped(started):
    paths.SERVER_HOME.mkdir(parents=True, exist_ok=True)
    (paths.SERVER_HOME / "restart-after-update.json").write_text(json.dumps([]))

    server_stage.run_stage(_emit()[1], "default")

    assert started == []
    assert not (paths.SERVER_HOME / "restart-after-update.json").exists()


def test_server_stop_leaves_ccproxy(monkeypatch):
    """Open sessions on the local backend route through it."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(manager, "stop_recorded_server", lambda: 42)
    stopped = []
    monkeypatch.setattr(ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append(1))

    CliRunner().invoke(app, ["server", "stop"])

    assert stopped == []


def test_service_stop_stops_server_and_ccproxy(monkeypatch):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    stopped = []
    monkeypatch.setattr(
        manager,
        "stop_recorded_server",
        lambda **_k: stopped.append("server") or 42,
    )
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append("ccproxy") or 7
    )

    result = CliRunner().invoke(app, ["service", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == ["server", "ccproxy"]


@pytest.mark.parametrize("ccproxy_pid", [7, None])
def test_service_stop_leaves_deploy_owned_server(monkeypatch, tmp_path, ccproxy_pid):
    """EvoSci deploy or the desktop app stops its own server; the ccproxy that
    server may route through stays too."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    stopped = []
    monkeypatch.setattr(
        manager, "RUNTIME", manager.LanggraphRuntimePaths.for_directory(tmp_path)
    )
    monkeypatch.setattr(
        manager,
        "_read_workspace_sidecar",
        lambda: {"workspace": "/a", "pid": 4242, "lasting": False},
    )
    monkeypatch.setattr(manager, "_langgraph_cmdline", lambda pid: ["langgraph"])
    monkeypatch.setattr(
        manager, "_stop_recorded_server_locked", lambda: stopped.append("server") or 42
    )
    monkeypatch.setattr(ccproxy_mod, "recorded_ccproxy_running", lambda: ccproxy_pid)
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append("ccproxy") or 7
    )

    result = CliRunner().invoke(app, ["service", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == []
    output = " ".join(result.output.split())
    assert "not the background service" in output
    assert ("EvoSci ccproxy stop" in output) is (ccproxy_pid is not None)


def test_server_stop_stops_deploy_owned_server(monkeypatch, tmp_path):
    """EvoSci server stop is the explicit way to stop any recorded server."""
    stopped = []
    monkeypatch.setattr(
        manager, "RUNTIME", manager.LanggraphRuntimePaths.for_directory(tmp_path)
    )
    monkeypatch.setattr(
        manager,
        "_read_workspace_sidecar",
        lambda: {"workspace": "/a", "pid": 4242, "lasting": False},
    )
    monkeypatch.setattr(manager, "_langgraph_cmdline", lambda pid: ["langgraph"])
    monkeypatch.setattr(
        manager, "_stop_recorded_server_locked", lambda: stopped.append("server") or 42
    )

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == ["server"]


def test_service_stop_with_lock_held_stops_nothing(monkeypatch):
    """A start in progress (deploy --json, the desktop app) holds the lock:
    the server is left running, and so is the ccproxy it may use."""
    from filelock import Timeout

    import EvoScientist.ccproxy_manager as ccproxy_mod

    class _Busy:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            raise Timeout("langgraph_dev.lock")

        def __exit__(self, *_exc):
            return False

    stopped = []
    monkeypatch.setattr(manager, "FileLock", _Busy)
    monkeypatch.setattr(ccproxy_mod, "recorded_ccproxy_running", lambda: 7)
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append("ccproxy") or 7
    )

    result = CliRunner().invoke(app, ["service", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == []
    assert "try again" in " ".join(result.output.split())


def test_server_start_that_lost_race_says_it_reused(monkeypatch, tmp_path):
    """Another start took the port first: this one only registered its project."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(manager, "ensure_langgraph_dev", lambda *_a, **_k: True)
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda config, *, lasting: None
    )

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "already running" in output
    assert "started from" not in output


def test_server_status_names_server_on_another_port(monkeypatch):
    """langgraph_dev_port changed while the server ran."""
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(
        manager, "_read_workspace_sidecar", lambda: {"workspace": "/a", "pid": 4242}
    )
    monkeypatch.setattr(manager, "_pid_serves_port", lambda *_a: False)
    monkeypatch.setattr(
        manager,
        "_langgraph_cmdline",
        lambda pid: ["langgraph", "dev", "--port", "7000"],
    )

    result = CliRunner().invoke(app, ["server", "status"])

    output = " ".join(result.output.split())
    assert "serves port 7000" in output
    assert "stale record" not in output


def test_service_start_starts_ccproxy_with_server(started, tmp_path, monkeypatch):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    lasting = []
    monkeypatch.setattr(
        ccproxy_mod,
        "maybe_start_ccproxy",
        lambda config, **kw: lasting.append(kw.get("lasting")),
    )

    result = CliRunner().invoke(app, ["service", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert lasting == [True]
    assert started == [(tmp_path, "langgraph_server")]
    assert "EvoSci service stop" in " ".join(result.output.split())


def test_server_start_oauth_warns_without_ccproxy(started, tmp_path, monkeypatch):
    """Only the server: ccproxy is its own command."""
    from types import SimpleNamespace

    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(
        "EvoScientist.config.get_effective_config",
        lambda: SimpleNamespace(
            anthropic_auth_mode="oauth",
            openai_auth_mode="api_key",
            ccproxy_port=8000,
            default_workdir="",
            langgraph_dev_port=6174,
            langgraph_dev_host="127.0.0.1",
        ),
    )
    monkeypatch.setattr(ccproxy_mod, "setup_ccproxy_env", lambda port: None)
    monkeypatch.setattr(ccproxy_mod, "is_ccproxy_running", lambda port: False)

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "ccproxy is not running" in " ".join(result.output.split())


def test_ccproxy_start_without_oauth_does_nothing(monkeypatch):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    started = []
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda *a, **k: started.append(1)
    )

    result = CliRunner().invoke(app, ["ccproxy", "start"])

    assert result.exit_code == 0, result.output
    assert started == []
    assert "not needed" in " ".join(result.output.split())


def test_stop_recorded_ccproxy_keeps_record_when_denied(monkeypatch, tmp_path):
    """It may still be ours; a later stop can try again."""
    import psutil

    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path)
    (tmp_path / "ccproxy.pid").write_text("4242")

    class _Denied:
        def __init__(self, _pid):
            raise psutil.AccessDenied(4242)

    monkeypatch.setattr(psutil, "Process", _Denied)

    assert ccproxy_mod.stop_recorded_ccproxy() is None
    assert (tmp_path / "ccproxy.pid").exists()


def test_update_wait_spares_service_ccproxy(monkeypatch, tmp_path):
    """The update stops it itself, after the wait."""
    import os

    import EvoScientist.ccproxy_manager as ccproxy_mod
    from EvoScientist import updater

    monkeypatch.setattr(ccproxy_mod, "recorded_ccproxy_running", os.getpid)
    monkeypatch.setattr(manager, "_read_workspace_sidecar", lambda: None)

    assert os.getpid() in updater._server_tree()


def test_lost_ccproxy_start_race_is_not_recorded(monkeypatch, tmp_path):
    """The one that lost the port exits; the winner records itself."""
    from types import SimpleNamespace

    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path)

    class _Lost:
        pid = 5

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(ccproxy_mod, "is_ccproxy_available", lambda: True)
    monkeypatch.setattr(ccproxy_mod, "check_ccproxy_auth", lambda _p: (True, ""))
    monkeypatch.setattr(
        ccproxy_mod, "ensure_ccproxy", lambda port, *, detached: _Lost()
    )
    monkeypatch.setattr(ccproxy_mod, "setup_ccproxy_env", lambda port: None)
    config = SimpleNamespace(anthropic_auth_mode="oauth", ccproxy_port=8000)

    ccproxy_mod.maybe_start_ccproxy(config, lasting=True)

    assert not (tmp_path / "ccproxy.pid").exists()
