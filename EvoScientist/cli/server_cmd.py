"""``EvoSci server`` — start, inspect and stop the background langgraph dev
server.

The server outlives the sessions that use it, so it needs an explicit way to
start, see and stop it. ``EvoSci service`` (:mod:`.service_cmd`) covers it
together with ccproxy.
"""

from __future__ import annotations

import sys

import typer  # type: ignore[import-untyped]
from rich.markup import escape

from ..stream.console import console
from ._app import server_app


def manual_stop_command(port: int) -> str:
    """The shell command that stops whatever listens on *port*, for a server
    EvoSci has no ownership record for."""
    if sys.platform == "win32":
        # Feed the PID to Stop-Process via a subexpression, not a $_
        # pipeline: when the user pastes this at a PowerShell prompt, the
        # outer shell expands $_ (empty there) before the child powershell
        # runs, so a $_-based command would stop nothing. (...) is literal
        # inside the double-quoted argument, so it reaches the child intact.
        # -State Listen + -Unique avoids stray/duplicate PIDs from
        # established connections on the same port.
        return (
            f'powershell "Stop-Process -Id (Get-NetTCPConnection '
            f"-LocalPort {port} -State Listen | Select-Object "
            f'-ExpandProperty OwningProcess -Unique) -Force"'
        )
    return f"kill $(lsof -ti :{port})"


@server_app.command("start")
def server_start(
    workdir: str | None = typer.Option(
        None,
        "--workdir",
        help="The project to start it from (default: default_workdir, else "
        "the current folder)",
    ),
) -> None:
    """Start the background langgraph dev server and leave it running, for
    every EvoSci session to share; `EvoSci server stop` stops it. Only the
    server: `EvoSci service start` also starts ccproxy."""
    start_background(workdir, with_ccproxy=False)


def start_background(workdir: str | None, *, with_ccproxy: bool) -> None:
    """``EvoSci server start`` / ``EvoSci service start``: start the server,
    or reuse a running one, for a project, and with *with_ccproxy* also the
    ccproxy it uses with Anthropic OAuth."""
    from ..ccproxy_manager import is_ccproxy_running, uses_oauth
    from ..config import get_effective_config
    from ..langgraph_dev import manager
    from ..langgraph_dev.registration import ServerUnusableError
    from ..langgraph_dev.sdk import langgraph_dev_url
    from ..paths import start_workspace_path
    from .commands import (
        _prepare_workdir_or_exit,
        _print_server_notices,
        _require_chatgpt_sign_in,
    )

    config = get_effective_config()
    _require_chatgpt_sign_in(config, interactive=False)
    project = _prepare_workdir_or_exit(
        start_workspace_path(workdir, config.default_workdir)
    )
    oauth = uses_oauth(config)
    port, host = manager.configured_address(config)
    was_running = manager.is_langgraph_dev_running(port=port, host=host)
    try:
        with console.status(
            "[dim]Starting the background agent server...[/dim]", spinner="dots"
        ):
            served = manager.serve_project(config, project, with_ccproxy=with_ccproxy)
    except (ServerUnusableError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if not served:
        console.print(
            "[red]The background agent server did not start or did not register "
            f"{escape(str(project))}; see {escape(str(manager.RUNTIME.log_file))}."
            "[/red]"
        )
        raise typer.Exit(1)
    stop = "EvoSci service stop" if with_ccproxy else "EvoSci server stop"
    start_cmd = "EvoSci service start" if with_ccproxy else "EvoSci server start"
    url = langgraph_dev_url(config)
    if was_running:
        console.print(
            f"[green]✓[/green] The background agent server at {url} was already "
            f"running; it serves {escape(str(project))} too."
        )
        if not manager.is_lasting(manager._read_workspace_sidecar()):
            console.print(
                "[yellow]⚠ It is not the background service: an EvoSci deploy, "
                "the desktop app, an older release or another program runs it, "
                "and it may stop with its owner; then the next session or "
                f"[bold]{start_cmd}[/bold] starts the background one.[/yellow]"
            )
    else:
        console.print(
            f"[green]✓[/green] Background agent server running at {url}, started "
            f"from {escape(str(project))}. It keeps running for every EvoSci "
            f"session until [bold]{stop}[/bold]."
        )
    if oauth and not with_ccproxy and not is_ccproxy_running(config.ccproxy_port):
        console.print(
            "[yellow]⚠ Anthropic uses OAuth, and ccproxy is not running: start it "
            "with [bold]EvoSci ccproxy start[/bold] (or use [bold]EvoSci service "
            "start[/bold]).[/yellow]"
        )
    _print_server_notices()


@server_app.command("status")
def server_status() -> None:
    """Show the background langgraph dev server's state."""
    from ..config import get_effective_config
    from ..langgraph_dev.manager import (
        _pid_serves_port,
        _read_workspace_sidecar,
        configured_address,
        is_langgraph_dev_running,
    )

    config = get_effective_config()
    port, host = configured_address(config)
    running = is_langgraph_dev_running(port=port, host=host)
    sidecar = _read_workspace_sidecar()

    if not running and sidecar is None:
        console.print("[dim]No background langgraph dev server is running.[/dim]")
        return
    state = "[green]running[/green]" if running else "[red]not responding[/red]"
    console.print(f"[bold]langgraph dev[/bold] on port {port}: {state}")
    if sidecar is not None:
        console.print(f"  started from: {sidecar.get('workspace')}")
        pid = sidecar.get("pid")
        if _pid_serves_port(pid, port):
            console.print(f"  pid:          {pid}")
        else:
            console.print(
                f"  pid:          {pid} [yellow](stale record — this pid does "
                f"not serve port {port})[/yellow]"
            )
    elif running:
        console.print(
            "  [yellow]no ownership record — externally managed or older server"
            "[/yellow]"
        )


@server_app.command("stop")
def server_stop() -> None:
    """Stop the background langgraph dev server started by EvoSci."""
    from ..config import get_effective_config
    from ..langgraph_dev.manager import (
        configured_address,
        is_langgraph_dev_running,
        stop_recorded_server,
    )

    pid = stop_recorded_server()
    if pid is not None:
        console.print(f"[green]✓[/green] Stopped langgraph dev (pid {pid}).")
        return
    config = get_effective_config()
    port, host = configured_address(config)
    if is_langgraph_dev_running(port=port, host=host):
        # A server without ownership records (crashed session, deleted state
        # files) can't be verified as ours — refuse to guess, hand the user
        # the manual path instead of a silent no-op.
        console.print(
            f"[yellow]⚠ A langgraph dev is still serving port {port}, but "
            f"EvoSci has no ownership record for it, so it was not "
            f"touched.[/yellow]"
        )
        console.print(
            f"[dim]If it is yours, stop it manually: "
            f"[bold]{manual_stop_command(port)}[/bold][/dim]"
        )
    else:
        console.print(
            "[dim]No EvoSci-owned langgraph dev server to stop "
            "(stale state, if any, was cleaned up).[/dim]"
        )
