"""A client's side of the project registry: the folder check every launcher
runs first (:func:`prepare_workdir`), and the registration of its project
with the langgraph dev server it uses (:func:`register_folder`).

The server serves a project once it is registered and while its folder
exists. Registering an already registered project is cheap, so a client
registers its project every time it starts or reuses a server.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import httpx

from ..paths import Workspace

# Seconds a registration may take: the first one restores the project's
# threads, which reads every checkpoint in sessions.db.
REGISTRATION_TIMEOUT = 120.0
# Seconds the read of the server's instance may take.
INSTANCE_TIMEOUT = 10.0


class ServerUnusableError(RuntimeError):
    """This client cannot use the running server for its project: the client
    stops, or keeps the session it has."""


class OlderServerError(ServerUnusableError):
    """The running server predates the project registry."""

    def __init__(self) -> None:
        super().__init__(
            "The running server is from an older EvoScientist. Stop it with "
            "`EvoSci server stop`, then start again."
        )


class ProjectFolderMissingError(ServerUnusableError):
    """A folder the registry knows as a project is gone."""

    def __init__(self, root: Path) -> None:
        super().__init__(
            f"{root} used to be an EvoScientist project, and its threads and "
            "scheduled tasks are still stored. Restore the folder to keep "
            "working in it, or create it empty to start again there."
        )


class RegistrationError(RuntimeError):
    """The server did not register the project. The client goes on without
    it, as when the server does not start."""


def is_known_project(workspace: Workspace) -> bool:
    """Whether the registry (``sessions.db``) knows *workspace*, registered or
    not. Read only; a database or table not created yet knows nothing."""
    from .. import paths
    from ..sessions import get_db_path

    # ``get_db_path`` creates the data folder; a missing one knows nothing.
    if not paths.DATA_DIR.is_dir():
        return False
    try:
        conn = sqlite3.connect(
            f"{get_db_path().as_uri()}?mode=ro", uri=True, timeout=30.0
        )
    except sqlite3.Error:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM workspaces WHERE workspace_dir = ?", (workspace.key,)
        ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return row is not None


def refuse_missing_project(path: str | Path) -> None:
    """Raise :class:`ProjectFolderMissingError` when *path* is missing but
    the registry knows it as a project: recreating it empty would hide the
    project's threads and scheduled tasks behind a new, empty project."""
    folder = Path(path)
    if folder.is_dir():
        return
    workspace = Workspace(folder)
    if is_known_project(workspace):
        raise ProjectFolderMissingError(workspace.root)


def prepare_workdir(path: str | Path) -> Path:
    """The folder a launcher works in, created when it is new; see
    :func:`refuse_missing_project`."""
    folder = Path(path)
    refuse_missing_project(folder)
    os.makedirs(folder, exist_ok=True)
    return folder


def _other_server(url: str) -> str:
    from .. import paths

    return (
        f"The EvoScientist server at {url} does not hold the local secret in "
        f"{paths.SERVER_HOME}: it uses another data folder "
        "(EVOSCIENTIST_DATA_DIR), runs for another user, or could not load its "
        "secret (see its log)."
    )


def _server_instance(url: str) -> str | None:
    """The instance the server at *url* names (``GET /api/instance``)."""
    from .sdk import LANGGRAPH_DEV_AUTH_HEADERS

    try:
        response = httpx.get(
            f"{url}/api/instance",
            headers=LANGGRAPH_DEV_AUTH_HEADERS,
            timeout=INSTANCE_TIMEOUT,
            trust_env=False,
        )
    except httpx.HTTPError as exc:
        raise RegistrationError(
            f"Could not reach the EvoScientist server at {url}: {exc}"
        ) from exc
    if response.status_code == 404:
        raise OlderServerError()
    if response.status_code != 200:
        raise RegistrationError(
            f"The EvoScientist server at {url} did not answer GET /api/instance "
            f"(HTTP {response.status_code})."
        )
    try:
        body = response.json()
    except ValueError:
        body = None
    return body.get("instance") if isinstance(body, dict) else None


def register_folder(workspace: Workspace, *, url: str) -> None:
    """Register *workspace* (a project root, never a run folder) with the
    langgraph dev server at *url*.

    The local secret goes only to a server that shows it holds the same one
    (its ``instance``): another user's server on the port never gets it.

    Raises :class:`OlderServerError` for a server without the registry,
    :class:`ProjectFolderMissingError` when the server finds the folder
    missing, and :class:`RegistrationError` for anything else that kept the
    server from registering it.
    """
    from .. import paths
    from ..local_secret import HEADER, instance_of, read_local_secret
    from .sdk import LANGGRAPH_DEV_AUTH_HEADERS

    url = url.rstrip("/")
    # First: a server too old to have created the secret has no instance.
    instance = _server_instance(url)
    secret = read_local_secret()
    if secret is None:
        raise RegistrationError(
            f"Cannot read the server's local secret in {paths.SERVER_HOME}. "
            "Is EVOSCIENTIST_DATA_DIR the same as the server's, on a file "
            "system that keeps permissions?"
        )
    if instance != instance_of(secret):
        raise RegistrationError(_other_server(url))
    try:
        response = httpx.post(
            f"{url}/api/workspaces",
            json={"workspace_dir": workspace.key},
            headers={**LANGGRAPH_DEV_AUTH_HEADERS, HEADER: secret},
            timeout=REGISTRATION_TIMEOUT,
            # Loopback: a system proxy must not intercept it.
            trust_env=False,
        )
    except httpx.HTTPError as exc:
        raise RegistrationError(
            f"Could not register {workspace.root} with the EvoScientist server "
            f"at {url}: {exc}"
        ) from exc
    if response.status_code == 200:
        return
    try:
        body = response.json()
    except ValueError:
        body = {}
    error = body.get("error") if isinstance(body, dict) else None
    if error == "folder_missing":
        if not workspace.root.is_dir() and is_known_project(workspace):
            raise ProjectFolderMissingError(workspace.root)
        raise RegistrationError(
            f"The EvoScientist server at {url} cannot see the folder {workspace.root}."
        )
    if error == "local_secret_required":
        raise RegistrationError(_other_server(url))
    detail = body.get("detail") if isinstance(body, dict) else None
    reason = ": ".join(str(part) for part in (error, detail) if part)
    raise RegistrationError(
        f"The EvoScientist server at {url} did not register {workspace.root} "
        f"(HTTP {response.status_code}{f', {reason}' if reason else ''})."
    )
