"""Session lifecycle shared by optional agent backends in native and container branches."""
from pathlib import Path

from harness.codex_cli import auto_research as lifecycle

CONTAINER = (Path(__file__).parent/'codex_cli/container_bridge.py').is_file()
CALL_SECONDS = lifecycle.CALL_SECONDS
LOGIN_FILE, PID_FILE = 'agent-login.json', 'agent.pid'
prepare_workspace = lifecycle.prepare_workspace
refresh_reviews = lifecycle.refresh_reviews
open_frontend_log = lifecycle.open_frontend_log

if not CONTAINER:
    from harness.codex_cli.auto_research import (expose_published, native_session,
        start_frontend, stop_services)
else:
    from harness.codex_cli import workspace_storage

    def native_session(root, home_name):
        workspace_storage.remount(root)
        for name in ('agent-workspace', home_name, 'configuration.json'):
            if not (root/name).exists():
                raise FileNotFoundError(root/name)

    def expose_published(root):
        pass  # Container mounts are installed by the launcher.

    # Container adapters own the frontend's container identity and cleanup.
    start_frontend = lifecycle.start_frontend
    stop_services = lifecycle.stop_services


def write_config(path, value):
    from harness.codex_cli import configuration
    if hasattr(configuration, 'write_config'):
        return configuration.write_config(path, value)
    import os
    import uuid
    path = Path(path)
    temporary = path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    try:
        configuration._write_config(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reference_sources(config):
    native = getattr(lifecycle, 'reference_sources', None)
    if native:
        return native(config)
    from services.benchmarks import for_config
    hook = getattr(for_config(config), 'reference_mounts', None)
    return [(source, str(destination).removeprefix('/workspace/'))
            for source, destination in hook(config)] if hook else []
