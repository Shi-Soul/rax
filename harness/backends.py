"""Agent adapters. Import an adapter only when its session needs it."""
from importlib import import_module

NAMES = ('codex', 'claude', 'dsh', 'kimi', 'qoder')
PROVIDERS = {
    'kimi': ('openrouter', 'moonshot-api', 'kimi-account', 'custom'),
    'qoder': ('qoder-account', 'openrouter', 'custom'),
}
MODULES = {
    'codex': 'harness.codex_cli.auto_research',
    'claude': 'harness.claude_cli.auto_research',
    'dsh': 'harness.deepseek_cli.auto_research',
    'kimi': 'harness.kimi_cli.auto_research',
    'qoder': 'harness.qoder_cli.auto_research',
}


def get(name):
    try:
        module = MODULES[name]
    except KeyError:
        raise ValueError(f'Unknown agent backend {name!r}; choose from {NAMES}') from None
    from harness.cli_runtime import CONTAINER
    if CONTAINER and name not in ('codex', 'claude'):
        from harness.container_cli.adapter import Adapter
        return Adapter(name, import_module(module))
    return import_module(module)
