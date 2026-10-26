"""Vendor-specific configuration; the agent loop is the official Kimi CLI."""
import json
import os
from pathlib import Path
import shutil
import tomllib

from harness import agent_host
from harness.cli_runtime import write_config
from services.controller.storage import persist

NAME, VERSION = 'kimi', '2.2.0'
PACKAGE, ENTRY = '@moonshot-ai/kimi-code', 'dist/main.mjs'
SOURCE_COMMIT = '67325c1b25e46b2fc66043bceab059221b719fa9'
ACCOUNT = 'kimi-account'
PROVIDERS = ('openrouter', 'moonshot-api', ACCOUNT, 'custom')
MODEL = 'moonshotai/kimi-k3'
API_ENDPOINT = 'https://api.moonshot.ai/v1'
ACCOUNT_ENDPOINT = 'https://api.kimi.com/coding/v1'


def model(provider):
    return MODEL if provider == 'openrouter' else 'k3' if provider == ACCOUNT else 'kimi-k3'


def default_binary():
    return agent_host.PROJECT/f'runtime/agent-backends/kimi-code/tools/kimi-{VERSION}/node_modules/.bin/kimi'


def installation(binary):
    for parent in binary.parents:
        if (parent/'node_modules/@moonshot-ai/kimi-code/package.json').is_file():
            return parent
    raise ValueError('Kimi CLI must use an isolated npm installation')


def write_home(config, home, selected_model, socket):
    # There is no account material here. Even native file/shell tools run inside
    # the same outer process boundary; only the host relay can read credentials.
    write_config(home/'config.toml', {
        'default_model': 'rax', 'default_permission_mode': 'auto',
        'builtin_product_skills': False, 'merge_all_available_skills': False,
        'providers': {'rax': {'type': 'openai', 'base_url': 'http://rax-model.invalid/v1',
                              'api_key': 'rax-session-transport'}},
        'models': {'rax': {'provider': 'rax', 'model': selected_model,
            'max_context_size': 1048576, 'max_output_size': 65536,
            'capabilities': ['thinking', 'always_thinking', 'image_in', 'tool_use'],
            'support_efforts': ['low', 'high', 'max'], 'default_effort': 'max'}},
        'thinking': {'enabled': True, 'effort': 'max', 'keep': 'all'},
        'tools': {'disabled': ['Agent', 'Task', 'WebSearch', 'WebFetch', 'Fetch',
                               'AgentSwarm', 'FetchURL', 'CreateGoal', 'GetGoal', 'UpdateGoal', 'SetGoalBudget',
                               'WaitFor', 'CronCreate', 'CronDelete', 'CronList', 'AskUserQuestion', 'EnterPlanMode', 'ExitPlanMode']},
        'cron': {'disabled': True}, 'background': {'keep_alive_on_exit': False},
        'mcp': {'tool_timeout_ms': 600000}})
    persist(home/'mcp.json', {'mcpServers': {'auto_research': {
        'command': str(agent_host.agent_python()), 'args': ['-I', '/tmp/rax-bridge/mcp_bridge.py', str(socket)],
        'toolTimeoutMs': 600000}}})


def arguments(prompt, resume):
    return ['--output-format', 'stream-json',
            *(['--session', resume] if resume else []), '--prompt', prompt]


def environment(home, *, install=None):
    return {'KIMI_CODE_HOME': str(home), 'KIMI_DISABLE_TELEMETRY': '1',
            'KIMI_CODE_NO_AUTO_UPDATE': '1', 'KIMI_DISABLE_CRON': '1',
            'KIMI_CODE_TRUST_WORKSPACE': '1'}


def event(value):
    ident = value.get('session_id') if value.get('type') == 'session.resume_hint' else None
    error = value.get('message', value.get('content')) if value.get('type') == 'error' else None
    return ident, error, value.get('usage')


def session_files(home, session_id):
    # Session stores are private to this single-agent home. Never include the
    # config, device identity, credentials, transport settings or global logs.
    return [p.relative_to(home) for folder in ('sessions', 'workspaces')
            for p in sorted((home/folder).rglob('*')) if p.is_file() and not p.is_symlink()]


def account_environment(home):
    return {k: v for k, v in os.environ.items() if k in ('PATH', 'LANG', 'LC_ALL', 'USER', 'LOGNAME', 'HOME')}


def account_command(binary, home, provider=ACCOUNT):
    config = tomllib.loads((home/'config.toml').read_text())
    provider = config.get('providers', {}).get('managed:kimi-code', {})
    oauth = provider.get('oauth', {})
    return [shutil.which('node'), str(Path(__file__).with_name('account.mjs')),
            str(installation(Path(binary))/'rax-oauth.cjs'), str(home), json.dumps(oauth)]
