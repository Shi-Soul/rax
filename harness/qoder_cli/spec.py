"""Pinned Qoder CLI configuration; Qoder owns the agent loop."""
from pathlib import Path

from harness import agent_host
from services.controller.storage import persist

NAME, VERSION = 'qoder', '1.1.78'
PACKAGE = '@qoder-ai/qodercli'
ENTRY = 'bundle/qodercli.js'
ACCOUNT = None
PROVIDERS = ('openrouter', 'custom')
MODEL = 'deepseek/deepseek-v4.1-flash-1017'
READABLE_PATHS = (Path(__file__).resolve().parent,)


def model(provider):
    if provider != 'openrouter':
        raise ValueError('Qoder custom endpoints require an explicit --agent-model')
    return MODEL


def default_binary():
    return agent_host.PROJECT/f'runtime/agent-backends/qoder-cli/tools/qoder-{VERSION}/node_modules/{PACKAGE}/{ENTRY}'


def installation(binary):
    for parent in binary.parents:
        if (parent/f'node_modules/{PACKAGE}/package.json').is_file():
            return parent
    raise ValueError('Qoder CLI must use an isolated npm installation')


def write_home(config, home, selected_model, socket):
    persist(home/'settings.json', {
        'model': {'name': 'rax/'+selected_model, 'reasoningEffort': config.agent_effort},
        'providers': {'rax': {'type': 'openai-compatible', 'protocol': 'openai',
            'baseUrl': 'http://rax-model.invalid/v1', 'apiKey': 'rax-session-transport',
            'model': selected_model, 'models': [{'model': selected_model,
                'contextWindow': 1048576, 'maxOutputTokens': 65536,
                'capabilities': {'tools': True, 'vision': True, 'thinking': {
                    'modes': ['enabled'], 'supportsEffort': True,
                    'supportedEffortLevels': ['low', 'high', 'max']}}}]}},
        'general': {'enableAutoUpdate': False, 'defaultPermissionMode': 'bypass_permissions'},
        'security': {'folderTrust': {'enabled': False}},
        'privacy': {'usageStatisticsEnabled': False},
        'aiCodeStatistics': {'enabled': False},
        'context': {'fileName': 'AGENTS.md'},
        'mcp': {'timeout': 600000}})
    persist(home/'mcp.json', {'mcpServers': {'auto_research': {
        'command': str(agent_host.agent_python()),
        'args': ['-I', '/tmp/rax-bridge/mcp_bridge.py', str(socket)]}}})


def arguments(prompt, resume):
    # An explicit tool list also disables subagents, web access and scheduling.
    return ['--output-format', 'stream-json', '--setting-sources', 'user',
            '--mcp-config', 'mcp.json', '--strict-mcp-config',
            '--tools', 'Read,Write,Edit,Glob,Grep,Bash',
            *(['--resume', resume] if resume else []), '--print', prompt]


def environment(home, *, install):
    return {'QODER_CONFIG_DIR': str(home),
            'QODER_PERSONAL_ACCESS_TOKEN': 'rax-session-transport',
            'QODER_DISABLE_UMID_REPORT': '1'}


def event(value):
    ident = value.get('session_id')
    error = (value.get('result') or value.get('errors') or value.get('error')) if value.get('is_error') else None
    if value.get('type') == 'error':
        error = value.get('message') or value.get('error')
    return ident, error, value.get('usage') if value.get('type') == 'result' else None


def session_files(home, session_id):
    return [p.relative_to(home) for p in sorted((home/'projects').rglob('*.jsonl'))
            if session_id in p.name and not p.is_symlink()]
