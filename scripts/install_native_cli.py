#!/usr/bin/env python3
"""Install an optional, pinned vendor CLI outside the RAX source checkout.

The OpenRouter path needs only this installation. Account authentication helpers
are separate host-only artifacts; they are not needed for OpenRouter trials.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from harness.backends import PROVIDERS


def run(command, *, cwd=None):
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def checkout(source, repository, commit):
    if not (source/'.git').is_dir():
        run(['git', 'init', source])
        run(['git', 'remote', 'add', 'origin', repository], cwd=source)
        run(['git', 'fetch', '--depth', '1', 'origin', commit], cwd=source)
        run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=source)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if actual != commit:
        raise ValueError('Use a separate prefix for a different upstream source commit')


def account_helper(spec, prefix, cache, npm):
    """Bundle the pinned vendor credential resolver for host-only account refresh."""
    source, tooling = prefix/'source', prefix/'build-tools'
    checkout(source, 'https://github.com/MoonshotAI/kimi-code.git', spec.SOURCE_COMMIT)
    run([npm, 'install', '--ignore-scripts', '--no-audit', '--no-fund', '--prefix', tooling,
         '--cache', cache/'npm', 'pnpm@10.33.2', 'esbuild@0.25.12'])
    run([tooling/'node_modules/.bin/pnpm', '--filter', '@moonshot-ai/kimi-code-oauth...', 'install', '--ignore-scripts',
         '--frozen-lockfile', '--store-dir', cache/'pnpm'], cwd=source)
    exports = {'KimiOAuthToolkit': 'packages/oauth/src/toolkit.ts',
               'createKimiDefaultHeaders': 'packages/oauth/src/identity.ts'}
    output = prefix/'rax-oauth.cjs'
    entry = tooling/'account-entry.mjs'
    entry.write_text(''.join('export {'+name+'} from '+json.dumps(str(source/path))+';\n'
                             for name, path in exports.items()))
    run([tooling/'node_modules/.bin/esbuild', entry, '--bundle', '--platform=node',
         '--format=cjs', '--packages=bundle', '--external:koffi', '--outfile='+str(output)])
    return {'helper_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'source_lock_sha256': hashlib.sha256((source/'pnpm-lock.yaml').read_bytes()).hexdigest()}


def install(spec, prefix, cache, *, with_account=False):
    if with_account and spec.NAME != 'kimi':
        raise ValueError('--with-account builds only the Kimi account helper')
    prefix, cache = prefix.resolve(), cache.resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    node, npm = shutil.which('node'), shutil.which('npm')
    if not node or not npm:
        raise RuntimeError('Install Node.js 24 and npm first')
    run([npm, 'install', '--ignore-scripts', '--no-audit', '--no-fund', '--prefix', prefix,
         '--cache', cache/'npm', f'{spec.PACKAGE}@{spec.VERSION}'])
    binary = prefix/'node_modules'/spec.PACKAGE/spec.ENTRY
    lock = prefix/'package-lock.json'
    version = subprocess.check_output([node, str(binary), '--version'], text=True).strip()
    record = {'backend': spec.NAME, 'release': spec.VERSION, 'package': spec.PACKAGE,
        'cli_version': version, 'binary': str(binary), 'node': node,
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest()}
    if spec.NAME == 'kimi':
        record['source_commit'] = spec.SOURCE_COMMIT
    if with_account:
        record['account'] = account_helper(spec, prefix, cache, npm)
    (prefix/'rax-installation.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=tuple(PROVIDERS), required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--with-account', action='store_true', help='Build the vendor account credential helper')
    args = parser.parse_args()
    spec = importlib.import_module('harness.'+args.backend+'_cli.spec')
    print(json.dumps(install(spec, args.prefix, args.cache, with_account=args.with_account), indent=2))


if __name__ == '__main__':
    main()
