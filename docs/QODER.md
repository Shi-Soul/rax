# Qoder CLI CN on DexJoCo

The `qoder` backend uses the official `@qodercn-ai/qoderclicn` package at version
`1.1.78`. Qoder runs its own agent loop. RAX supplies the existing workspace,
robot MCP service, model request relay and code formal evaluation.

Qoder CLI CN requires a Chinese Qoder account even when a custom provider
supplies the model. Use a dedicated experiment login directory with mode `0700`.
After installation, sign in through the browser:

```bash
QODERCN_CONFIG_DIR=/private/experiment-qoder \
  node runtime/agent-backends/qoder-cli/tools/qodercn-1.1.78/node_modules/@qodercn-ai/qoderclicn/bundle/qoderclicn.js login
```

Pass that directory as `--agent-auth-home`. RAX mounts only its `.auth` store
inside the CLI's private home, so Qoder can read and refresh its browser login.
The rest of the operator's home stays hidden. The account must permit custom
providers. Pass the OpenRouter API key separately as `--agent-key-file`.

For unattended PAT authentication, create a personal access token through
[Qoder CN account integrations](https://qoder.cn/account/integrations). Store it
in a `qoder-pat` file with mode `0600` in the auth directory. A PAT takes
precedence over browser login and stays on the host. The CLI receives only the
account credentials issued by Qoder.

The model key stays on the host. The CLI and its shell and file tools run inside
RAX's existing bubblewrap boundary. They cannot reach the network directly or
see the GPU. A separate Unix socket permits only Qoder CN login, token refresh
and catalog requests. It cannot send inference requests to Qoder. Model requests
go through the existing relay, which enforces the selected model and reasoning
effort.

Qoder's bundled HTTP client needs a separate transport adapter. A Node.js load
hook checks the CN `1.1.78` bundle's SHA-256 hash and routes its HTTP entry point
through the same Unix sockets. It does not change the installed package. An
unknown bundle is rejected before execution. The startup regression test runs
the real CLI against a local authentication fixture; live account and model
validation still requires the credentials below.

Install the pinned package:

```bash
runtime/envs/host/bin/python scripts/install_native_cli.py \
  --backend qoder \
  --prefix runtime/agent-backends/qoder-cli/tools/qodercn-1.1.78 \
  --cache /tmp/rax-qoder-install
```

Before a model experiment, run the real CLI validation. It checks MCP calls,
image transmission, native file and shell tools, conversation resume and process
isolation. It makes paid model calls but uses no robot exploration episodes.

```bash
runtime/envs/host/bin/python scripts/validate_native_cli.py \
  --backend qoder --provider openrouter \
  --root runtime/validation/qoder-flash-1017 \
  --auth-home /private/experiment-qoder \
  --key-file /private/openrouter.key \
  --model deepseek/deepseek-v4.1-flash-1017 \
  --image /path/to/robot.jpg --fixture-workspace
```

Prepare the Click Mouse configuration used by the repository's DeepSeek
comparison: 16 exploration episodes, 150 official evaluation episodes, four
formal workers, one exploration environment, no subagents, RGB-D observations,
three terminal-state goal images and no expert action demonstrations.

```bash
runtime/envs/host/bin/python scripts/configure_auto_research.py \
  --benchmark dexjoco --task click_mouse --sim-gpu GPU-UUID \
  --episodes 16 --formal-episodes 150 --formal-workers 4 \
  --agent-cli qoder --agent-provider openrouter --agent-effort max \
  --exploration-envs 1 --subagents 0 --external-perception off \
  --observation-profile rgbd --demonstration-context terminal_state \
  --expert-demonstrations 0 --episode-reviews off

bash scripts/start_auto_research_agent.sh \
  --config /path/printed/by/configure/operator-input.json --non-interactive \
  --agent-model deepseek/deepseek-v4.1-flash-1017 \
  --agent-auth-home /private/experiment-qoder \
  --agent-key-file /private/openrouter.key
```

Use `--resume` with the same configuration to continue an existing session.
Agent formal evaluation, adaptation and subagents are not enabled for this backend.
The SLURM submission script also accepts `--backend qoder --auth-home ...` and
requires a matching successful validation report.

Reference: [Qoder CLI authentication](https://docs.qoder.cn/cli/authentication)
and [CLI parameters](https://docs.qoder.cn/cli/cli-reference).
