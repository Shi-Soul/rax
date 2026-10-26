# Qoder CLI on DexJoCo

The `qoder` backend uses the official `@qoder-ai/qodercli` package at version
`1.1.78`. Qoder runs its own agent loop. RAX supplies the existing workspace,
robot MCP service, model request relay and code formal evaluation.

Qoder requires a Qoder account even when a custom provider supplies the model.
Use a dedicated experiment account whose plan enables custom providers. Create
a personal access token through [Qoder account integrations](https://qoder.com/account/integrations).
Keep it in a file named `qoder-pat`, with mode `0600`, inside an experiment
directory with mode `0700`. Pass that directory as `--agent-auth-home`.
Pass the OpenRouter API key separately as `--agent-key-file`.

The model key and Qoder personal access token stay on the host. The CLI receives
the short-lived account credentials issued by Qoder. The CLI and its shell and
file tools run inside RAX's existing bubblewrap boundary. They cannot reach the
network directly, see the GPU, or read the operator's home. A separate Unix
socket permits only Qoder login, token refresh and catalog requests. It cannot
send inference requests to Qoder. Model requests go through the existing relay,
which enforces the selected model and reasoning effort.

Install the pinned package:

```bash
runtime/envs/host/bin/python scripts/install_native_cli.py \
  --backend qoder \
  --prefix runtime/agent-backends/qoder-cli/tools/qoder-1.1.78 \
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

Reference: [Qoder CLI authentication](https://docs.qoder.com/cli/authentication)
and [CLI parameters](https://docs.qoder.com/cli/cli-reference).
