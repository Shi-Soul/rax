# RAX on DexJoCo (native branch)

**[Project website](https://robot-agent-exploration.com/)** · [Paper](https://robot-agent-exploration.com/rax.pdf) · [All branches](https://github.com/robot-agent-exploration/rax/tree/main)

This branch runs the DexJoCo experiments of *RAX: Scaling Online Exploration for Robot
Agents*. A coding agent (Codex CLI or Claude Code) learns a dexterous manipulation task by
exploring the simulator through MCP tools. It keeps notes and verified skills in its
workspace, and then is evaluated on the benchmark's official evaluation episodes. Evaluation
either runs the robot program the agent wrote, or runs live copies of the agent.

The agent CLI runs natively on the host, as your user, with its own login. Its own
sandbox confines every command the agent runs: no network, writes only in the session's
workspace, and the repository, other sessions and your credentials hidden. Robot programs
the agent submits run in a bubblewrap sandbox. The agent and its programs have no GPU and
no access to the simulator's ground truth. Robot, Gemini and episode access go through
one host-owned MCP service. (The container branch runs the same experiments with the
agent in an offline Docker container instead.)

## Benchmark

| Backend | Robot | Tasks | Exploration layouts | Formal layouts |
| --- | --- | --- | --- | --- |
| [DexJoCo](https://dexjoco.github.io) | Franka Panda arm (two on bimanual tasks), each with a 16-DoF Allegro hand, official flange-pose control at 30 Hz, the task's official evaluation cameras | the 11 official tasks, each as `<task>` (object randomization) and `<task>_random` (also camera, lighting and table texture): `hammer_nail`, `click_mouse`, `pick_bucket`, `pinch_tongs`, `fold_glasses`, `water_plant`, `bimanual_unlock_ipad`, `bimanual_hanoi`, `bimanual_assembly`, `bimanual_microwave_cook`, `bimanual_photograph` | fresh seeded layouts, never an official evaluation layout | the 150 official evaluation episodes of each task (seeds 0-2, 50 each) |

The simulator is the official DexJoCo package, unmodified, in MuJoCo. It runs natively in its
own conda environment and renders headless on one GPU through EGL; one GPU is enough. The
backend sits behind the harness's benchmark interface: an entry in `services/benchmarks.py`
(its tools, robot facts, setup defaults and capabilities) and the adapter
`services/dexjoco/benchmark.py`. The harness calls only that interface; see
[Adding a backend](docs/ARCHITECTURE.md#adding-a-simulator-backend).

Sessions run in parallel. Each session owns its simulators, its MCP service, its agent
process and its storage, so several sessions (on one or more GPUs) never share state.
Inside a session, exploration can use up to eight simulator environments at once, and
formal evaluation runs several episodes concurrently.

## How a session runs

1. **Explore.** The agent reads `TASK.md` (the official task prompt and, if selected, a
   demonstration), then solves the task by hand: it observes, moves the arm and hand through
   MCP tools, and checks the result. Each exploration episode uses a new layout, and the
   operator sets the episode budget.
2. **Compile.** The agent turns what worked into `code/<project>/controller.py:main(ctx)`,
   a program made of step functions that check what they observe, and tests it unattended.
3. **Evaluate (formal).** One of two modes, chosen when the session is prepared:
   - **Code (default):** the agent registers its program, passes one clean rehearsal, and
     submits it. The frozen program then plays the official evaluation episodes alone.
   - **Agent formal:** after exploration, live copies of the agent each play one evaluation
     episode, carrying their memory and code.

The operator dashboard prepares, launches, monitors, stops and resumes sessions. Every
dashboard action is also a terminal command.

## Quick start

```bash
git clone --branch dexjoco/native https://github.com/robot-agent-exploration/rax.git
cd rax
export RAX_RUNTIME_ROOT=/fast-disk/rax/runtime            # where environments, demos and sessions live
bash scripts/bootstrap_sources.sh                         # pinned official DexJoCo source
bash scripts/setup_host_env.sh                            # the harness's (and the agent's) Python environment
bash scripts/setup_dexjoco_env.sh                         # the official DexJoCo simulator environment
runtime/envs/host/bin/python scripts/setup_dexjoco_demos.py --gpu 0 --task click_mouse   # demonstrations
# Install bubblewrap (and socat for Claude Code), the Codex CLI or Claude Code, and a model
# credential (docs/SETUP.md, steps 0 and 4), e.g. an OpenRouter key in
# ~/.codex/secrets/openrouter_api_key.
bash scripts/doctor.sh                                    # checks the host
runtime/envs/host/bin/python scripts/dexjoco_doctor.py --gpu 0 --task click_mouse   # checks the simulator
bash scripts/run_operator_dashboard.sh                    # prepare and launch sessions
```

The full guides:

- [Qoder CLI](docs/QODER.md): installation, account credentials, validation and the DeepSeek Click Mouse experiment.
- [Agent harnesses](https://github.com/robot-agent-exploration/rax/blob/main/docs/AGENT_HARNESSES.md): Codex CLI, Claude Code, DeepSeek Harness and Kimi Code; providers, authentication and setup.
- [DeepSeek serving](https://github.com/robot-agent-exploration/rax/blob/main/docs/DEEPSEEK_SERVING.md): four-H200 setup, tuning and protocol checks.
- [DeepSeek model results](docs/DEEPSEEK_MODEL_RESULTS.md): DexJoCo task scores, baseline comparisons, configurations, API pricing and Click Mouse harness results.
- [Setup](docs/SETUP.md): host requirements, every installation step, credentials for the
  agent CLIs, the demonstrations, and the tests.
- [Usage](docs/USAGE.md): preparing, launching, monitoring, stopping and resuming
  sessions; agent formal, adaptation and parallel exploration; results and scores.
- [DexJoCo backend](docs/DEXJOCO_BACKEND.md): tasks, layouts, tools, observations, scoring,
  demonstrations and the optional ground-truth episode reviews.
- [Architecture](docs/ARCHITECTURE.md): services, isolation, the MCP contract, the
  evaluation lifecycle, and how to add a backend.

## Repository layout

| Path | Contents |
| --- | --- |
| `auto_research_agent/` | The agent's instructions (`AGENTS.md`, `FORMAL_AGENTS.md`, `START_PROMPT.md`) and its Python API (`api/`) |
| `agent_skills/`, `agent_blocks/` | Every agent skill and shared text block, written once and layered per robot, observation profile and environment count ([README](agent_skills/README.md)) |
| `harness/` | Launchers for Codex CLI and Claude Code, the agent's host process and sandbox paths, the robot-tool bridge, session control |
| `services/controller/` | The per-session MCP frontend, supervisor, bundle sandbox, gateway and formal evaluation |
| `services/dexjoco/` | The DexJoCo backend: host adapter, episode and preview simulator servers, task rules, demonstrations and review analysis |
| `services/episode_reviews/` | Operator-side reviews of closed exploration episodes |
| `services/simulator/` | The RPC between the host and a simulator process |
| `services/exploration/` | Parallel exploration environments |
| `services/agent_formal/` | Live-agent formal evaluation |
| `services/workspace/` | The workspace generator |
| `services/dashboard/` | The operator dashboard ([README](services/dashboard/README.md)) |
| `scripts/` | Setup, launch, stop, dashboard and doctor scripts |
| `local_tests/` | Unit and integration tests ([README](local_tests/README.md)) |

## Acknowledgements

This branch builds on [DexJoCo](https://github.com/brave-eai/dexjoco), its official raw
demonstration dataset (`DexJoCo/DexJoCo-Datasets-Raw` on Hugging Face), and
[MuJoCo](https://mujoco.org). `dependencies.lock` pins every version. The DexJoCo source and
demonstrations are used under their own licenses.

## License

RAX's original source code and accompanying documentation are licensed under the
[Apache License 2.0](LICENSE). Copyright 2026 RAX contributors.

Third-party code, simulator software, assets, model weights and datasets retain their
own licenses. See [NOTICE](NOTICE) and the relevant upstream releases.

## Citation

If you find RAX useful, please cite:

```bibtex
@misc{huang2026scaling,
  title        = {Scaling Online Exploration for Robot Agents},
  author       = {Huang, Runhan and Zhang, Xiangcheng and Chen, Haoyu
                  and Li, Yitong and Wang, Yen-Jen and Lyu, Bohan
                  and Ye, Haotian and Hong, Chuye and Wang, Yixuan
                  and Su, Chloe and Xiong, Haoyu and Wu, Yangzhen
                  and Xie, Enze and Manderson, Travis and Ermon, Stefano
                  and Abbeel, Pieter and Torralba, Antonio and Song, Dawn
                  and Ye, Weirui and Du, Yilun},
  year         = {2026},
  howpublished = {Preprint},
  url          = {https://robot-agent-exploration.com/}
}
```
