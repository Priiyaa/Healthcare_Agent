# Alarm Triage Agent

An agent that triages patient-monitor alerts on **synthetic** data and is scored against **injected ground truth**, not LLM-judge opinions.

> Research and portfolio project on non-clinical, synthetic data. It is decision support research, not a medical device, and it must not be used for patient care.

## The question

Hospital monitors raise many alarms that are really sensor artifacts or data errors (alarm fatigue). Can a cheap, calibrated classifier handle the easy alerts and hand only the uncertain ones to an LLM agent, without ever dismissing a real deterioration?

For each alert (a 120-second window of HR, SpO2 and systolic BP plus patient context) the system outputs one of:

`normal` · `real_deterioration` · `sensor_artifact` · `unit_error` · `tampering` · `escalate` (hand to a human)

## How it works

```
alert -> graph context lookup -> Jev triage gate --confident--> decision + rationale
                                       |
                                       +--uncertain--> LLM agent <-> MCP tool server
                                                        (guardrail on tool calls/results)
                                                                  -> decision + evidence
```

1. **Graph context** (deterministic): medications, device and known artifact types become facts in the state, so the cheap classifier gets better inputs without an LLM call.
2. **Jev triage gate**: a System One classifier (TypeSafe AI) that returns calibrated probabilities per label. Confident cases are decided directly.
3. **LLM agent**: uncertain cases, or cases where real deterioration is plausible but not predicted, go to a plain tool-calling loop over an **MCP server** (vitals, detectors, cross-signal check, patient context, graph query, guideline retrieval).
4. **Guardrail**: tool results are screened for instruction-like text (injection tests use this path).

## Why the evals are credible

- **Ground truth by construction.** A fault injector plants known causes (flatline, spike, dropout, unit swap, multi-channel tampering, genuine multi-signal deterioration), so every case has a true label.
- **No label leakage.** Tools and models see observations only. A test asserts that tool outputs never contain labels or fault parameters.
- **No patient leakage.** Splits are assigned per patient, so no patient appears in more than one split.
- **Metrics that match the risk:**
  - accuracy
  - **unsafe-miss rate**: real deterioration dismissed as normal, artifact or unit error (tampering and escalate still reach a human, so they are not misses)
  - false-alarm rate and escalation rate
  - calibration (expected calibration error)
  - cost and latency per case
- **Deferral curve:** accuracy and unsafe-miss rate versus the fraction of cases sent to the agent, with cost. This is the headline chart.
- **CI regression gate:** the build fails if unsafe-miss rate or accuracy crosses a threshold.

## Planned comparisons and ablations

| Comparison | Question |
|---|---|
| Prompted frontier model vs cheap model vs fine-tuned small model vs Jev-gated hybrid | Does the hybrid match accuracy at lower cost and latency? |
| No retrieval / no graph (baseline) | Reference point |
| + guideline RAG only | Does retrieval improve decisions or explanations? |
| + flat FHIR-style lookup only | Is the graph better than a simple table lookup? |
| + knowledge graph only | Do multi-hop facts help the gate? |
| + both | Do they add up? |

If the flat lookup matches the graph, the write-up says so and the simpler option stays.

## Results

**Not run yet.** This table gets filled from `results/` after the benchmark runs. Nothing below is a measured result.

| System | Accuracy | Unsafe-miss rate | ECE | Deferral rate | Cost / case | Latency |
|---|---|---|---|---|---|---|
| Rule-based baseline | TBD | TBD | TBD | n/a | $0 | TBD |
| Frontier model (agent) | TBD | TBD | n/a | n/a | TBD | TBD |
| Cheap model (agent) | TBD | TBD | n/a | n/a | TBD | TBD |
| Fine-tuned small model | TBD | TBD | n/a | n/a | TBD | TBD |
| Jev gate + agent | TBD | TBD | TBD | TBD | TBD | TBD |

## Quickstart

```bash
pip install -e ".[dev]"            # core + tests
python -m triage.data.make_dataset --out data/cases.jsonl
pytest -q
python -m triage.evals.run --system heuristic --split test \
    --max-unsafe-miss 0.05 --min-accuracy 0.80
```

Optional extras:

```bash
pip install -e ".[mcp]"            # MCP server
python -m triage.mcp_server --cases data/cases.jsonl

pip install -e ".[llm]"            # agent loop (needs ANTHROPIC_API_KEY)
pip install -e ".[jev]"            # Jev gate (needs TYPESAFE_API_KEY)
```

## Repo layout

```
configs/models.yaml          model IDs, gate thresholds, eval settings (one-line swaps)
data/kg_edges.csv            clinical knowledge-graph edges (each needs a citation)
data/guidelines/             local guideline corpus for retrieval
src/triage/schema.py         labels, case format, ground-truth separation
src/triage/data/             synthetic vitals + fault injector, dataset builder
src/triage/features.py       window features, text state for the classifier
src/triage/gate.py           rule-based gate, Jev gate, defer rules
src/triage/graph.py          NetworkX knowledge graph
src/triage/tools.py          tool layer + injection screening
src/triage/mcp_server.py     MCP server over the tools
src/triage/agent.py          plain tool-calling agent loop
src/triage/evals/            metrics and eval runner
tests/                       injector, tools (no label leakage), metrics, gate
.github/workflows/ci.yml     tests + regression gate
web/                         frontend plan (React/TypeScript, replay mode first)
```

## Status

This is an early scaffold.

- Written: label schema, fault injector, dataset builder, rule-based gate, tool layer, MCP server, agent loop, metrics, eval runner, tests, CI.
- **Not yet verified:** the test suite has not been run (dependency installation failed in the build environment), so treat all code as untested until `pytest` passes locally.
- **Not yet exercised:** the Jev gate and the agent loop have not been run against live APIs. Confirm the Jev request/response shapes, pricing and rate limits in the TypeSafe docs, and run a ~30-case pilot to check cost per case before the full benchmark.
- **Placeholders to replace before publishing:** every knowledge-graph edge has `source = TODO`, and the guideline corpus is a placeholder. Add real open references and record licenses in `docs/SOURCES.md`.

## Roadmap

1. Run tests locally and tune the rule-based baseline.
2. Benchmark API models with cached responses at temperature 0; add the Jev gate and the deferral curve.
3. Add guideline retrieval and graph ablations with a context-dependent test subset.
4. Generate teacher trajectories, run one QLoRA SFT on a 3B-8B open model, evaluate offline on held-out patients and fault settings.
5. Add tool-result injection tests and the guardrail comparison.
6. Build the frontend (results page, case replay, architecture) and deploy; keep replay as the default demo.
7. Write up what the evals caught and where the system fails.

## Data and licensing

- All training and evaluation data is synthetic (generated here, optionally supplemented by Synthea).
- Check each PhysioNet demo dataset's license before using or redistributing it, and keep it eval-only if the terms are unclear.
- Credentialed datasets such as full MIMIC must not be sent to external LLM APIs, so they are out of scope.
- Jev is a third-party API: send synthetic data only.
