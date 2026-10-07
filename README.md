# Healthcare_Agent
What happens to each alert

Alert arrives with a window of vitals (HR, SpO2, BP, ECG) for one synthetic patient.
Graph context adds deterministic facts before any model runs, such as medications that change expected vitals, the device in use, and known artifact patterns. No LLM call is needed.
Jev triage gate answers questions about that state and returns probabilities: real deterioration, sensor artifact, unit error, tampering, or normal, plus an escalation probability.
Confident: the system outputs the label and a short rationale directly. This is the cheap, fast path.
Uncertain: the case goes to the LLM agent, which calls the MCP tools (vitals queries, detectors, cross-signal checks, guideline RAG, graph queries) over several steps. A guardrail checks tool calls and tool results for injected instructions.
Output: a decision with the evidence it used, or an escalation to a human. When the agent isn't sure, escalating is the correct answer.

Offline pipeline that builds and proves it

Generate data: synthetic patients, then the fault injector plants known causes, so every case has a ground-truth label.
Build the tools: the MCP server, graph, and local RAG index.
Run the benchmark: the same cases through four setups (a prompted frontier model, a cheap model, your fine-tuned small model, and the Jev-gated hybrid). Responses are cached and temperature is 0.
Fine-tune: generate correct tool-calling trajectories, train the QLoRA model on them, and test it on held-out patients and fault settings.
Score: accuracy, unsafe-miss rate, calibration, cost and latency, injection catch rate, and the RAG and graph ablations.
Gate in CI: a regression check fails the build if a change drops the score.
Publish: results and traces are exported as static files that the React frontend displays (results page, case replay, architecture view), with live mode as an optional extra.
