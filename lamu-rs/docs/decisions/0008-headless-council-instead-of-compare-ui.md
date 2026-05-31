# ADR 0008: Provide multi-model comparison as a headless judged council, not a compare UI

## Status
Accepted 2026-05-31

## Context
LAMU dispatches one prompt across many models — local GGUF backends and cloud endpoints share a single dispatch path (`council.rs:67-89` reuses the same local/cloud branch as `parallel_query`). A recurring want is "run this against several models and tell me which is best." The conventional shape for that is a graphical side-by-side compare panel where a human reads two or more outputs and clicks a winner (the pattern in tools like Odysseus's blind-compare feature).

But LAMU has no UI surface and is positioned not to grow one (ADR 0002: lean backend, no batteries-included framework; ADR 0001: MCP-first, HTTP is a thin compat shim). Every caller is an automated one — the outer Claude Code agent or a harness driving the `local-llm` MCP server over stdio. There is no human in the loop at request time to eyeball outputs and click. A compare feature whose output is "two panels for a person to read" produces nothing an agent can act on programmatically: the agent needs a single answer it can paste into the next step, not a layout. The fan-out and concurrency machinery already existed (`parallel_query`, `handlers.rs:664`), so the open question was purely the *output contract*: raw N answers, a UI, or a machine-consumable verdict.

## Decision
The multi-model comparison capability ships as the headless `council` MCP tool (`tools.rs:602-608`, registered MCP-only — there is no `council` route in `lamu-api`). `handle_council` (`council.rs:44`) fans the same prompt to N>=2 models concurrently (`council.rs:67-89`), anonymizes the survivors as blind labels A, B, C, … (`council.rs:108-115`), and sends them to a judge model (`mimo-v2.5-pro` by default, `council.rs:50`) under a fixed `JUDGE_PROMPT` (`council.rs:10-15`) that requires the judge to pick the single best blind answer AND synthesize a final answer combining their strengths, returned as JSON `{"best","synthesis","reasoning"}`. The tool returns structured text: roster, `Winner: <model> [<letter>]`, the synthesis, and — gated on `include_answers` (default true, `council.rs:51,145`) — each member's full answer. The judge verdict, not a human click, is the primary output.

## Rationale
- The caller is an agent, not a person. An agent consumes a single synthesized answer it can act on; a graphical compare panel produces nothing it can read. The output contract is built for machine consumption — `Winner` + `Synthesis` text, parseable, with raw answers as optional backing evidence (`council.rs:126-150`).
- Blind labeling (`council.rs:108-115`) removes model-identity bias from the judge the same way a human blind compare would, but without needing a human. The judge sees "Answer A / Answer B", never the model names; the roster is reattached only in the returned text for the caller's audit (`council.rs:128-130, 135-139`).
- Synthesis beats pick-one for an automated pipeline. A human comparing two outputs mentally merges them; an agent can't. The judge prompt explicitly demands a merged-and-corrected final answer (`council.rs:12-13`), so the council can output something better than any single member rather than just routing to a winner.
- It reuses existing fan-out. `council` is "the same local+cloud dispatch as `parallel_query`" (`council.rs:4-5`) plus a judge stage — no new serving infrastructure, consistent with ADR 0002's lean stance.
- It honors routing mode without a special case. Under `local-only`, cloud members are refused per-member and the judge runs over the survivors (`council.rs:61-64, 77-80`), consistent with the routing policy in ADR 0007 / 0010.
- A UI would contradict the product's spine. ADR 0001/0002 commit to no graphical surface; a compare panel would be the first one and would need its own serving/asset story.

## Alternatives Considered
- **Graphical blind-compare panel (Odysseus-style):** a web UI showing N outputs side by side for a human to rank. Rejected because the request-time caller is an agent over MCP stdio, not a browser — there is no human to read the panels or click, so the output would be unusable to the actual consumer. It also requires a UI/asset-serving surface LAMU explicitly does not have (ADR 0001 makes HTTP a thin shim; ADR 0002 forbids the framework). The blind-bias benefit a UI provides is captured headlessly by anonymized labels (`council.rs:108-115`) instead.
- **No compare capability at all:** rely on the agent to call models one at a time and reason over outputs itself. Rejected because it burns expensive outer-agent tokens (Opus-tier) doing the cross-read and merge that a cheap judge model (`mimo-v2.5-pro`) can do in one call, and it loses the structured blind protocol — the agent would see model identities and inherit bias.
- **Return raw N outputs, no judge (i.e. just `parallel_query`):** `parallel_query` already fans out and returns all answers (`handlers.rs:664`, `tools.rs:798`). Rejected as the comparison primitive because it pushes the synthesize/rank step back onto the caller — the very work an automated caller can't cheaply do — and produces no single actionable answer. `parallel_query` is kept for the independent-tasks case; `council` is one-prompt-many-models-one-verdict. The `include_answers` flag (`council.rs:145`) preserves the raw-output view as optional backing data when the caller wants to inspect.

## Consequences
- The judge is a hard dependency and a single point of failure. The verdict comes from one model call (`council.rs:116-124`); if its output is unparseable, `parse_judge_verdict` (`council.rs:19-42`) returns `None` and the tool degrades to dumping the raw judge reply (`council.rs:142`) rather than a clean winner. The parser is deliberately tolerant (code fences, prose-wrapped, first `{...}` fallback) but cannot recover from genuinely malformed JSON.
- Judge cost and latency are added to every comparison: the judge call always goes to cloud (`handle_cloud_query`, `council.rs:116`) and runs after the fan-out completes (`council.rs:89` joins all members first), so wall time is `max(members) + judge`, not `max(members)`.
- The default judge `mimo-v2.5-pro` is a cloud model (`council.rs:50`). Under `local-only` the members are filtered to local survivors but the judge call itself is not routing-gated in `handle_council`, so a `local-only` council still issues a cloud judge request — a latent inconsistency with the per-member refusal logic that a future change may need to close.
- Winner attribution is heuristic: `best.contains(*c)` (`council.rs:136-137`) maps the judge's letter back to a model by substring match. A judge that replies with a stray uppercase letter in `best` could mis-map; on no match it falls back to printing the raw `best` string (`council.rs:139`).
- We commit to never building a human compare UI for this; if a human ever needs side-by-side, they read the `include_answers` block in the tool's text output. This keeps LAMU on the no-UI spine (ADR 0001/0002).
- Requires >=2 members and >=2 *successful* answers, else it errors out with a per-model ok/err breakdown (`council.rs:57-59, 91-106`) — a council of one is meaningless, so partial-failure down to one survivor is treated as failure rather than silently returning that survivor.

## Related Decisions
ADR 0001 (MCP-first; council is MCP-only, no HTTP route), ADR 0002 (lean no-UI backend — the reason this is headless), ADR 0007 (unified cloud routing — judge + cloud members ride that path), ADR 0010 (capability/modality routing — members must be chat-routable models).

## Validation
- Right if agent callers consume the `Winner`/`Synthesis` output directly and rarely need the raw `include_answers` block — i.e. the judge's synthesis is trusted as the answer.
- Right if judge verdicts parse cleanly in practice; track the rate of `None` from `parse_judge_verdict` (the unparsed-reply fallback at `council.rs:142`). A high unparse rate means the JSON contract or the judge model needs revisiting.
- Revisit if a human-facing use case appears at request time (someone genuinely wanting to eyeball and pick) — that would reopen the UI question and conflict with ADR 0001/0002.
- Revisit the `local-only` judge inconsistency if anyone runs council under `local-only` expecting zero cloud calls and observes the judge still hitting cloud.
- Revisit if judge cost/latency dominates: if the added cloud judge call makes council uneconomical versus the outer agent self-judging cheaper members, the synthesis stage may need a local-judge option.

