# Brian's Programming Bible — v2

*v1 was a strong synthesis of proven canon (NASA Power-of-10, type-driven design, distributed fundamentals, a real testing taxonomy) with a sound priority hierarchy. v2 fixes what a 3-vendor review (MiMo V2.5 Pro + GPT-5.5 + Claude) found: it ported C rules that fight idiomatic Rust, overclaimed what memory-safety buys, used assertion/size quotas that contradict the type-invariant philosophy, applied "always/total" framing that fights Rule 0, and was silent on the async/concurrency/unsafe/ops hazards that actually bite Rust systems. Spirit unchanged. Rigor scoped.*

## CHANGELOG (what the review changed, and why)
- **R25 rewritten** — "no function pointers / one indirection" is a C rule; in Rust it forbids trait objects/closures/async (idiomatic + safe) and kills DI/mocking, contradicting the testing rules. Now: minimize *hot-path dynamic dispatch*, document indirection, one-deref-max applies *to C/FFI only*.
- **R21 rewritten** — "min 2 assertions/function" contradicts R8 (invariants in types) and produces `debug_assert!(true)` spam. Now: every invariant asserted *somewhere*, prefer types > tests > runtime asserts; assert what types can't encode, at boundaries.
- **R9/R13 corrected (the dangerous fix)** — memory-safety kills data races + use-after-free; it does NOT kill deadlocks, lock-ordering, logical/TOCTOU races across `.await`, async-cancellation corruption, livelock, or resource exhaustion. New §XII makes these first-class.
- **R12 reframed** — "total state coverage" is undecidable for async/distributed; the *target* is an inductive set covering each input class, not a gate.
- **R7 softened** — you can't enumerate kernel/hw/network/clock failure totality; classify failure domains + make every *reachable* state handled. No silent failures.
- **R0 reconciled with the checklist** — added reliability tiers + a list of rules that NEVER suspend even in a prototype.
- **R20 de-proxied** — "one page" → "one responsibility; extract when you'd comment a section."
- **R17 + R14 + R15 extended** — redaction in logging; threat-model in security; supply-chain in deps.
- **New rules 42–48** — async/cancellation, unsafe policy, error taxonomy, resource budgets, deployment/ops, perf methodology, data integrity.
- **New §XIII** — fallback behavior when verification is blocked.

---

## Hierarchy (tie-break)
1 Correctness over performance — *except where latency/throughput/realtime IS the contract (then perf is correctness; state it).*  2 Architecture over features  3 Explicit over implicit  4 Compile-time over runtime — *without type-system theater; push validation to compile time where it removes a real runtime failure, not to satisfy the rule.*  5 Simple over clever — until clever solves a real problem.

## Rule 0 — Tiered discipline, not "no rules."
Declare the **reliability tier** before writing: **PROTOTYPE** (validate the shape; it exists to die and inform — most rules relax), **SERVICE** (default — correctness, error handling, observability, tests on load-bearing paths), **CRITICAL** (safety/security/data/money/realtime — full rigor, adversarial tests, fail-closed). NASA discipline always welcome; it just costs time. **Never suspended at any tier:** data integrity, security boundaries, irreversible storage/wire formats, benchmark methodology, and "no silent failure."

## I. Architecture
1. Do one thing well; compose small pieces.
2. Architecture before features under any constraint — but validate architecture by shipping, deletion, and change-cost, not by elegance. ("Architecture cosplay" is the failure mode.)
3. Design the longest-lived *domain* component first (the data/wire format, the core invariant). Use boring defaults for infra.
4. Version anything serialized or public from day one (formats, APIs, schemas, registries, wire). Not every internal interface.
5. Principles govern design; data governs optimization. Design → measure → optimize. State the perf budget (see R47).

## II. Correctness
6. Strict types at every boundary. Types prove SHAPE; invariants prove MEANING. Encode meaning in types where it pays.
7. No *unhandled* failure. Classify failure domains; make every reachable state explicit. You cannot enumerate the totality of hardware/kernel/network failure — you can guarantee none is silent and each reachable one has a path.
8. Type invariants are a passive regression harness — old tests catch new bugs.
9. Define data ownership + lifecycle explicitly. Rust enforces *memory* ownership mechanically — NOT logical lifecycle, task lifetime, or protocol state (see §XII).

## III. Testing
10. Unit tests on the load-bearing paths; E2E for the contract. Avoid brittle tests coupled to implementation detail — for systems work, property/fuzz/golden-corpus/integration often beat high unit density.
11. Types of tests > number of tests.
12. Aim for an inductive set covering every input *class* (total state coverage is the asymptote, not the gate). Add invariants, generators, fuzz targets, model checks, production monitors.

## IV. Safety & Security
13. Use a memory-safe language to delete the memory-bug class — knowing it deletes ONLY that class (see §XII).
14. Security is first-class: least surface, least value, least privilege — *plus a threat model per boundary*: path traversal, SSRF, injection, auth bypass, secrets-in-logs/argv, untrusted deserialization, malicious files, sandboxing. Sometimes the correct posture is fail-CLOSED (see R27).

## V. Maintainability
15. Minimize deps, maximize modularity — and treat dependencies as **supply-chain risk**: audit (cargo-audit/deny), pin, MSRV policy, license check, prefer not pulling a large transitive tree for a tiny helper, reproducible builds.
16. Comments mark the extraordinary (non-obvious reasoning/hazards/tradeoffs), not the obvious.

## VI. Observability
17. Every meaningful state transition + failure path is loggable — **with redaction** (never log secrets, keys, tokens, PHI/EEG, raw payloads) and cardinality control. Beyond logs: metrics (latency histograms, queue depth), traces/spans, request IDs, SLOs, health checks.

## VII. Control Flow & Low-Level
18. Simplify control flow. Bound the things that grow — loops over data, **queues, retries, waits, resource use** — not the server's intentional event loop. No goto.
19. Bound or eliminate recursion (provable bound or none).
20. One responsibility per function; extract a block the moment you'd write a comment to explain it. (Not a line/page count.)
21. Assert every invariant **somewhere** — prefer types > tests > runtime asserts. Put runtime asserts on what the type system can't encode, at trust boundaries. No assertion quotas; `debug_assert!(true)` is noise.
22. Declare data at the narrowest scope.
23. Validate at **trust boundaries** (external input, IO, deserialization, FFI). Inside a typed module, lean on types/constructors/privacy — blanket re-validation is defensive noise.
24. Macros rare + clean + complete; distinguish declarative vs proc vs build-script; minimize conditional compilation.
25. **[Rust]** Minimize *unnecessary dynamic dispatch* on hot/codec paths (prefer monomorphization); document every `dyn`/layer of indirection that obscures ownership or perf. Trait objects, closures, fn-pointers, async are idiomatic and allowed. **[C only]** one level of indirection max; no function-pointer tables that defeat static analysis.
26. A compile error beats a runtime error: compile pedantic. **[Rust]** clippy at deny, `deny(unsafe_op_in_unsafe_fn)`, no `unwrap`/`expect` on library paths, miri/sanitizers on `unsafe`, cargo-deny/audit, MSRV + feature-matrix in CI. Distinguish invariant-violations (→ types/compile) from expected operational failures (→ `Result`) — don't panic on edge cases you can foresee.

## VIII. Resilience
27. Backup/degrade beats failure — **except** where wrong output is worse than no output (corruption, auth, money, split-brain, data leak): there, fail CLOSED. Pick per case; never fail silently.
28. Always have something to ship — labeled with its reliability tier (don't ship a prototype as a service).
29. In-place refinement: if memory holds a valid approximation and the refinement can't finish in time, use the existing value; commit atomically (CAS/generation-counter/lease/snapshot — name the mechanism); never leave state half-written.
30. Design interfaces as if the caller is hostile, angry, and stupid. Make misuse impossible before correct use convenient.

## IX. Distributed Systems
31. Idempotency + determinism by default; deviation needs justification.
32. Make the convergence model **explicit** — and honest about CAP: under partition you choose (quorum write / single-writer / CRDT / lease / eventual). "Bounded + guaranteed" only where the model actually provides it; don't claim convergence a partition can break.
33. Backpressure is first-class: every queue/pipeline/async interface answers "what happens when full" + carries an end-to-end **deadline** (backpressure without deadlines just slows everything instead of failing fast).
34. Use consensus only when you actually need agreement — often the right answer is *avoid* consensus (centralize ownership, durable queue, accept eventual). When you need it: quorum, explicit.
35. Failure detection is first-class: nodes are suspected, not known-dead. Define detection, isolation, recovery as normal operation.

## X. Scope & Structure
36. Define the minimum shippable surface before coding.
37. Price scope changes honestly: fits-arch 1×, structural 10×, architectural 100×.
38. Know the blast radius before any structural decision (what must change if this changes).
39. Good structural decisions are reversible/replaceable, make the next decision easier, reduce hiding spots for bugs, increase observability.
40. Boring architecture by default; novelty only with the testing arsenal to absorb its blast radius.
41. Novel solutions are cherries on a robust, tested, observable system. Punish ambition without a safety net — never ambition itself.

## XI. Paradigm — TDD + measurement-driven optimization
Contract-first → test-before-code → correct-before-fast → optimize-with-measurement → verify-the-contract-held → architecture-before-scaling. Reliability > architecture/scalability > features. Speed is a *consequence* of architecture that wastes nothing, not a target. State your reliability/speed dial position and don't drift.

## XII. Concurrency, Async & `unsafe` (NEW — the biggest v1 gap)
42. **Concurrency hazards memory-safety does NOT solve:** deadlock, lock-ordering, logical/TOCTOU races across `.await`, async-cancellation corruption, livelock, starvation, priority inversion, resource exhaustion. Rules: define + document a global lock ordering; **never hold a sync lock guard across an `.await`**; treat cancellation/timeout as a real code path (a future dropped mid-transaction must not corrupt state); bound spawned tasks; use structured concurrency (scoped tasks, propagate shutdown); pick channel types deliberately (mpsc/broadcast/watch/oneshot have different backpressure + cancellation semantics); mind `Send`/`Sync` at spawn boundaries.
43. **`unsafe` policy:** allowed only where required (SIMD, zero-copy, FFI); every block carries a safety comment stating the invariant it upholds and a test that fails if violated; run miri/sanitizers; isolate + minimize; FFI boundaries validate both ways.

## (extends earlier sections)
44. **Error taxonomy:** distinguish programmer-bug (panic/`unreachable!`) vs invalid-input (typed error to caller) vs transient (retry w/ backoff+jitter) vs permanent vs corruption vs cancellation vs overload (shed/backpressure) vs unauthorized vs degraded-mode. `thiserror` for library errors, `anyhow` for application edges — typed at the boundary.
45. **Resource budgets:** explicit ceilings for memory, fds, sockets, GPU/VRAM, model cache, queue depth, request size, timeouts, retries, per-tenant limits. A server without budgets is a time-delayed OOM.
46. **Deployment/ops:** config validation at startup (fail to start, not mid-flight), kill switches/feature flags, rollback path, migration + mixed-version compatibility, incident runbook. Decisions that outlive a deploy get an ADR.
47. **Performance methodology:** fixed corpus, warmup, report p50/p95/p99 + memory + throughput (+ compression ratio/quality for codecs), regression threshold in CI. Never optimize by guess; never benchmark without methodology.
48. **Data integrity (codec/EEG):** checksums + corruption detection, deterministic decode, a golden corpus, metadata provenance, explicit endianness + units, schema validation + versioned migration.

## Enforcement — tiered (replaces the flat checklist)
**Always (every tier):** no silent failures · types validated at boundaries · secrets never logged · `unsafe` justified + tested · serialized/wire formats versioned · deps audited · scope defined before coding · structural blast-radius understood · convergence/failure model explicit for distributed code.
**SERVICE adds:** invariants in types where they pay · error taxonomy applied · resource budgets set · backpressure + deadlines · observability on failure paths · load-bearing paths tested · config validated at startup.
**CRITICAL adds:** adversarial + property/fuzz tests · inductive state coverage · fail-closed where wrong>none · perf budget + regression gate · data-integrity guarantees · independent (ideally cross-model adversarial) review · golden corpus + mutation testing.

## XIII. When verification is blocked (NEW)
If you cannot verify (sandbox blocks it, build unavailable, suite too large, read-only request): do NOT fabricate or assume success. Degrade explicitly — run the narrowest check you can, state exactly what was and wasn't verified, list the unverified claims, and surface the block. An honest "unverified: couldn't run X" beats a confident false "done."

---
*48 rules. The prototype finds the shape. The rules — scoped to the stakes — build the thing. The foundation earns the reach. v2: same spirit, honest about what the tools actually guarantee.*
