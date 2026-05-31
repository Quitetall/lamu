# ADR 0005: Bind 127.0.0.1 by default

## Status
Accepted 2026-05-31

## Context
LAMU's `lamu serve` exposes an OpenAI/Ollama-compatible HTTP surface (`lamu-api/src/lib.rs:11` `serve`, app built at line 16). That surface has **no auth layer** — no API key check, no token gate, no allowlist on the request handlers. Any client that can reach the socket can drive inference, load/unload models, and consume VRAM.

Until commit `7c404a3` (`fix(serve)!: bind 127.0.0.1 by default, not 0.0.0.0`), `serve` hardcoded the listen address to `0.0.0.0:<port>`. That meant from the very first launch the unauthenticated API was reachable from every host on the LAN, with no opt-in. For a backend being prepared for release this is a default-insecure footgun: the operator gets remote exposure without ever asking for it, and without any credential standing between the network and the model. The commit subject flags it as a ship-blocker. The constraint driving the fix: LAMU is positioned as a *local* backend (default bind is one concern; the broader lean-backend posture is ADR 0002), and a local backend that silently listens on all interfaces violates the principle of least surprise for its own threat model.

## Decision
`lamu serve` binds the loopback interface (`127.0.0.1`) by default. The listen host is read from the `LAMU_BIND_HOST` environment variable (`lamu-api/src/lib.rs:22`), defaulting to `127.0.0.1` when unset; an invalid value produces a clear startup error rather than a silent fallback (line 23-25). Off-host exposure is explicitly opt-in: the operator sets `LAMU_BIND_HOST=0.0.0.0` (or a specific interface IP). When the resolved address is non-loopback, `serve` emits a loud `warn!` stating that the API is unauthenticated and reachable off-host, and that it should be put behind a reverse proxy or firewall (lines 27-33). No authentication is added to the HTTP surface; security-by-default is achieved by not listening off-host, not by gating requests.

## Rationale
- The HTTP surface is unauthenticated — there is no credential check anywhere on the request path, so network reachability *is* the entire access-control boundary. Confining that boundary to loopback is the only control available without adding auth.
- Secure-by-default beats convenient-by-default for a tool that loads arbitrary models and burns GPU memory on request: the failure mode of the old `0.0.0.0` default (LAN-wide unauthenticated inference) is worse than the failure mode of the new default (a remote user has to flip one env var).
- The opt-in is a single, discoverable env var that reuses the same `LAMU_BIND_HOST` name the backends already use, so there is one mental model for "where does this listen," not two.
- The non-loopback `warn!` (lines 28-32) keeps the opt-in honest: an operator who deliberately exposes the port still gets told, on every launch, that nothing is authenticating the callers.
- Invalid `LAMU_BIND_HOST` fails fast with a message naming valid examples (line 24), so a typo can't silently degrade to a wrong or insecure bind.

## Alternatives Considered
- **Keep `0.0.0.0` as the default** — what `7c404a3` reversed. Zero-config remote access for multi-machine setups, but it exposes an unauthenticated inference + model-management endpoint to every host on the network from first launch, with no consent step. Rejected as a ship-blocker: the convenience saves one env var; the cost is LAN-wide unauthenticated control of a GPU. The asymmetry is decisive.
- **Add an auth/token layer to the HTTP surface** — would let `0.0.0.0` stay safe-ish by gating requests on a bearer token. Rejected as scope creep for a local backend: it pulls in token issuance, storage, rotation, per-route enforcement, and a config surface — a meaningful subsystem to maintain (and to get wrong) — when the actual requirement is "don't be reachable off-host by accident." Binding loopback solves the real problem with one line. Auth remains the documented path *only* for operators who choose `0.0.0.0`, and we explicitly defer it to a reverse proxy (ADR 0001 keeps HTTP a thin compat shim, not a place to grow an auth stack).
- **Bind a specific LAN IP by default via interface autodetection** — picking the "primary" interface automatically. Rejected: it's still off-host exposure without consent, just less predictable, and autodetection is fragile across multi-homed hosts and VPNs.

## Consequences
- This is a **breaking change** for any setup that relied on implicit remote reachability of `lamu serve`. Those operators must now set `LAMU_BIND_HOST=0.0.0.0` to restore the old behavior. This must be a release-note headline (it is flagged BREAKING in `7c404a3`). Future readers should know the `0.0.0.0` default was intentional-then-reversed, not an oversight.
- We are committed to *not* shipping auth on the HTTP path; the security story is "loopback by default, your responsibility past that." If LAMU ever needs authenticated multi-host access, that's a new decision, not a tweak.
- The non-loopback path stays load-bearing: the `warn!` and the opt-in env var are the contract for anyone exposing the port. Removing the warning would silently weaken the only signal a remote-exposed operator gets.
- Honest downside: `bind_reuseaddr` constructs an `IPV4` socket unconditionally (`lamu-api/src/lib.rs:67`, `Socket::new(Domain::IPV4, ...)`). An operator who sets `LAMU_BIND_HOST` to an IPv6 literal (e.g. `::1`) will parse a valid `SocketAddr` at line 23 but fail at `sock.bind` because the domain mismatches. The default `127.0.0.1` and the common `0.0.0.0` opt-in both work; IPv6 binds do not. This constrains a future "bind IPv6 loopback" request to first generalize the socket-domain selection.
- The `SO_REUSEADDR` choice (lines 63-68) is orthogonal to the bind host but lives in the same function; it exists so a fast restart after SIGTERM doesn't trip on TIME_WAIT. It does not weaken the loopback default.

## Related Decisions
ADR 0001 (HTTP serve is a thin compat shim — keeps auth out of this layer), ADR 0002 (lean Rust backend — no batteries-included auth framework), ADR 0006 (HTTP path never auto-evicts; the HTTP surface is deliberately minimal), ADR 0004 (managed-subprocess backends share the same `LAMU_BIND_HOST` env).

## Validation
- Right if: fresh installs are not reachable off-host without an explicit `LAMU_BIND_HOST` change, and operators who do expose the port report seeing the unauthenticated warning. No incident reports of "I didn't know it was on the network."
- Wrong / revisit if: the loopback default proves to block a legitimate, common workflow so often that operators are documenting `LAMU_BIND_HOST=0.0.0.0` as a required step (signal that remote-by-design is the real use case and auth should be built rather than deferred); or if IPv6 loopback demand surfaces, forcing the `Domain::IPV4` hardcode at `lib.rs:67` to be generalized; or if a real auth requirement appears for multi-host deployments, which would supersede the "no auth, just loopback" posture.

