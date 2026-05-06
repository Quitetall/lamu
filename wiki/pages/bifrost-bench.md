# Bifrost gateway overhead

Empirical measurements of Bifrost (`:8080`) as a proxy in front of the
local model on `:8020`. Run `bash scripts/bench-bifrost.sh` to append a
new entry to this file.

The v3 path-consolidation plan keeps Bifrost iff overhead is under 3%
of total request latency (LAN-local proxy on the same host). Above 3%,
Bifrost gets stripped from the runtime path.
