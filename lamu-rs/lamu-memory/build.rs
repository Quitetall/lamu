//! Link-glue for the opt-in `turbovec` feature (ADR 0031).
//!
//! turbovec pulls BLAS through `ndarray{blas}` + `openblas-src{system}`.
//! `openblas-src`'s build script emits the `-lopenblas` directive, but in
//! this workspace that directive demonstrably reaches lib/bin link lines
//! and NOT the test harness binaries — `cargo test --features turbovec`
//! died at link time with `undefined symbol: cblas_sgemm` while
//! `cargo build --features turbovec` stayed green. Re-emitting the system
//! link directive from THIS crate's build script attaches it to
//! lamu-memory's own metadata, which cargo propagates to every downstream
//! target that links the crate — including test binaries.
//!
//! No-op for default builds: the directive is gated on the `turbovec`
//! feature, so a feature-off build links nothing extra.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_TURBOVEC").is_some() {
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
