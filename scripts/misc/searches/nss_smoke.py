"""
nss_smoke.py — blackjax >=1.6 environment validation (Phase 0(b), issue #142).

Two CPU-scale checks, each of which must PASS after any environment's
blackjax upgrade (local venv, RAL stack) before Phase 2 NSS work runs there:

1. ``blackjax.nss`` on a 2D toy (Gaussian likelihood x uniform prior) with an
   analytic evidence check, exercising the exact mainline API surface the
   Phase 2 profiling-local runner will wire: native-space ``logprior_fn``,
   ``num_delete``, dlogz termination on ``state.integrator.logZ_live - logZ``,
   and the ``ns.utils.finalise`` + ``log_weights`` simulated-volume logZ
   ensemble.
2. ``af.BlackJAXNUTS`` on a 2D toy posterior, confirming the PyAutoFit wrapper
   (window_adaptation + nuts + blackjax.diagnostics) against the installed
   blackjax.

Run: ``python scripts/misc/searches/nss_smoke.py``. Exits non-zero on FAIL.
Wall times printed here are CPU toy scale and carry no profiling meaning
(wrong tier — see results/notes/inference/PROGRAMME.md section 3).
"""

import sys
import time


def smoke_nss() -> bool:
    """Mainline blackjax.nss on a 2D toy with an analytic logZ check."""
    import blackjax
    import blackjax.ns.utils as nsu
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    def logprior(x):
        inside = jnp.all(jnp.abs(x) <= 5.0)
        return jnp.where(inside, -jnp.log(10.0**2), -jnp.inf)

    def loglik(x):
        mu = jnp.array([1.0, -1.0])
        sig = 0.5
        return -0.5 * jnp.sum(((x - mu) / sig) ** 2) - jnp.log(2 * jnp.pi * sig**2)

    # Z = (1/100) * (Gaussian mass, ~fully inside the box) -> logZ = -log(100).
    analytic_logz = float(-jnp.log(100.0))

    # d=2 -> num_inner_steps = max(5, 2d) = 5 per upstream guidance.
    algo = blackjax.nss(logprior, loglik, num_inner_steps=5, num_delete=10)
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    particles = jax.random.uniform(init_key, (500, 2), minval=-5.0, maxval=5.0)
    state = algo.init(particles, rng_key=init_key)

    step = jax.jit(algo.step)
    dead = []
    t0 = time.time()
    n = 0
    while float(state.integrator.logZ_live - state.integrator.logZ) > -3.0:
        key, k = jax.random.split(key)
        state, info = step(k, state)
        dead.append(info)
        n += 1
        if n > 5000:
            print("nss: FAIL (no dlogz convergence in 5000 steps)")
            return False
    wall = time.time() - t0

    final = nsu.finalise(state, dead)
    key, k = jax.random.split(key)
    logw = nsu.log_weights(k, final, shape=100)  # (n_particles, 100)
    logz_ens = logsumexp(logw, axis=0)
    logz_mean = float(jnp.mean(logz_ens))
    logz_std = float(jnp.std(logz_ens))

    key, k = jax.random.split(key)
    post = nsu.sample(k, final, shape=2000)
    pos = post.position if hasattr(post, "position") else post
    mean = jnp.mean(pos, axis=0)

    print(
        f"nss: steps={n} wall={wall:.1f}s (CPU toy) "
        f"logZ={logz_mean:.3f}+/-{logz_std:.3f} analytic={analytic_logz:.3f} "
        f"posterior mean={mean}"
    )
    ok = abs(logz_mean - analytic_logz) < 3 * max(logz_std, 0.05)
    ok = ok and bool(jnp.all(jnp.abs(mean - jnp.array([1.0, -1.0])) < 0.1))
    print(f"nss: {'PASS' if ok else 'FAIL'}")
    return ok


def smoke_blackjax_nuts() -> bool:
    """af.BlackJAXNUTS wrapper against the installed blackjax."""
    import autofit as af
    import jax.numpy as jnp  # noqa: F401 — asserts the jax path is importable

    class Point:
        def __init__(self, a=0.0, b=0.0):
            self.a = a
            self.b = b

    class ToyAnalysis(af.Analysis):
        def __init__(self):
            super().__init__(use_jax=True)

        def log_likelihood_function(self, instance):
            return -((instance.a - 1.0) ** 2 + (instance.b + 1.0) ** 2) / (2 * 0.5**2)

    model = af.Model(Point)
    model.a = af.GaussianPrior(mean=0.0, sigma=2.0)
    model.b = af.GaussianPrior(mean=0.0, sigma=2.0)

    search = af.BlackJAXNUTS(num_warmup=200, num_samples=300, seed=1)
    result = search.fit(model=model, analysis=ToyAnalysis())
    mp = result.samples.median_pdf()
    info = result.samples.samples_info
    print(
        f"BlackJAXNUTS: median=({mp.a:.3f}, {mp.b:.3f}) truth=(1, -1) "
        f"divergent={info.get('n_divergent')} "
        f"acc={info.get('mean_acceptance'):.3f} ess_min={info.get('ess_min'):.1f}"
    )
    ok = abs(mp.a - 1.0) < 0.2 and abs(mp.b + 1.0) < 0.2
    ok = ok and info.get("n_divergent") == 0
    print(f"BlackJAXNUTS: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    import blackjax
    import jax

    jax.config.update("jax_enable_x64", True)
    print(f"blackjax={blackjax.__version__} jax={jax.__version__}")
    major, minor, *_ = (int(x) for x in blackjax.__version__.split(".")[:2])
    if (major, minor) < (1, 6):
        print(f"FAIL: blackjax {blackjax.__version__} < 1.6 — upgrade first")
        return 1
    ok = smoke_nss()
    ok = smoke_blackjax_nuts() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
