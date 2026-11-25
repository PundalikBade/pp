"""Shor's algorithm: build the real circuit in Qiskit and run it locally or on BlueQubit.

Usage:
    python shor_bluequbit.py 15                      # local Qiskit Aer simulator
    python shor_bluequbit.py 15 --device cpu         # BlueQubit cloud CPU simulator
    python shor_bluequbit.py 15 --device gpu         # BlueQubit cloud GPU simulator
    python shor_bluequbit.py 15 --device quantum     # BlueQubit real quantum hardware (costs credits, noisy)

    python shor_bluequbit.py 15 --device ibm-fake    # IBM Fez noise model, local, no account
    python shor_bluequbit.py 15 --device ibm         # real IBM quantum hardware (free Open Plan)

IBM needs:  pip install qiskit-ibm-runtime   and   export IBM_QUANTUM_TOKEN=...   (quantum.cloud.ibm.com)
BlueQubit cloud devices need your token:  export BLUEQUBIT_API_TOKEN=...   (from app.bluequbit.io)

Qubits used = t counting + n work = 3n for N of n bits (t = 2n).
Statevector simulators handle roughly 30-36 qubits, so this exact circuit is for N up to about 10-11 bits.
"""

import argparse
import os
import math
import random
import threading
import time
from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate, QFTGate


# ---------- API keys ----------
# Keys can be set per thread (the web app passes each visitor's key this way) or via environment
# variables (command line). Set SHOR_PUBLIC=1 on a public server so the owner's env keys are never used.
KEYS = threading.local()


def _key(name: str, env: str) -> str:
    k = getattr(KEYS, name, None)
    if not k and os.environ.get("SHOR_PUBLIC") != "1":
        k = os.environ.get(env)
    if not k:
        raise RuntimeError(f"No {name.upper()} API key given.")
    return k


# ---------- classical helpers ----------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def perfect_power(n: int):
    for k in range(2, n.bit_length() + 1):
        r = round(n ** (1 / k))
        for c in (r - 1, r, r + 1):
            if c > 1 and c**k == n:
                return c, k
    return None


# ---------- quantum part ----------
def _flip(qc, ctrl, work, state, bit):
    """Flip work[bit] when ctrl=1 and the other work bits equal `state` (one multi-controlled X)."""
    others = [i for i in range(len(work)) if i != bit]
    zeros = [work[i] for i in others if not (state >> i) & 1]
    if zeros:
        qc.x(zeros)
    qc.append(MCXGate(len(others) + 1), [ctrl, *(work[i] for i in others), work[bit]])
    if zeros:
        qc.x(zeros)


def _swap_states(qc, ctrl, work, u, v):
    """Swap basis states |u> and |v> of the work register (Gray-code path of single-bit flips)."""
    path, cur = [u], u
    for b in range(len(work)):
        if ((u ^ v) >> b) & 1:
            cur ^= 1 << b
            path.append(cur)
    steps = list(zip(path, path[1:]))
    for g, g2 in steps + steps[-2::-1]:
        _flip(qc, ctrl, work, g, (g ^ g2).bit_length() - 1)


def controlled_modmul(m: int, N: int, n: int) -> QuantumCircuit:
    """Controlled |y> -> |m*y mod N> (identity for y >= N), built from multi-controlled X gates.

    Much smaller than a generic unitary: fits BlueQubit's 10 MB upload limit for larger N.
    Qubit 0 is the control, qubits 1..n are the work register.
    """
    qc = QuantumCircuit(n + 1, name=f"c*{m} mod {N}")
    work = list(range(1, n + 1))
    perm = [(m * y) % N if y < N else y for y in range(2**n)]
    seen = set()
    for s in range(2**n):
        if s in seen or perm[s] == s:
            seen.add(s)
            continue
        cyc, x = [s], perm[s]
        seen.add(s)
        while x != s:
            cyc.append(x)
            seen.add(x)
            x = perm[x]
        for c in cyc[1:]:
            _swap_states(qc, 0, work, cyc[0], c)
    return qc


def shor_circuit(N: int, a: int, t: int | None = None) -> QuantumCircuit:
    n = N.bit_length()
    t = t or 2 * n
    qc = QuantumCircuit(t + n, t, name=f"shor_{N}_{a}")
    qc.h(range(t))
    qc.x(t)  # work register starts in |1>
    for j in range(t):
        m = pow(a, 2**j, N)
        if m == 1:  # identity: skip, saves lots of gates
            continue
        qc.append(controlled_modmul(m, N, n).to_instruction(), [j, *range(t, t + n)])
    qc.append(QFTGate(t).inverse(), range(t))
    qc.measure(range(t), range(t))
    return qc


def run_circuit(qc: QuantumCircuit, device: str, shots: int) -> dict[str, int]:
    # unroll our modmul blocks and the inverse QFT, but keep MCX gates compact
    flat = qc.decompose(["c*", "qft_dg"]).decompose(["qft_dg", "QFTGate*", "IQFT*"])
    if device == "local":
        from qiskit_aer import AerSimulator

        sim = AerSimulator()
        tqc = transpile(flat, sim, optimization_level=1)
        print(f"  circuit: {qc.num_qubits} qubits, {tqc.size()} gates after transpile")
        return sim.run(tqc, shots=shots).result().get_counts()

    if device in ("ibm", "ibm-fake"):
        return run_on_ibm(flat, device, shots)

    import bluequbit

    print(f"  circuit: {qc.num_qubits} qubits, {flat.size()} gates sent (MCX form)")
    bq = bluequbit.init(_key("bluequbit", "BLUEQUBIT_API_TOKEN"))
    job = bq.run(flat, device=device, shots=shots, job_name=qc.name)
    return {k: int(v) for k, v in job.get_counts().items()}


def run_on_ibm(qc: QuantumCircuit, device: str, shots: int) -> dict[str, int]:
    """IBM Quantum hardware (free Open Plan) or a local copy of IBM's noise model."""
    from qiskit_ibm_runtime import SamplerV2

    if device == "ibm-fake":  # real IBM Fez calibration data, simulated locally, no account needed
        from qiskit_ibm_runtime.fake_provider import FakeFez

        backend = FakeFez()
    else:
        backend = _ibm_service().least_busy(operational=True, simulator=False)
    tqc = transpile(qc, backend=backend, optimization_level=3)
    ops = tqc.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
    print(f"  backend {backend.name}: {tqc.size()} gates, {two_q} two-qubit gates, depth {tqc.depth()}")
    job = SamplerV2(mode=backend).run([tqc], shots=shots)
    if device == "ibm":
        print(f"  IBM job id: {job.job_id()} (waiting in queue...)")
    return job.result()[0].data.c.get_counts()


def _minimal_period(r: int, a: int, N: int) -> int:
    """Shrink a valid multiple of the period to the period itself (classical post-processing)."""
    primes, m, p = set(), r, 2
    while p * p <= m:
        while m % p == 0:
            primes.add(p)
            m //= p
        p += 1
    if m > 1:
        primes.add(m)
    for p in primes:
        while r % p == 0 and pow(a, r // p, N) == 1:
            r //= p
    return r


def order_from_counts(counts: dict[str, int], a: int, N: int, t: int):
    Q = 2**t
    for bits, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        y = int(bits.replace(" ", ""), 2)
        if y == 0:
            continue
        frac = Fraction(y, Q).limit_denominator(N)
        for k in range(1, 6):  # small multiples fix a common k/r cancellation
            r = frac.denominator * k
            if r < N and pow(a, r, N) == 1:
                return _minimal_period(r, a, N), y, frac
    return None, None, None


def _ibm_service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=_key("ibm", "IBM_QUANTUM_TOKEN"))
    except Exception as e:
        msg = str(e)
        if "No matching instances" in msg:
            raise RuntimeError("Your IBM account has no instance yet. On quantum.cloud.ibm.com press "
                               "'Create an instance' and pick the free Open Plan, then try again.") from None
        if "401" in msg or "Unauthorized" in msg or "valid API token" in msg or "invalid" in msg.lower():
            raise RuntimeError("IBM did not accept this API key. Create a new key on quantum.cloud.ibm.com "
                               "and paste it again.") from None
        raise


def ibm_usage() -> dict:
    """Seconds used / left on your IBM instance this month. Uses no QPU time."""
    u = _ibm_service().usage()
    return {k: u.get(k) for k in ("usage_consumed_seconds", "usage_limit_seconds",
                                  "usage_allocation_seconds", "usage_remaining_seconds", "usage_period")}


def good_base(N: int):
    """Base a with an even period that splits N, picking the one whose circuit has the fewest
    multi-controlled gates (less noise on hardware). Checked classically, as hardware demos do,
    so each N needs only one hardware run."""
    best = None
    for a in range(2, N):
        if math.gcd(a, N) != 1:
            continue
        r, x = 1, a % N
        while x != 1:
            x, r = x * a % N, r + 1
        if r % 2 or pow(a, r // 2, N) == N - 1:
            continue
        n = N.bit_length()
        cost = sum(controlled_modmul(pow(a, 2**j, N), N, n).size()
                   for j in range(2 * n) if pow(a, 2**j, N) != 1)
        if best is None or cost < best[2]:
            best = (a, r, cost)
    return (best[0], best[1]) if best else (None, None)


def _order(a: int, N: int) -> int:
    r, x = 1, a % N
    while x != 1:
        x, r = x * a % N, r + 1
    return r


def _peak_share(counts: dict[str, int], N: int, a: int, t: int):
    """Share of shots on the correct peaks (y nearest to k * 2^t / r), and the same share for random outcomes.
    A perfect machine gives ~100%; pure noise gives only the peaks' share of all 2^t outcomes."""
    Q, r = 2**t, _order(a, N)
    peaks = set()
    for k in range(r):
        c = k * Q / r
        peaks.update({math.floor(c) % Q, math.ceil(c) % Q})
    total = sum(counts.values())
    got = sum(c for b, c in counts.items() if int(b, 2) in peaks) / total
    return got, len(peaks) / Q


def ibm_analyze(counts: dict[str, int], N: int, a: int, t: int) -> dict:
    """Turn measured counts into period, factors and the useful-vs-noise comparison."""
    counts = {k.replace(" ", ""): int(v) for k, v in counts.items()}
    r, y, frac = order_from_counts(counts, a, N, t)
    factors = None
    if r and r % 2 == 0 and pow(a, r // 2, N) != N - 1:
        p = math.gcd(pow(a, r // 2, N) - 1, N)
        if 1 < p < N:
            factors = sorted([p, N // p])
    got, noise = _peak_share(counts, N, a, t)
    total = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    return {"r_found": r, "y_used": y, "factors": factors, "peak_share": round(got, 4),
            "noise_share": round(noise, 4), "shots": total,
            "top": [[b, int(b, 2), c] for b, c in top]}


def ibm_submit(N: int, shots: int = 1000) -> dict:
    """Build, compile and submit one Shor circuit to the least busy IBM chip. Returns at once with the job id."""
    from qiskit_ibm_runtime import SamplerV2

    if N < 4 or N % 2 == 0 or is_prime(N) or perfect_power(N):
        raise ValueError(f"{N} is even, prime or a perfect power: no quantum step needed.")
    a, r = good_base(N)
    n = N.bit_length()
    backend = _ibm_service().least_busy(operational=True, simulator=False)
    print(f"Chip: {backend.name}. Building the circuit for N={N}, a={a}...")
    flat = shor_circuit(N, a).decompose(["c*", "qft_dg"]).decompose(["qft_dg", "QFTGate*", "IQFT*"])
    tqc = transpile(flat, backend=backend, optimization_level=3)
    two_q = sum(v for k, v in tqc.count_ops().items() if k in ("cz", "ecr", "cx"))
    print(f"Compiled: {3 * n} qubits, {two_q} two-qubit gates, depth {tqc.depth()}. Submitting...")
    job = SamplerV2(mode=backend).run([tqc], shots=shots)
    print(f"Submitted. IBM job id: {job.job_id()}")
    return {"job_id": job.job_id(), "N": N, "a": a, "t": 2 * n, "r_expected": r, "backend": backend.name,
            "qubits": 3 * n, "two_qubit_gates": two_q, "depth": tqc.depth(), "shots": shots}


def ibm_job_info(job_id: str, N: int) -> dict:
    """Rebuild the details of a job sent by ibm_submit, from its id and N (a is chosen the same way)."""
    a, r = good_base(N)
    job = _ibm_service().job(job_id)
    try:
        backend = job.backend().name
    except Exception:
        backend = "IBM"
    return {"job_id": job_id, "N": N, "a": a, "t": 2 * N.bit_length(), "r_expected": r, "backend": backend,
            "qubits": 3 * N.bit_length(), "two_qubit_gates": "—"}


def ibm_check(job_id: str, N: int, a: int, t: int) -> dict:
    """Status of an IBM job; once it is DONE, also the analysed result. Uses no QPU time."""
    job = _ibm_service().job(job_id)
    status = str(job.status())
    out = {"status": status}
    if status == "DONE":
        out.update(ibm_analyze(job.result()[0].data.c.get_counts(), N, a, t))
        try:
            out["quantum_seconds"] = job.usage()
        except Exception:
            out["quantum_seconds"] = None
    elif status in ("ERROR", "CANCELLED"):
        out["error"] = str(getattr(job, "error_message", lambda: "")() or status)
    return out


def ibm_batch(Ns: list[int], device: str = "ibm", shots: int = 1000) -> dict:
    """Run one Shor circuit per N in a single IBM job (one queue wait, least overhead)."""
    from qiskit_ibm_runtime import SamplerV2

    if device == "ibm-fake":
        from qiskit_ibm_runtime.fake_provider import FakeFez

        backend = FakeFez()
    else:
        backend = _ibm_service().least_busy(operational=True, simulator=False)
    print(f"backend: {backend.name}")
    rows, circuits = [], []
    for N in Ns:
        if N < 4 or N % 2 == 0 or is_prime(N) or perfect_power(N):
            print(f"N={N}: skipped (even, prime or a perfect power: no quantum step needed)")
            continue
        a, r = good_base(N)
        n = N.bit_length()
        qc = shor_circuit(N, a)
        flat = qc.decompose(["c*", "qft_dg"]).decompose(["qft_dg", "QFTGate*", "IQFT*"])
        tqc = transpile(flat, backend=backend, optimization_level=3)
        two_q = sum(v for k, v in tqc.count_ops().items() if k in ("cz", "ecr", "cx"))
        print(f"N={N}: a={a}, {3 * n} qubits, {two_q} two-qubit gates, depth {tqc.depth()}")
        rows.append({"N": N, "a": a, "r_expected": r, "t": 2 * n, "qubits": 3 * n,
                     "two_qubit_gates": two_q, "depth": tqc.depth()})
        circuits.append(tqc)
    if not circuits:
        return {"rows": [], "quantum_seconds": 0}
    job = SamplerV2(mode=backend).run(circuits, shots=shots)
    if device == "ibm":
        print(f"IBM job id: {job.job_id()} ({len(circuits)} circuits, waiting in queue...)")
    result = job.result()
    for row, pub in zip(rows, result):
        row.update(ibm_analyze(pub.data.c.get_counts(), row["N"], row["a"], row["t"]))
        print(f"N={row['N']}: r found={row['r_found']} (expected {row['r_expected']}), factors={row['factors']}, "
              f"shots on correct peaks {row['peak_share']:.1%} (perfect machine ~100%, random noise {row['noise_share']:.1%})")
    secs = None
    if device == "ibm":
        try:
            secs = job.usage()
        except Exception:
            secs = None
        print(f"QPU time used by this job: {secs} s")
    return {"rows": rows, "quantum_seconds": secs, "backend": backend.name}


def shor_split(N: int, device: str, shots: int, a: int | None, tries: int):
    if N % 2 == 0:
        return 2
    pp = perfect_power(N)
    if pp:
        return pp[0]
    n = N.bit_length()
    t = 2 * n
    for attempt in range(1, tries + 1):
        aa = a or random.randrange(2, N - 1)
        g = math.gcd(aa, N)
        print(f"\n[attempt {attempt}] a = {aa}")
        if g > 1:
            print(f"  lucky: gcd(a, N) = {g}")
            return g
        t0 = time.time()
        counts = run_circuit(shor_circuit(N, aa, t), device, shots)
        print(f"  ran in {time.time() - t0:.2f}s; top outcomes:",
              sorted(counts.items(), key=lambda kv: -kv[1])[:6])
        r, y, frac = order_from_counts(counts, aa, N, t)
        if r is None:
            print("  no period found from these shots, retrying")
            continue
        print(f"  y = {y}, y/2^{t} ~ {frac}  ->  period r = {r}")
        if r % 2:
            print("  r is odd, retrying with another a")
            continue
        x = pow(aa, r // 2, N)
        if x == N - 1:
            print("  a^(r/2) = -1 mod N, retrying")
            continue
        p = math.gcd(x - 1, N)
        if 1 < p < N:
            print(f"  gcd(a^(r/2)-1, N) = {p}")
            return p
        q = math.gcd(x + 1, N)
        if 1 < q < N:
            return q
    raise RuntimeError("Shor did not split N in the allowed tries")


def factorize(N: int, **kw) -> list[int]:
    if N == 1:
        return []
    if is_prime(N):
        return [N]
    d = shor_split(N, **kw)
    return sorted(factorize(d, **kw) + factorize(N // d, **kw))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("N", type=int)
    ap.add_argument("--device", default="local",
                    choices=["local", "cpu", "gpu", "mps.cpu", "mps.gpu", "quantum", "ibm", "ibm-fake"])
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--a", type=int, default=None, help="fix the base a (default: random)")
    ap.add_argument("--tries", type=int, default=8)
    args = ap.parse_args()
    n = args.N.bit_length()
    print(f"N = {args.N} ({n} bits) -> {3 * n} qubits for the full circuit")
    if 3 * n > 34:
        print("WARNING: above ~34 qubits a statevector simulator cannot hold this circuit.")
    f = factorize(args.N, device=args.device, shots=args.shots, a=args.a, tries=args.tries)
    print(f"\nRESULT: {args.N} = {' x '.join(map(str, f))}")
