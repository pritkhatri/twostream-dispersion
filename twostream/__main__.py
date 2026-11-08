"""Command line entry point: python -m twostream --drift 3 --vth 1 --out growth.png"""
import argparse

import numpy as np

from .dispersion import peak, scan, symmetric_two_stream


def main():
    p = argparse.ArgumentParser(
        description="Cold vs warm-kinetic growth rate of two counter-streaming "
                    "electron beams (equal density, equal temperature).")
    p.add_argument("--drift", type=float, default=3.0,
                   help="beam drift speed in units of v_ref (default 3)")
    p.add_argument("--vth", type=float, default=1.0,
                   help="beam thermal speed in units of v_ref (default 1)")
    p.add_argument("--kmax", type=float, default=0.6,
                   help="largest wavenumber in units of w_pe/v_ref (default 0.6)")
    p.add_argument("--nk", type=int, default=300, help="number of K points")
    p.add_argument("--out", default="docs/growth_rate.png", help="figure path")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = symmetric_two_stream(args.drift, args.vth)
    K = np.linspace(args.kmax / args.nk, args.kmax, args.nk)
    _, g_cold = scan(sp, K, "cold")
    _, g_kin = scan(sp, K, "kinetic")
    gc, Kc = peak(K, g_cold)
    gk, Kk = peak(K, g_kin)

    print(f"drift = {args.drift}, vth = {args.vth}")
    print(f"cold          gamma_max = {gc:.4f} w_pe at K = {Kc:.3f}")
    print(f"warm kinetic  gamma_max = {gk:.4f} w_pe at K = {Kk:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K, g_cold, label="cold fluid")
    ax.plot(K, np.clip(g_kin, 0, None), label="warm kinetic")
    ax.set_xlabel(r"$k\,v_{ref}/\omega_{pe}$")
    ax.set_ylabel(r"growth rate $\gamma/\omega_{pe}$")
    ax.set_title(
        f"Two-stream growth rate, drift ±{args.drift:g} $v_{{ref}}$, "
        f"beam $v_{{th}}$ = {args.vth:g} $v_{{ref}}$"
    )
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure written to {args.out}")


if __name__ == "__main__":
    main()
