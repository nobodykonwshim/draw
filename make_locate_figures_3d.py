import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


XLIM: Tuple[float, float] = (0, 100)
YLIM: Tuple[float, float] = (0, 100)
ZLIM: Tuple[float, float] = (0, 3000)


def _setup_3d_ax(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_zlim(*ZLIM)
    ax.invert_zaxis()
    ax.view_init(elev=22, azim=-60)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (m)")
    ax.set_facecolor("white")
    return ax


def _save_fig(fig: plt.Figure, outpath: str) -> None:
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig_4_1_constraints_3d(outpath: str) -> None:
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = _setup_3d_ax(fig)

    z_min, z_max = 200, 2000

    x_edges = np.array([XLIM[0], XLIM[1]])
    y_edges = np.array([YLIM[0], YLIM[1]])

    for x in x_edges:
        for y in y_edges:
            ax.plot([x, x], [y, y], [z_min, z_max], color="tab:blue", alpha=0.6)

    for z in [z_min, z_max]:
        ax.plot([XLIM[0], XLIM[1]], [YLIM[0], YLIM[0]], [z, z], color="tab:blue", alpha=0.6)
        ax.plot([XLIM[0], XLIM[1]], [YLIM[1], YLIM[1]], [z, z], color="tab:blue", alpha=0.6)
        ax.plot([XLIM[0], XLIM[0]], [YLIM[0], YLIM[1]], [z, z], color="tab:blue", alpha=0.6)
        ax.plot([XLIM[1], XLIM[1]], [YLIM[0], YLIM[1]], [z, z], color="tab:blue", alpha=0.6)

    seabed_x = np.linspace(*XLIM, 60)
    seabed_y = np.linspace(*YLIM, 60)
    seabed_xx, seabed_yy = np.meshgrid(seabed_x, seabed_y)
    seabed_zz = 1800 + 6 * seabed_xx + 5 * seabed_yy + 150 * np.sin(seabed_xx / 12) * np.cos(seabed_yy / 15)
    seabed_zz = np.clip(seabed_zz, 1200, 2800)

    ax.plot_surface(seabed_xx, seabed_yy, seabed_zz, cmap="terrain", alpha=0.65, linewidth=0, antialiased=True)

    ax.plot([], [], [], color="tab:blue", label="Feasible volume")
    ax.plot([], [], [], color="saddlebrown", label="Seabed surface")
    ax.plot([], [], [], color="black", linestyle="--", label="Depth limits")
    ax.plot([XLIM[0], XLIM[1]], [YLIM[0], YLIM[0]], [z_min, z_min], color="black", linestyle="--", alpha=0.6)
    ax.plot([XLIM[0], XLIM[1]], [YLIM[1], YLIM[1]], [z_max, z_max], color="black", linestyle="--", alpha=0.6)

    ax.legend(loc="upper left")

    _save_fig(fig, outpath)


def make_fig_4_2_currents_3d(outpath: str) -> None:
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = _setup_3d_ax(fig)

    x = np.linspace(*XLIM, 10)
    y = np.linspace(*YLIM, 10)
    z = np.linspace(300, 2500, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    u = 0.8 * np.sin(2 * np.pi * yy / YLIM[1]) * np.exp(-zz / 3000)
    v = 0.8 * np.cos(2 * np.pi * xx / XLIM[1]) * np.exp(-zz / 3000)
    w = 0.2 * np.sin(np.pi * zz / ZLIM[1])

    up_mask = w >= 0
    down_mask = ~up_mask

    ax.quiver(
        xx[up_mask],
        yy[up_mask],
        zz[up_mask],
        u[up_mask],
        v[up_mask],
        w[up_mask],
        length=8,
        normalize=True,
        color="tab:cyan",
        alpha=0.8,
    )
    ax.quiver(
        xx[down_mask],
        yy[down_mask],
        zz[down_mask],
        u[down_mask],
        v[down_mask],
        w[down_mask],
        length=8,
        normalize=True,
        color="tab:purple",
        alpha=0.8,
    )

    ax.plot([], [], [], color="tab:cyan", label="w > 0")
    ax.plot([], [], [], color="tab:purple", label="w < 0")
    ax.legend(loc="upper left")

    _save_fig(fig, outpath)


def _draw_box(ax: plt.Axes, center: Tuple[float, float, float], size: Tuple[float, float, float], color: str) -> None:
    cx, cy, cz = center
    lx, ly, lz = size
    x = np.array([cx - lx / 2, cx + lx / 2])
    y = np.array([cy - ly / 2, cy + ly / 2])
    z = np.array([cz - lz / 2, cz + lz / 2])

    for xi in x:
        for yi in y:
            ax.plot([xi, xi], [yi, yi], [z[0], z[1]], color=color, linewidth=1.6)
    for zi in z:
        ax.plot([x[0], x[1]], [y[0], y[0]], [zi, zi], color=color, linewidth=1.6)
        ax.plot([x[0], x[1]], [y[1], y[1]], [zi, zi], color=color, linewidth=1.6)
        ax.plot([x[0], x[0]], [y[0], y[1]], [zi, zi], color=color, linewidth=1.6)
        ax.plot([x[1], x[1]], [y[0], y[1]], [zi, zi], color=color, linewidth=1.6)


def make_fig_4_3_buoyancy_attitude_density_3d(outpath: str) -> None:
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = _setup_3d_ax(fig)

    layer_depths = np.array([300, 800, 1300, 1800, 2300])
    x = np.linspace(*XLIM, 30)
    y = np.linspace(*YLIM, 30)
    xx, yy = np.meshgrid(x, y)
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(layer_depths)))

    for depth, color in zip(layer_depths, colors):
        zz = np.full_like(xx, depth)
        ax.plot_surface(xx, yy, zz, color=color, alpha=0.25, linewidth=0)

    vehicles = [
        (25, 30, 500, 15),
        (60, 55, 1200, -20),
        (75, 80, 2000, 30),
    ]

    for cx, cy, cz, angle in vehicles:
        _draw_box(ax, (cx, cy, cz), (8, 4, 4), color="tab:gray")
        ax.plot(
            [cx - 6, cx + 6],
            [cy - 2, cy + 2],
            [cz, cz + np.tan(np.deg2rad(angle)) * 6],
            color="tab:orange",
            linewidth=2,
        )

        ax.quiver(cx, cy, cz, 0, 0, 220, color="black", length=1, normalize=False)
        buoyancy = 200 + (cz / 3000) * 80
        ax.quiver(cx, cy, cz, 0, 0, -buoyancy, color="tab:green", length=1, normalize=False)
        ax.quiver(cx, cy, cz, 8, -6, -20, color="tab:red", length=1, normalize=False)

    ax.plot([], [], [], color="tab:blue", label="Density layers")
    ax.plot([], [], [], color="tab:green", label="Buoyancy")
    ax.plot([], [], [], color="black", label="Gravity")
    ax.plot([], [], [], color="tab:red", label="Equivalent perturbation")
    ax.legend(loc="upper left")

    _save_fig(fig, outpath)


def make_fig_4_4_stochastic_diffusion_3d(outpath: str) -> None:
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = _setup_3d_ax(fig)

    rng = np.random.default_rng(42)

    base_mean = np.array([40, 40, 1200])
    t0 = rng.normal(loc=base_mean, scale=[3, 3, 40], size=(200, 3))
    t1 = rng.normal(loc=base_mean + [8, 6, 120], scale=[8, 8, 120], size=(300, 3))
    t2 = rng.normal(loc=base_mean + [18, 14, 260], scale=[14, 14, 220], size=(400, 3))

    ax.scatter(t0[:, 0], t0[:, 1], t0[:, 2], color="tab:blue", alpha=0.8, s=12, label="t0")
    ax.scatter(t1[:, 0], t1[:, 1], t1[:, 2], color="tab:orange", alpha=0.5, s=10, label="t1")
    ax.scatter(t2[:, 0], t2[:, 1], t2[:, 2], color="tab:purple", alpha=0.25, s=8, label="t2")

    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    center = base_mean + np.array([18, 14, 260])
    rx, ry, rz = 25, 25, 350
    ex = center[0] + rx * np.sin(phi) * np.cos(theta)
    ey = center[1] + ry * np.sin(phi) * np.sin(theta)
    ez = center[2] + rz * np.cos(phi)
    ax.plot_wireframe(ex, ey, ez, color="tab:gray", alpha=0.4, linewidth=0.6)

    ax.legend(loc="upper left")

    _save_fig(fig, outpath)


def main() -> None:
    out_dir = os.path.join("outputs", "locate", "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    make_fig_4_1_constraints_3d(os.path.join(out_dir, "Fig_4_1_constraints_3D.png"))
    make_fig_4_2_currents_3d(os.path.join(out_dir, "Fig_4_2_currents_3D.png"))
    make_fig_4_3_buoyancy_attitude_density_3d(
        os.path.join(out_dir, "Fig_4_3_buoyancy_attitude_density_3D.png")
    )
    make_fig_4_4_stochastic_diffusion_3d(os.path.join(out_dir, "Fig_4_4_stochastic_diffusion_3D.png"))


if __name__ == "__main__":
    main()
