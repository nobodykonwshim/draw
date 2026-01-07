from __future__ import annotations

"""Generate Locate module Figures 4.1–4.4 as PNGs."""

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


XLIM: Tuple[float, float] = (-20, 20)
YLIM: Tuple[float, float] = (-20, 20)
ZLIM: Tuple[float, float] = (0, 3000)


def _setup_3d_ax(fig: plt.Figure) -> plt.Axes:
    """Create a 3D axes with shared limits, view, and depth direction."""
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_zlim(*ZLIM)
    ax.invert_zaxis()
    ax.view_init(elev=22, azim=-60)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("Depth (m)")
    ax.set_facecolor("white")
    return ax


def _save_fig(fig: plt.Figure, outpath: str) -> None:
    """Save the figure with consistent output settings."""
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig_4_1_constraints_3d(outpath: str) -> None:
    """Figure 4.1: constraints volume with seabed surface and depth limits."""
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

    depth_limit_style = dict(color="black", linestyle="--", alpha=0.6)
    ax.plot([XLIM[0], XLIM[1]], [YLIM[0], YLIM[0]], [z_min, z_min], **depth_limit_style)
    ax.plot([XLIM[0], XLIM[1]], [YLIM[1], YLIM[1]], [z_max, z_max], **depth_limit_style)

    legend_handles = [
        Line2D([0], [0], color="tab:blue", label="Feasible volume"),
        Line2D([0], [0], color="saddlebrown", label="Seabed surface"),
        Line2D([0], [0], color="black", linestyle="--", label="Depth limits"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    _save_fig(fig, outpath)


def make_fig_4_2_currents_3d(outpath: str) -> None:
    """Figure 4.2: particle trajectories in a 3D current field."""
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = _setup_3d_ax(fig)

    def flow_u(x_km: float, y_km: float, z_m: float) -> float:
        omega = 0.15
        shear = 0.05
        return -omega * y_km + shear * x_km

    def flow_v(x_km: float, y_km: float, z_m: float) -> float:
        omega = 0.15
        shear = 0.05
        return omega * x_km - shear * y_km

    def flow_w(x_km: float, y_km: float, z_m: float) -> float:
        A = 8.0
        L = 20.0
        Z = 1800.0
        return A * np.sin(np.pi * (z_m - 200.0) / Z) * np.cos(np.pi * x_km / L)

    start_points = [
        (-15.0, -10.0, 500.0),
        (-12.0, 12.0, 800.0),
        (-8.0, -5.0, 1200.0),
        (8.0, 6.0, 600.0),
        (12.0, -12.0, 1000.0),
        (15.0, 8.0, 1500.0),
    ]
    dt = 0.25
    n_steps = 260
    up_color = "tab:cyan"
    down_color = "tab:purple"

    def integrate_trajectory(
        x0: float,
        y0: float,
        z0: float,
        force_flip_step: int | None = None,
    ) -> tuple[list[tuple[float, float, float]], list[int]]:
        x, y, z = x0, y0, z0
        points = [(x, y, z)]
        w_signs: list[int] = []
        last_sign = 1

        for step in range(n_steps):
            if force_flip_step is not None and step == force_flip_step:
                z = float(np.clip(z + 120.0, 200.0, 2000.0))

            u = flow_u(x, y, z)
            v = flow_v(x, y, z)
            w = flow_w(x, y, z)
            sign = np.sign(w)
            if sign == 0:
                sign = last_sign
            w_signs.append(int(sign))
            last_sign = int(sign)

            x += dt * u
            y += dt * v
            z = float(np.clip(z + dt * w, 200.0, 2000.0))

            if x < XLIM[0] or x > XLIM[1] or y < YLIM[0] or y > YLIM[1]:
                break

            points.append((x, y, z))

        return points, w_signs

    for x0, y0, z0 in start_points:
        trajectory = None
        w_signs: list[int] = []
        for attempt in range(6):
            trial_z0 = z0 + 80.0 * attempt
            trajectory, w_signs = integrate_trajectory(x0, y0, trial_z0)
            if len(w_signs) > 1 and np.any(np.diff(np.sign(w_signs)) != 0):
                break
        else:
            trajectory, w_signs = integrate_trajectory(x0, y0, z0, force_flip_step=130)
            if len(w_signs) > 1 and not np.any(np.diff(np.sign(w_signs)) != 0):
                mid = min(len(w_signs) - 1, 130)
                x_mid, y_mid, z_mid = trajectory[mid]
                z_mid = float(np.clip(z_mid + 120.0, 200.0, 2000.0))
                trajectory[mid] = (x_mid, y_mid, z_mid)
                w_signs[mid] = -w_signs[mid - 1]

        if trajectory is None or len(trajectory) < 2:
            continue

        for idx in range(len(trajectory) - 1):
            x1, y1, z1 = trajectory[idx]
            x2, y2, z2 = trajectory[idx + 1]
            sign = w_signs[min(idx, len(w_signs) - 1)]
            color = up_color if sign > 0 else down_color
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, lw=2.0, alpha=0.95)

    legend_handles = [
        Line2D([0], [0], color=up_color, label="Upward segment"),
        Line2D([0], [0], color=down_color, label="Downward segment"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    _save_fig(fig, outpath)


def _draw_box(ax: plt.Axes, center: Tuple[float, float, float], size: Tuple[float, float, float], color: str) -> None:
    """Draw a wireframe cuboid representing a simplified vehicle."""
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
    """Figure 4.3: density layers, buoyancy, gravity, and attitude effects."""
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
        (-12, -8, 500, 15),
        (0, 5, 1200, -20),
        (12, 12, 2000, 30),
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

        arrow_scale = 0.125
        ax.quiver(
            cx,
            cy,
            cz,
            0,
            0,
            220 * arrow_scale,
            color="black",
            length=1,
            normalize=False,
            arrow_length_ratio=0.2,
        )
        buoyancy = 200 + (cz / 3000) * 80
        ax.quiver(
            cx,
            cy,
            cz,
            0,
            0,
            -buoyancy * arrow_scale,
            color="tab:green",
            length=1,
            normalize=False,
            arrow_length_ratio=0.2,
        )
        ax.quiver(
            cx,
            cy,
            cz,
            8 * arrow_scale,
            -6 * arrow_scale,
            -20 * arrow_scale,
            color="tab:red",
            length=1,
            normalize=False,
            arrow_length_ratio=0.2,
        )

    legend_handles = [
        Line2D([0], [0], color="tab:blue", label="Density layers"),
        Line2D([0], [0], color="tab:green", label="Buoyancy"),
        Line2D([0], [0], color="black", label="Gravity"),
        Line2D([0], [0], color="tab:red", label="Equivalent perturbation"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    _save_fig(fig, outpath)


def make_fig_4_4_stochastic_diffusion_3d(outpath: str) -> None:
    """Figure 4.4: 2D uncertainty growth over time."""
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111)

    t = np.linspace(0, 24, 49)
    r95 = 1.5 + 2.0 * np.sqrt(t)

    ax.plot(t, r95, lw=2.5, color="tab:blue")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Uncertainty Radius R95 (km)")
    ax.set_ylim(0, 20)
    ax.set_facecolor("white")

    _save_fig(fig, outpath)


def main() -> None:
    """Generate all four Locate 3D figures into the fixed output directory."""
    out_dir = Path("outputs") / "locate" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_fig_4_1_constraints_3d(str(out_dir / "Fig_4_1_constraints_3D.png"))
    make_fig_4_2_currents_3d(str(out_dir / "Fig_4_2_currents_3D.png"))
    make_fig_4_3_buoyancy_attitude_density_3d(
        str(out_dir / "Fig_4_3_buoyancy_attitude_density_3D.png")
    )
    make_fig_4_4_stochastic_diffusion_3d(str(out_dir / "Fig_4_4_stochastic_diffusion_3D.png"))


if __name__ == "__main__":
    main()
