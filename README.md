# draw

Generate the Locate module Section 4.2 3D schematic figures using a standalone
Python script. The script relies only on NumPy and Matplotlib (including
`mpl_toolkits.mplot3d`) and produces four PNGs with fixed camera settings and
axis ranges.

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

## Usage

Run the script from the repository root:

```bash
python make_locate_figures_3d.py
```

The images are written to:

```
outputs/locate/visualizations/
```

with the following filenames:

- `Fig_4_1_constraints_3D.png`
- `Fig_4_2_currents_3D.png`
- `Fig_4_3_buoyancy_attitude_density_3D.png`
- `Fig_4_4_stochastic_diffusion_3D.png`
