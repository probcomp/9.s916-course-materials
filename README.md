# 9.s916 Course Materials

This repository contains the course materials for 9.s916, a course on probabilistic programming and Bayesian methods taught by Vikash Mansinghka at MIT.

## Course Resources

MIT students can access course videos, assignments, and the full syllabus on the [course Canvas page](https://canvas.mit.edu/courses/32225).

## Setup

This repository uses [pixi](https://github.com/prefix-dev/pixi) for environment management. To get started:


1. Clone and navigate to the repository:

```bash
git clone https://github.com/ChiSym/9.s916-course-materials.git
cd 9.s916-course-materials
```

2. Run `scripts/bootstrap-env.sh` to install pixi and create the Python virtual environment
3. Restart your terminal (or re-source your shell config) to ensure pixi is on your PATH

### IDE Setup

If using VS Code or Cursor:

1. Run the "Python: Select Interpreter" command (Ctrl/Cmd + Shift + P)
2. Select `.pixi/envs/default/bin/python` as your Python interpreter
3. When opening Jupyter notebooks, select the "default" kernel

## Repository Contents

The `src` directory contains the following materials:

### Localization Tutorial

A comprehensive tutorial on probabilistic robot localization implemented in JAX and Gen:

- `localization-tutorial.py` - The source file in Jupytext percent format
- `localization-tutorial.ipynb` - The Jupyter notebook generated from the .py file

The tutorial demonstrates:

- Modeling robot motion and sensor observations
- Implementing particle filtering for state estimation
- Visualizing robot paths and particle distributions
- Working with probabilistic programming concepts in JAX/Gen

The notebook can be run interactively using JupyterLab or any Jupyter-compatible editor:

```bash
# Run JupyterLab on CPU
pixi run lab

# Run JupyterLab on GPU
pixi run -e gpu lab
```

# License

The course materials are licensed under the MIT license. See the LICENSE file for details.
