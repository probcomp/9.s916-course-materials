# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
# pyright: reportUnusedExpression=false
# %%
# import sys

# if "google.colab" in sys.modules:
#     from google.colab import auth  # pyright: ignore [reportMissingImports]

#     auth.authenticate_user()
#     %pip install --quiet keyring keyrings.google-artifactregistry-auth  # type: ignore # noqa
#     %pip install --quiet genjax==0.7.0 genstudio==2024.9.7 --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/  # type: ignore # noqa
# %% [markdown]
# # Localization Tutorial
#
# This notebook provides an introduction to probabilistic computation (ProbComp). This term refers to a way of expressing probabilistic constructs in a computational paradigm, made precise by a probabilistic programming language (PPL). The programmer can encode their probabilistic intuition for solving a problem into an algorithm. Back-end language work automates the routine but error-prone derivations.
#
# Dependencies are specified in pyproject.toml.
# %%
# Global includes

import json
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import genjax
from genjax import SelectionBuilder as S
from genjax import ChoiceMapBuilder as C
from genjax.typing import Array, FloatArray, PRNGKey, IntArray
from penzai import pz
from typing import TypeVar, Generic, Callable
from genstudio.plot import js

html = Plot.Hiccup

# %% [markdown]
# ## Sensing a robot's location on a map
#
# ### The map
#
# The tutorial will revolve around modeling the activity of a robot within some space.  A large simplifying assumption, which could be lifted with more effort, is that we have been given a *map* of the space, to which the robot will have access.
#
# The code below loads such a map, along with other data for later use.  Generally speaking, we keep general code and specific examples in separate cells, as signposted here.

# %%
# General code here

def create_segments(points):
    """
    Given an array of points of shape (N, 2), return an array of
    pairs of points. [p_1, p_2, p_3, ...] -> [[p_1, p_2], [p_2, p_3], ...]
    where each p_i is [x_i, y_i]
    """
    return jnp.stack([points, jnp.roll(points, shift=-1, axis=0)], axis=1)


def make_world(wall_verts, clutters_vec):
    """
    Constructs the world by creating segments for walls and clutters, calculates the bounding box, and prepares the simulation parameters.

    Args:
    - wall_verts (list of list of float): A list of 2D points representing the vertices of walls.
    - clutters_vec (list of list of list of float): A list where each element is a list of 2D points representing the vertices of a clutter.
    - start (Pose): The starting pose of the robot.
    - controls (list of Control): Control actions for the robot.

    Returns:
    - tuple: A tuple containing the world configuration, the initial state, and the total number of control steps.
    """
    # Create segments for walls and clutters
    walls = create_segments(wall_verts)
    clutters = jax.vmap(create_segments)(clutters_vec)

    # Combine all points for bounding box calculation
    all_points = jnp.vstack(
        (jnp.array(wall_verts), jnp.concatenate(clutters_vec))
    )
    x_min, y_min = jnp.min(all_points, axis=0)
    x_max, y_max = jnp.max(all_points, axis=0)

    # Calculate bounding box, box size, and center point
    bounding_box = jnp.array([[x_min, x_max], [y_min, y_max], [-jnp.pi, +jnp.pi]])
    box_size = max(x_max - x_min, y_max - y_min)
    center_point = jnp.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

    return {
            "walls": walls,
            "wall_verts": wall_verts,
            "clutters": clutters,
            "bounding_box": bounding_box,
            "box_size": box_size,
            "center_point": center_point,
        }

def load_file(file_name):
    # load from cwd or its parent
    # (differs depending on dev environment)
    try:
        with open(file_name) as f:
            return json.load(f)
    except FileNotFoundError:
        with open(f"../{file_name}") as f:
            return json.load(f)

def load_world(file_name):
    """
    Loads the world configuration from a specified file and constructs the world.

    Args:
    - file_name (str): The name of the file containing the world configuration.

    Returns:
    - tuple: A tuple containing the world configuration, the initial state, and the total number of control steps.
    """
    # Try both the direct path and one directory up
    data = load_file(file_name)

    walls_vec = jnp.array(data["wall_verts"])
    clutters_vec = jnp.array(data["clutter_vert_groups"])

    return make_world(walls_vec, clutters_vec)


# %%
# Specific example code here

world = load_world("world.json")

# %% [markdown]
# ### Plotting
#
# It is crucial to picture what we are doing at all times, so we develop plotting code early and often.

# %%
walls_plot = Plot.new(
    Plot.line(
        world["wall_verts"],
        strokeWidth=2,
        stroke="#ccc",
    ),
    {"margin": 0, "inset": 50, "width": 500, "axis": None, "aspectRatio": 1},
    Plot.domain(world["bounding_box"][0]),
)

world_plot = (
    walls_plot
    + Plot.frame(strokeWidth=4, stroke="#ddd")
    + Plot.color_legend()
)

clutters_plot = (
    [Plot.line(c[:, 0], fill=Plot.constantly("clutters")) for c in world["clutters"]],
    Plot.color_map({"clutters": "magenta"}),
)

(
    world_plot
    + clutters_plot
    + {"title": "Given data"}
)

# %% [markdown]
# Following this initial display of the given data, we *suppress the clutters* until much later in the notebook.

# %%
(
    world_plot
    + {"title": "Given data"}
)


# %% [markdown]
# ### Robot poses
#
# We will model the robot's physical state as a *pose* (or mathematically speaking a ray), defined to be a *position* (2D point relative to the map) plus a *heading* (angle from -$\pi$ to $\pi$).
#
# These will be visualized using arrows whose tip is at the position, and whose direction indicates the heading.

# %%
@pz.pytree_dataclass
class Pose(genjax.PythonicPytree):
    p: FloatArray
    hd: FloatArray

    def __repr__(self):
        return f"Pose(p={self.p}, hd={self.hd})"

    def as_array(self):
        return jnp.append(self.p, self.hd)

    def as_dict(self):
        return {"p": self.p, "hd": self.hd}

    def dp(self):
        return jnp.array([jnp.cos(self.hd), jnp.sin(self.hd)])

    def step_along(self, s: float) -> "Pose":
        """
        Moves along the direction of the pose by a scalar and returns a new Pose.

        Args:
            s (float): The scalar distance to move along the pose's direction.

        Returns:
            Pose: A new Pose object representing the moved position.
        """
        new_p = self.p + s * self.dp()
        return Pose(new_p, self.hd)

    def apply_control(self, control):
        return Pose(self.p + control.ds * self.dp(), self.hd + control.dhd)

    def rotate(self, a: float) -> "Pose":
        """
        Rotates the pose by angle 'a' (in radians) and returns a new Pose.

        Args:
            a (float): The angle in radians to rotate the pose.

        Returns:
            Pose: A new Pose object representing the rotated pose.
        """
        return Pose(self.p, self.hd + a)

# %%
def pose_wings(pose, opts={}):
    return Plot.line(js("""
                   const pose = %1;
                   let positions = pose.p;
                   let angles = pose.hd;
                   if (typeof angles === 'number') {{
                       positions = [positions];
                       angles = [angles];
                   }}
                   return Array.from(positions).flatMap((p, i) => {{
                     const angle = angles[i]
                     const wingAngle = Math.PI / 12
                     const wingLength = 0.6
                     const wing1 = [
                       p[0] - wingLength * Math.cos(angle + wingAngle),
                       p[1] - wingLength * Math.sin(angle + wingAngle),
                       i
                     ]
                     const center = [p[0], p[1], i]
                     const wing2 = [
                       p[0] - wingLength * Math.cos(angle - wingAngle),
                       p[1] - wingLength * Math.sin(angle - wingAngle),
                       i
                     ]
                     return [wing1, center, wing2]
                   }})
                   """, pose, expression=False),
                z="2",
                **opts)

def pose_body(pose, opts={}):
    return Plot.dot(js("typeof %1.hd === 'number' ? [%1.p] : %1.p", pose), {"r": 4} | opts)

def pose_plots(poses, wing_opts={}, body_opts={}, **opts):
    """
    Creates a plot visualization for one or more poses.

    Args:
        poses_or_stateKey: Either a collection of poses or a state key string
        **opts: Optional styling applied to both lines and dots. If 'color' is provided,
               it will be used as 'stroke' for lines and 'fill' for dots.

    Returns:
        A plot object showing the poses with direction indicators
    """

    # Handle color -> stroke/fill conversion
    if "color" in opts:
        wing_opts = wing_opts | {"stroke": opts["color"]}
        body_opts = body_opts | {"fill": opts["color"]}
    return (
        pose_wings(poses, opts | wing_opts) + pose_body(poses, opts | body_opts)
    )


def pose_widget(label, initial_pose, **opts):
    return (
        pose_plots(js(f"$state.{label}"),
            render=Plot.renderChildEvents({"onDrag": js(
                f"""
                (e) => {{
                    if (e.shiftKey) {{
                        const dx = e.x - $state.{label}.p[0];
                        const dy = e.y - $state.{label}.p[1];
                        const angle = Math.atan2(dy, dx);
                        $state.update({{{label}: {{hd: angle, p: $state.{label}.p}}}})
                    }} else {{
                        $state.update({{{label}: {{hd: $state.{label}.hd, p: [e.x, e.y]}}}})
                    }}
                }}
                """)}), **opts)
        | Plot.initialState({label: initial_pose.as_dict()}, sync=label)
    )

# %%
some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))

Plot.html("Click-drag on pose to change location.  Shift-click-drag on pose to change heading.") | (
    world_plot
    + pose_widget("pose", some_pose, color="blue")
) | Plot.html(js("`pose = Pose([${$state.pose.p.map((x) => x.toFixed(2))}], ${$state.pose.hd.toFixed(2)})`"))

# %% [markdown]
# A static picture in case of limited interactivity:

# %%
key = jax.random.key(0)

def random_pose(k):
    p_array = jax.random.uniform(k, shape=(3,),
        minval=world["bounding_box"][:, 0],
        maxval=world["bounding_box"][:, 1])
    return Pose(p_array[0:2], p_array[2])

some_poses = jax.vmap(random_pose)(jax.random.split(key, 20))

(
    world_plot
    + pose_plots(some_poses, color="green")
    + {"title": "Some poses"}
)


# %% [markdown]
# ### Ideal sensors
#
# The robot will need to reason about its location on the map, on the basis of LIDAR-like sensor data.
#
# An "ideal" sensor reports the exact distance cast to a wall.  (It is capped off at a max value in case of error.)

# %%
def distance(p, seg, PARALLEL_TOL=1.0e-6):
    """
    Computes the distance from a pose to a segment, considering the pose's direction.

    Args:
    - p: The Pose object.
    - seg: The segment [p1, p2].

    Returns:
    - float: The distance to the segment. Returns infinity if no valid intersection is found.
    """
    pdp = p.dp()
    segdp = seg[1] - seg[0]
    # Compute unique s, t such that p.p + s * pdp == seg[0] + t * segdp
    pq = p.p - seg[0]
    det = pdp[0] * segdp[1] - pdp[1] * segdp[0]
    st = jnp.where(
        jnp.abs(det) < PARALLEL_TOL,
        jnp.array([jnp.nan, jnp.nan]),
        jnp.array([
            segdp[0] * pq[1] - segdp[1] * pq[0],
              pdp[0] * pq[1] -   pdp[1] * pq[0]
        ]) / det
    )
    return jnp.where(
        (st[0] >= 0) & (st[1] >= 0) & (st[1] <= 1),
        st[0],
        jnp.inf
    )


# %%
sensor_settings = {
    "fov": 2 * jnp.pi * (2 / 3),
    "num_angles": 41,
    "box_size": world["box_size"],
}

def sensor_distance(pose, walls, box_size):
    d = jnp.min(jax.vmap(distance, in_axes=(None, 0))(pose, walls))
    # Capping to a finite value avoids issues below.
    return jnp.where(jnp.isinf(d), 2 * box_size, d)

# This represents a "fan" of sensor angles, with given field of vision, centered at angle 0.

def make_sensor_angles(sensor_settings):
    na = sensor_settings["num_angles"]
    return sensor_settings["fov"] * (jnp.arange(na) - ((na - 1) / 2)) / (na - 1)

sensor_angles = make_sensor_angles(sensor_settings)

def ideal_sensor(pose):
    return jax.vmap(
        lambda angle: sensor_distance(pose.rotate(angle), world["walls"], sensor_settings["box_size"])
    )(sensor_angles)


# %%
# Plot sensor data.

def plot_sensors(pose, readings, sensor_angles, show_legend=False):
    return Plot.Import("""export const projections = (pose, readings, angles) => Array.from({length: readings.length}, (_, i) => {
                const angle = angles[i] + pose.hd
                const reading = readings[i]
                return [pose.p[0] + reading * Math.cos(angle), pose.p[1] + reading * Math.sin(angle)]
            })""",
            refer=["projections"]) | (
        Plot.line(
            js("projections(%1, %2, %3).flatMap((projection, i) => [%1.p, projection, i])", pose, readings, sensor_angles),
            opacity=0.1,
        ) +
        Plot.dot(
            js("projections(%1, %2, %3)", pose, readings, sensor_angles),
            r=2.75,
            fill="#f80"
        ) +
        Plot.cond(show_legend, Plot.colorMap({"sensor rays": "rgb(0,0,0,0.1)", "sensor readings": "#f80"}) | Plot.colorLegend())
    )


# %%
def pose_at(state, label):
    pose_dict = getattr(state, label)
    return Pose(jnp.array(pose_dict["p"]), jnp.array(pose_dict["hd"]))

def update_ideal_sensors(widget, label):
    widget.state.update({
        (label + "_readings"): ideal_sensor(pose_at(widget.state, label))
    })


some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))
(
    (
        world_plot
        + plot_sensors(js("$state.pose"), js("$state.pose_readings"), sensor_angles, show_legend=True)
        + pose_widget("pose", some_pose, color="blue")
    )
    | Plot.html(js("`pose = Pose([${$state.pose.p.map((x) => x.toFixed(2))}], ${$state.pose.hd.toFixed(2)})`"))
    | Plot.initialState({
        "pose_readings": ideal_sensor(some_pose)
    })
    | Plot.onChange({
        "pose": lambda widget, _: update_ideal_sensors(widget, "pose")
    })
)

# %% [markdown]
# Some pictures in case of limited interactivity:

# %%
some_readings = jax.vmap(ideal_sensor)(some_poses)

Plot.Frames([
    (
        world_plot
        + plot_sensors(pose, some_readings[i], sensor_angles, show_legend=True)
        + pose_plots(pose)
    )
    for i, pose in enumerate(some_poses)
], fps=2)

# %% [markdown]
# ## First steps in modeling uncertainty using Gen
#
# The robot will need to reason about its possible location on the map using incomplete information—in a pun, it must nagivate the uncertainty.  The `Gen` system facilitates programming the required probabilistic logic.  We will introduce the features of Gen, starting with some simple features now, and bringing in more complex ones later.
#
# Each piece of the model is declared as a *generative function* (GF).  The `Gen` library provides the decorator `@genjax.gen` for constructing GFs.  The library moreover offers primitive *distributions* such as "Bernoulli" and "normal", and the *sampling operator* `@`.  GFs may sample from distributions and, recursively, other GFs using `@`.  A generative function embodies the *joint distribution* over the latent choices indicated by the sampling operations.

# %% [markdown]
# ### Creating noisy measurements using `propose`
#
# We have on hand two kinds of things to model: the robot's pose (and possibly its motion), and its sensor data.  We tackle the sensor model first because it is simpler.
#
# Its declarative model in `Gen` starts with the case of just one sensor reading:

# %%
model_sensor_noise = 0.1

@genjax.gen
def sensor_model_one(pose, angle, sensor_noise):
    return (
        genjax.normal(
            sensor_distance(pose.rotate(angle), world["walls"], sensor_settings["box_size"]),
            sensor_noise,
        )
        @ "distance"
    )


# %% [markdown]
# Under this model, a computed sensor distsance is used as the mean of a Gaussian distribution (representing our uncertainty about it).  *Sampling* from this distribution, using the `@` operator, occurs at the address `"distance"`.
#
# We draw samples from `sensor_model_one` with `propose` semantics.  Since this operation is stochastic, the method is called with a PRNG key in addition to a tuple of model arguments.  The code is then run, performing the required draws from the sampling operations.  The random draws get organized according to their addresses, forming a *choice map* data structure.  This choice map, a score (to be discussed below), and the return value are all returned by `propose`.

# %%
key, sub_key = jax.random.split(key)
some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))
cm, score, retval = sensor_model_one.propose(sub_key, (some_pose, sensor_angles[0], model_sensor_noise))
retval

# %% [markdown]
# The choice map records the Gaussian draw at address `"distance"`, whose value agrees with the return value in this case.

# %%
cm

# %% [markdown]
# We are interested in the related model whose *single* draw consists of a *vector* of the sensor distances computed across the vector of sensor angles.  This is exactly what we get using the GenJAX `vmap` combinator on GFs.

# %%
sensor_model = sensor_model_one.vmap(in_axes=(None, 0, None))

# %%
key, sub_key = jax.random.split(key)
some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))
cm, score, retval = sensor_model.propose(sub_key, (some_pose, sensor_angles, model_sensor_noise))
retval

# %% [markdown]
# Now for choice map:

# %%
cm


# %% [markdown]
# We see the address `"distance"` now associated with the array of Gaussian draws.
#
# With a little wrapping, one gets a function of the same type as `ideal_sensor`, ignoring the PRNG key.

# %%
def noisy_sensor(key, pose, sensor_noise):
    return sensor_model.propose(key, (pose, sensor_angles, sensor_noise))[2]


# %%
def noise_slider(key, label, init):
    return Plot.Slider(
        key=key,
        label=label,
        showValue=True,
        range=[0.01, 5.0],
        step=0.01,
    ) | Plot.initialState({key: init}, sync={key})

def update_noisy_sensors(widget, pose_key, slider_key):
    k1, k2 = jax.random.split(jax.random.wrap_key_data(widget.state.k))
    readings = noisy_sensor(k1, pose_at(widget.state, pose_key), float(getattr(widget.state, slider_key)))
    widget.state.update({
        "k": jax.random.key_data(k2),
        (pose_key + "_readings"): readings
    })
    return readings


# %%
def on_slider_change(widget, _):
    update_noisy_sensors(widget, "pose", "noise_slider")

key, k1, k2 = jax.random.split(key, 3)
some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))
(
    (
        world_plot
        + plot_sensors(js("$state.pose"), js("$state.pose_readings"), sensor_angles)
        + pose_widget("pose", some_pose, color="blue")
    )
    | noise_slider("noise_slider", "Sensor noise =", model_sensor_noise)
    | Plot.html(js("`pose = Pose([${$state.pose.p.map((x) => x.toFixed(2))}], ${$state.pose.hd.toFixed(2)})`"))
    | Plot.initialState({
        "k": jax.random.key_data(k1),
        "pose_readings": noisy_sensor(k2, some_pose, model_sensor_noise)
    }, sync={"k"})
    | Plot.onChange({"pose": on_slider_change, "noise_slider": on_slider_change})
)

# %% [markdown]
# ### Weighing data with `assess`
#
# The mathematical picture is as follows.  Given the parameters of a pose $z$, walls $w$, and settings $\nu$, one gets a distribution $\text{sensor}(z, w, \nu)$ over certain choice maps.  The supporting choice maps are identified with vectors $o = o^{(1:J)} = (o^{(1)}, o^{(2)}, \ldots, o^{(J)})$ of observations, where $J := \nu_\text{num\_angles}$, each $o^{(j)}$ independently following a certain normal distribution (depending, notably, on a distance to a wall).  Thus the density of $o$ factors into a product of the form
# $$
# P_\text{sensor}(o) = \prod\nolimits_{j=1}^J P_\text{normal}(o^{(j)})
# $$
# where we begin a habit of omitting the parameters to distributions that are implied by the code.
#
# As `propose` draws a sample, it simultaneously computes this density or *score* and returns its logarithm:

# %%
jnp.exp(score)

# %% [markdown]
# There are many scenarios where one has on hand a full set of data, perhaps via observation, and seeks their score according to the model.  One could write a program by hand to do this—but one would simply recapitulate the code for `noisy_sensor`, where the sampling operations would be replaced with density computations, and their log product would be returned.
#
# The construction of a log density function is automated by the `assess` semantics for generative functions.  This method is passed a choice map and a tuple of arguments, and it returns the log score plus the return value.

# %%
some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))
score, retval = sensor_model.assess(cm, (some_pose, sensor_angles, model_sensor_noise))
jnp.exp(score)


# %% [markdown]
# ## First steps in probabilistic reasoning
#
# We consider the following problem: given an batch of sensor data (plus the map), decide the likely robot pose where it was recorded.
#
# ### Gaining an intuition
#
# Theory point: likelihood versus posterior —> "theory exercises".  Maybe just name here our uniform prior, and point towards later in the nb where we use non-uniform priors.  Encourage students to adapt this part of the notebook to nonuniform priors.

# %% [markdown]
# In the following widget, a hidden "target" pose has been chosen, and the goal is to reason about where it might be.  You are given aids in reasoning:
# * Sensor readings have been taken at the target pose, and attached to a manipulable "guess" pose that you may move around and try to fit.
# * These readings are plotted on the right, in comparison with the would-be wall distances from the "guess" pose.
# * The difference beetween these two data sets goes into the computation of the *likelihood* of the guess, which is also shown.

# %%
key, k1, k2, k3 = jax.random.split(key, 4)

guess_pose = Pose(jnp.array([2.0, 16.0]), jnp.array(0.0))
target_pose = Pose(jnp.array([15.0, 4.0]), jnp.array(-1.6))

def likelihood_function(cm, pose, sensor_noise):
    return sensor_model.assess(cm, (pose, sensor_angles, sensor_noise))[0]

def on_guess_pose_chage(widget, _):
    update_ideal_sensors(widget, "guess")
    widget.state.update({"likelihood":
        likelihood_function(
            C["distance"].set(widget.state.target_readings),
            pose_at(widget.state, "guess"),
            model_sensor_noise
        )
    })

def on_target_pose_chage(widget, _):
    update_noisy_sensors(widget, "target", "noise_slider")
    widget.state.update({"likelihood":
        likelihood_function(
            C["distance"].set(widget.state.target_readings),
            pose_at(widget.state, "guess"),
            model_sensor_noise
        )
    })

(
    Plot.Grid(
        (
            world_plot
            + plot_sensors(js("$state.guess"), js("$state.target_readings"), sensor_angles)
            + pose_widget("guess", guess_pose, color="blue")
            + Plot.cond(js("$state.show_target_pose"),
                pose_widget("target", target_pose, color="gold"))
        ),
        (
            Plot.rectY(
                Plot.js("""
                const data = [];
                for (let i = 0; i < $state.guess_readings.length; i++) {
                    data.push({
                        "sensor index": i - 0.15,
                        "distance": $state.guess_readings[i],
                        "group": "wall distances from guess pose"
                    });
                    data.push({
                        "sensor index": i + 0.15,
                        "distance": $state.target_readings[i],
                        "group": "sensor readings from hidden pose"
                    });
                }
                return data;
                """, expression=False),
                x="sensor index",
                y="distance",
                fill="group",
                interval=0.5
            )
            + Plot.domainY([0, 15])
            + {"height": 300, "marginBottom": 50}
            + Plot.color_map({
                "wall distances from guess pose": "blue",
                "sensor readings from hidden pose": "gold"
            })
            + Plot.colorLegend()
            + {"legend": {"anchor": "middle", "x": 0.5, "y": 1.2}}
            | [
                "div",
                {"class": "text-lg mt-2 text-center w-full"},
                Plot.js("'log likelihood (greater is better): ' + $state.likelihood.toFixed(2)")
            ]
        ),
        cols=2
    )
    | noise_slider("noise_slider", "Sensor noise =", model_sensor_noise)
    | (
        Plot.html([
            "label",
            {"class": "flex items-center gap-2 cursor-pointer"},
            [
                "input",
                {
                    "type": "checkbox",
                    "checked": js("$state.show_target_pose"),
                    "onChange": js("(e) => $state.show_target_pose = e.target.checked")
                }
            ],
            "show target pose"
        ])
        & Plot.html(js("`guess = Pose([${$state.guess.p.map((x) => x.toFixed(2))}], ${$state.guess.hd.toFixed(2)})`"))
        & Plot.html(js("`target = Pose([${$state.target.p.map((x) => x.toFixed(2))}], ${$state.target.hd.toFixed(2)})`"))
    )
    | Plot.initialState(
        {
            "k": jax.random.key_data(k1),
            "guess_readings": ideal_sensor(guess_pose),
            "target_readings": (initial_target_readings := noisy_sensor(k3, target_pose, model_sensor_noise)),
            "likelihood": likelihood_function(C["distance"].set(initial_target_readings), guess_pose, model_sensor_noise),
            "show_target_pose": False,
        }, sync={"k", "target_readings"})
    | Plot.onChange({
            "guess": on_guess_pose_chage,
            "target": on_target_pose_chage,
            "noise_slider": on_target_pose_chage,
    })
)


# %% [markdown]
# ### Likelihood, prior, and posterior
#
# A common way to proceed is to consider the likelihood function—the probability density of the fixed observations, with varying pose parameter—as the basis of our reasoning.  Reasoning about the likelihood can be recovered as a special case of reasoning about the *posterior distribution* over the pose parameter, where the *prior* distribution was uniform with respect to some reference volume over poses.
#
# We introduce here three choices of prior, to illustrate how inference depends on it.

# %%
# Uniform prior over the whole map.
# (This is just a recapitulation of `random_pose` from above.)

@genjax.gen
def uniform_pose(mins, maxes):
    p_array = genjax.uniform(mins, maxes) @ "p_array"
    return Pose(p_array[0:2], p_array[2])

whole_map_prior = uniform_pose.partial_apply(
    world["bounding_box"][:, 0],
    world["bounding_box"][:, 1]
)

def whole_map_cm_builder(pose):
    return C["p_array"].set(pose.as_array())


# %%
key, sub_key = jax.random.split(key)
some_poses = jax.vmap(lambda k: whole_map_prior.simulate(k, ()))(jax.random.split(sub_key, 100)).get_retval()

(
    world_plot
    + pose_plots(some_poses, color="green")
    + {"title": "Some poses"}
)

# %%
# Even mixture of uniform priors over two rooms.

room_mixture = jnp.ones(2) / 2
room1 = jnp.array([[12.83, 15.81], [11.19, 15.26], [-jnp.pi, +jnp.pi]])
room2 = jnp.array([[15.73, 18.90], [ 5.79,  9.57], [-jnp.pi, +jnp.pi]])

two_room_prior = genjax.mix(
    uniform_pose.partial_apply(room1[:, 0], room1[:, 1]),
    uniform_pose.partial_apply(room2[:, 0], room2[:, 1])
).partial_apply(jnp.log(room_mixture), (), ())

def two_room_cm_builder(pose):
    return (
        C["mixture_component"].set(jnp.array(pose.p[1] < 10, int))
        | C["component_sample", "p_array"].set(pose.as_array())
    )


# %%
key, sub_key = jax.random.split(key)
some_poses = jax.vmap(lambda k: two_room_prior.simulate(k, ()))(jax.random.split(sub_key, 100)).get_retval()

(
    world_plot
    + pose_plots(some_poses, color="green")
    + {"title": "Some poses"}
)

# %%
# Prior localized around a single pose
pose_for_localized_prior = Pose(jnp.array([2.0, 16.0]), jnp.array(0.0))
spread_of_localized_prior = (0.1, 0.75)
@genjax.gen
def localized_prior():
    p = (
        genjax.mv_normal_diag(
            pose_for_localized_prior.p,
            spread_of_localized_prior[0] * jnp.ones(2)
        )
        @ "p"
    )
    hd = (
        genjax.normal(
            pose_for_localized_prior.hd,
            spread_of_localized_prior[1]
        )
        @ "hd"
    )
    return Pose(p, hd)

def localized_cm_builder(pose):
    return C["p"].set(pose.p) | C["hd"].set(pose.hd)


# %%
key, sub_key = jax.random.split(key)
some_poses = jax.vmap(lambda k: localized_prior.simulate(k, ()))(jax.random.split(sub_key, 100)).get_retval()

(
    world_plot
    + pose_plots(some_poses, color="green")
    + {"title": "Some poses"}
)

# %% [markdown]
# We also introduce joint models, whose densities serve as unnormalized posterior densities.

# %%
model_dispatch = {
    "whole_map": (whole_map_prior, whole_map_cm_builder),
    "two_room": (two_room_prior, two_room_cm_builder),
    "localized": (localized_prior, localized_cm_builder),
}

def make_posterior_density_fn(prior_label, readings, model_noise):
    prior, cm_builder = model_dispatch[prior_label]
    @genjax.gen
    def joint_model():
        pose = prior() @ "pose"
        sensor = sensor_model(pose, sensor_angles, model_noise) @ "sensor"
    return jax.jit(
        lambda pose:
            joint_model.assess(
                C["pose"].set(cm_builder(pose)) | C["sensor", "distance"].set(readings),
                ()
            )[0]
    )


# %% [markdown]
# ### Visualization setup
#
# The next few widgets all operate on the same principle.  A manipulable "camera" pose is shown.  Hitting the button makes the following happen:
# * The "target" pose gets fixed to the camera pose and a batch of sensor readings is sampled at the target.  (The first slider controls the noise in the taking of these readings.)
# * Then some computation (optimization or inference) is performed on these sensor readings and everything gets displayed: the target, its sensor readings, and the computation results.  (The second slider controls the sensor noise assumed by the model in this computation.)

# %%
def on_camera_button(button_handler):
    def handler(widget, _):
        k1, k2 = jax.random.split(jax.random.wrap_key_data(widget.state.k))
        widget.state.update({
            "k": jax.random.key_data(k1),
            "target": widget.state.camera,
        })
        readings = update_noisy_sensors(widget, "target", "world_noise")
        button_handler(widget, k2, readings)
        widget.state.update({
            "target_exists": True,
        })
    return handler

def camera_widget(
        k, camera_pose,
        button_label, button_handler,
        result_plots=Plot.dot([jnp.sum(world["bounding_box"], axis=1)[0:2]], opacity=1),
        bottom_elements=(),
        initial_state={},
        sync=set()):
    return (
        (
            world_plot
            + Plot.cond(js("$state.target_exists"),
                result_plots
                + plot_sensors(js("$state.target"), js("$state.target_readings"), sensor_angles)
                + pose_plots(js("$state.target"), color="red")
            )
            + pose_widget("camera", camera_pose, color="blue")
        )
        | noise_slider("world_noise", "World/data noise = ", model_sensor_noise)
        | Plot.html([
            "p",
            "Prior:",
            [
                "select",
                {"onChange": js("(e) => $state.prior = e.target.value")},
                ["option", {"value": "whole_map", "selected": "True"}, "whole map"],
                ["option", {"value": "two_room"}, "two room"],
                ["option", {"value": "localized"}, "localized"],
            ]
        ])
        | noise_slider("model_noise", "Model/inference noise = ", model_sensor_noise)
        | (
            Plot.html([
                "button",
                {
                    "class": "w-24 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 active:bg-blue-700",
                    "onClick": on_camera_button(button_handler)
                },
                button_label
            ])
            & Plot.html(
                Plot.js("""`camera = Pose([${$state.camera.p.map((x) => x.toFixed(2))}], ${$state.camera.hd.toFixed(2)})`""")
            )
            & Plot.html(
                Plot.js("""$state.target_exists ?
                                `target = Pose([${$state.target.p.map((x) => x.toFixed(2))}], ${$state.target.hd.toFixed(2)})` : ''""")
            )
            & bottom_elements
        )
        | Plot.initialState(
            {
                "k": jax.random.key_data(k),
                "target_exists": False,
                "target": {"p": None, "hd": None},
                "target_readings": [],
                "prior": "whole_map"
            } | initial_state,
            sync=({"k", "target", "camera_readings", "prior"} | sync))
    )


# %% [markdown]
# ### Doing optimization
#
# A common response is to optimize the posterior density.  The field of optimization is vast, but it is not the path for us here, so we just take a quick look.
#
# The idea here is just to search for the good poses by brute force, ranging over a suitable discretization grid of the map.  So in this widget, the button fires a sweep over a grid of possible poses, computing the posterior density of each.  The top `N_keep` are kept, and shown with opacity proportional to their position in that sublist.  Moreover, the best fit is shown in purple.

# %%
def make_grid(bounds, ns):
    return [dim.reshape(-1) for dim in jnp.meshgrid(*(jnp.linspace(*bound, num=n) for (bound, n) in zip(bounds, ns)))]

def make_poses_grid_array(bounds, ns):
    grid_xs, grid_ys, grid_hds = make_grid(bounds, ns)
    return jnp.array([grid_xs, grid_ys]).T, grid_hds

def make_poses_grid(bounds, ns):
    return Pose(*make_poses_grid_array(bounds, ns))


# %%
N_grid = jnp.array([50, 50, 20])
N_keep = 1000  # keep the top this many out of the total `jnp.prod(N_grid)` of them

key, sub_key = jax.random.split(key)

camera_pose = Pose(jnp.array([2.0, 16.0]), jnp.array(0.0))

def grid_search_handler(widget, k, readings):
    model_noise = float(getattr(widget.state, "model_noise"))
    jitted_posterior = make_posterior_density_fn(widget.state.prior, readings, model_noise)

    grid_poses = make_poses_grid(world["bounding_box"], N_grid)
    posterior_densities = jax.vmap(jitted_posterior)(grid_poses)
    best = jnp.argsort(posterior_densities, descending=True)[0:N_keep]
    widget.state.update({
        "grid_poses": grid_poses[best].as_dict(),
        "best": grid_poses[best[0]].as_dict()
    })

camera_widget(
    sub_key,
    camera_pose,
    "grid search",
    grid_search_handler,
    result_plots=(
        pose_plots(js("$state.grid_poses"), color="green", opacity=jnp.arange(1.0, 0.0, -1/N_keep))
        + pose_plots(js("$state.best"), color="purple")
    ),
    bottom_elements=(
        Plot.html(
            # For some reason `toFixed` very stubbonrly malfunctions in the following line:
            Plot.js("""$state.target_exists ?
                                `best = Pose([${$state.best.p.map((x) => x.toFixed(2))}], ${$state.best.hd.toFixed(2)})` : ''""")
        )
    ),
    initial_state={
        "grid_poses": {"p": [], "hd": []},
        "best": {"p": None, "hd": None},
    },
)

# %% [markdown]
# Note how the purple pose is a long way from summarizing the ambiguity in the solution set.

# %% [markdown]
# ### Doing probabilistic inference
#
# We show some ways one might approach this problem probabilistically: our answers to "Where might the robot be?" will all be given in the form of probability distributions.  Moreover, these distributions will be embodied as generative samplers.

# %% [markdown]
# #### Grid approximation sampler
#
# Although computing the grid takes work, afterwards accessing its members is cheap.  Instead of only taking the best fit, we can draw members from the grid with probability in proportion to their posterior density.  The result is the following sampler.

# %%
N_grid = jnp.array([50, 50, 20])
N_samples = 100

key, sub_key = jax.random.split(key)

camera_pose = Pose(jnp.array([15.13, 14.16]), jnp.array(1.5))

def grid_approximation_handler(widget, k, readings):
    model_noise = float(getattr(widget.state, "model_noise"))
    jitted_posterior = make_posterior_density_fn(widget.state.prior, readings, model_noise)

    grid_poses = make_poses_grid(world["bounding_box"], N_grid)
    posterior_densities = jax.vmap(jitted_posterior)(grid_poses)
    def grid_sample_one(k):
        return grid_poses[genjax.categorical.propose(k, (posterior_densities,))[2]]

    grid_samples = jax.vmap(grid_sample_one)(jax.random.split(k, N_samples))
    widget.state.update({
        "sample_poses": grid_samples,
    })

camera_widget(
    sub_key,
    camera_pose,
    "grid sampler",
    grid_approximation_handler,
    result_plots=pose_plots(js("$state.sample_poses"), color="green"),
    initial_state={"sample_poses": {"p": [], "hd": []}},
)

# %% [markdown]
# Two shortcomings are clear.  One is that a single pose tends to dominate the samples, regardless of whether other poses in the grid seem to be plausible fits.  The other is that one never sees any poses that do not belong to the grid!

# %% [markdown]
# #### Importance resampling
#
# What if we need not be systematic—and instead we just try a bunch of points, uniformly over all poses, instead of constrained to a grid?
#
# Here we first draw `N` pre-samples, assess them, and pick a single representative one in probability proportional to its posterior density, to obtain one sample.  The samples obtained this way are then more closely distributed to the posterior.

# %%
N_presamples = 1000
N_samples = 100

key, sub_key = jax.random.split(key)

camera_pose = Pose(jnp.array([15.13, 14.16]), jnp.array(1.5))

def importance_resampling_handler(widget, k, readings):
    model_noise = float(getattr(widget.state, "model_noise"))
    jitted_posterior = make_posterior_density_fn(widget.state.prior, readings, model_noise)

    def importance_resample_one(k):
        k1, k2 = jax.random.split(k)
        presamples = jax.vmap(random_pose)(jax.random.split(k1, N_presamples))
        posterior_densities = jax.vmap(jitted_posterior)(presamples)
        return presamples[genjax.categorical.propose(k2, (posterior_densities,))[2]]

    grid_samples = jax.vmap(importance_resample_one)(jax.random.split(k, N_samples))
    widget.state.update({
        "sample_poses": grid_samples,
    })

camera_widget(
    sub_key,
    camera_pose,
    "importance resampler",
    importance_resampling_handler,
    result_plots=pose_plots(js("$state.sample_poses"), color="green"),
    initial_state={"sample_poses": {"p": [], "hd": []}},
)

# %% [markdown]
# #### Markov chain Monte Carlo
#
# We could also explore the space with a random walk.  Here we guide the particle using the MH rule.

# %%
N_MH_steps = 1000
N_samples = 100

key, sub_key = jax.random.split(key)

camera_pose = Pose(jnp.array([15.13, 14.16]), jnp.array(1.5))

def MCMC_handler(widget, k, readings):
    model_noise = float(getattr(widget.state, "model_noise"))
    jitted_posterior = make_posterior_density_fn(widget.state.prior, readings, model_noise)

    def do_MH_step(pose_posterior_density, k):
        pose, posterior_density = pose_posterior_density
        k1, k2 = jax.random.split(k)
        p_hd = pose.as_array()
        delta = jnp.array([0.5, 0.5, 0.1])
        mins = jnp.maximum(p_hd - delta, world["bounding_box"][:, 0])
        maxs = jnp.minimum(p_hd + delta, world["bounding_box"][:, 1])
        new_p_hd = jax.random.uniform(k1, shape=(3,), minval=mins, maxval=maxs)
        new_pose = Pose(new_p_hd[0:2], new_p_hd[2])
        new_posterior = jitted_posterior(new_pose)
        accept = (jnp.log(genjax.uniform.propose(k2, ())[2]) <= new_posterior - posterior_density)
        return (
            jax.tree.map(
                lambda x, y: jnp.where(accept, x, y),
                (new_pose, posterior_density),
                (pose, posterior_density)
            ),
            None
        )
    def sample_MH_one(k):
        k1, k2 = jax.random.split(k)
        start_pose = random_pose(k1)
        start_posterior = jitted_posterior(start_pose)
        return jax.lax.scan(do_MH_step, (start_pose, start_posterior), jax.random.split(k2, N_MH_steps))[0][0]

    grid_samples = jax.vmap(sample_MH_one)(jax.random.split(k, N_samples))
    widget.state.update({
        "sample_poses": grid_samples,
    })

camera_widget(
    sub_key,
    camera_pose,
    "MCMC trajectories",
    MCMC_handler,
    result_plots=pose_plots(js("$state.sample_poses"), color="green"),
    initial_state={"sample_poses": {"p": [], "hd": []}},
)


# %% [markdown]
# ## Modeling robot motion
#
# ### Robot programs
#
# We also assume given a description of a robot's movement via
# * an estimated initial pose (= position + heading), and
# * a program of controls (= advance distance, followed by rotate heading).

# %%
@pz.pytree_dataclass
class Control(genjax.PythonicPytree):
    ds: FloatArray
    dhd: FloatArray

def load_robot_program(file_name):
    """
    Loads the robot program from a specified file.

    Args:
    - file_name (str): The name of the file containing the world configuration.

    Returns:
    - tuple: A tuple containing the initial state, and the total number of control steps.
    """
    robot_program = load_file(file_name)

    start = Pose(
        jnp.array(robot_program["start_pose"]["p"], dtype=float),
        jnp.array(robot_program["start_pose"]["hd"], dtype=float),
    )

    cs = jnp.array([[c["ds"], c["dhd"]] for c in robot_program["program_controls"]])
    controls = Control(cs[:, 0], cs[:, 1])

    # We prepend a zero-effect control step to the control array. This allows
    # numerous simplifications in what follows: we can consider the initial
    # pose uncertainty as well as each subsequent step to be the same function
    # of current position and control step.
    noop_control = Control(jnp.array(0.0), jnp.array(0.0))
    controls = controls.prepend(noop_control)

    # Determine the total number of control steps
    T = len(controls.ds)

    return ({"start": start, "controls": controls}, T)


# %%
world["bounce"] = 0.1

robot_inputs, T = load_robot_program("robot_program.json")


# %% [markdown]
# Before we can visualize such a program, we will need to model robot motion.

# %% [markdown]
# ### Integrate a path from a starting pose and controls
#
# If the motion of the robot is determined in an ideal manner by the controls, then we may simply integrate to determine the resulting path.  Naïvely, this results in the following.

# %%
def diag(x): return (x, x)

def integrate_controls_unphysical(robot_inputs):
    """
    Integrates the controls to generate a path from the starting pose.

    This function takes the initial pose and a series of control steps (ds for distance, dhd for heading change)
    and computes the resulting path by applying each control step sequentially.

    Args:
    - robot_inputs (dict): A dictionary containing the starting pose and control steps.

    Returns:
    - list: A list of Pose instances representing the path taken by applying the controls.
    """
    return jax.lax.scan(
        lambda pose, control: diag(pose.apply_control(control)),
        robot_inputs["start"],
        robot_inputs["controls"],
    )[1]


# %%
def update_unphysical_path(widget, _):
    start = pose_at(widget.state, "start")
    widget.state.update({
        "path": integrate_controls_unphysical(robot_inputs | {"start": start})
    })

(
    (
        world_plot
        + pose_plots(js("$state.path"), color=Plot.constantly("path from integrating controls (UNphysical)"))
        + pose_widget("start", robot_inputs["start"], color=Plot.constantly("start pose"))
        + Plot.color_map({"start pose": "blue", "path from integrating controls (UNphysical)": "green"})
    )
    | Plot.html(js("`start = Pose([${$state.start.p.map((x) => x.toFixed(2))}], ${$state.start.hd.toFixed(2)})`"))
    | Plot.initialState({
        "path": integrate_controls_unphysical(robot_inputs)
    })
    | Plot.onChange({"start": update_unphysical_path})
)


# %% [markdown]
# This code has the problem that it is **unphysical**: the walls in no way constrain the robot motion.
#
# We employ the following simple physics: when the robot's forward step through a control comes into contact with a wall, that step is interrupted and the robot instead "bounces" a fixed distance from the point of contact in the normal direction to the wall.

# %%
@jax.jit
def physical_step(p1: FloatArray, p2: FloatArray, hd):
    """
    Computes a physical step considering wall collisions and bounces.

    Args:
    - p1, p2: Start and end points of the step.
    - hd: Heading direction.

    Returns:
    - Pose: The new pose after taking the step, considering potential wall collisions.
    """
    # Calculate step direction and length
    step_direction = p2 - p1
    step_length = jnp.linalg.norm(step_direction)
    step_pose = Pose(p1, jnp.arctan2(step_direction[1], step_direction[0]))

    # Calculate distances to all walls
    distances = jax.vmap(distance, in_axes=(None, 0))(step_pose, world["walls"])

    # Find the closest wall
    closest_wall_index = jnp.argmin(distances)
    closest_wall_distance = distances[closest_wall_index]
    closest_wall = world["walls"][closest_wall_index]

    # Calculate wall normal and collision point
    collision_point = p1 + closest_wall_distance * step_pose.dp()
    wall_direction = closest_wall[1] - closest_wall[0]
    normalized_wall_direction = wall_direction / jnp.linalg.norm(wall_direction)
    wall_normal = jnp.array([-normalized_wall_direction[1], normalized_wall_direction[0]])

    # Ensure wall_normal points away from the robot's direction
    wall_normal = jnp.where(
        jnp.dot(step_pose.dp(), wall_normal) > 0, -wall_normal, wall_normal
    )

    # Calculate bounce off point
    bounce_off_point: FloatArray = collision_point + world["bounce"] * wall_normal

    # Determine final position based on whether a collision occurred
    final_position = jnp.where(
        closest_wall_distance > step_length, p2, bounce_off_point
    )

    return Pose(final_position, hd)

def integrate_controls_physical(robot_inputs):
    """
    Integrates controls to generate a path, taking into account physical interactions with walls.

    Args:
    - robot_inputs: Dictionary containing the starting pose and control steps.

    Returns:
    - Pose: A Pose object representing the path taken by applying the controls.
    """
    return jax.lax.scan(
        lambda pose, control: diag(physical_step(
                pose.p, pose.p + control.ds * pose.dp(), pose.hd + control.dhd
            )),
        robot_inputs["start"],
        robot_inputs["controls"],
    )[1]


# %%
path_integrated = integrate_controls_physical(robot_inputs)


# %%
def update_physical_path(widget, _):
    start = pose_at(widget.state, "start")
    widget.state.update({
        "path": integrate_controls_physical(robot_inputs | {"start": start})
    })

(
    (
        world_plot
        + pose_plots(js("$state.path"), color=Plot.constantly("path from integrating controls (physical)"))
        + pose_widget("start", robot_inputs["start"], color=Plot.constantly("start pose"))
        + Plot.color_map({"start pose": "blue", "path from integrating controls (physical)": "green"})
    )
    | Plot.html(js("`start = Pose([${$state.start.p.map((x) => x.toFixed(2))}], ${$state.start.hd.toFixed(2)})`"))
    | Plot.initialState({
        "path": integrate_controls_physical(robot_inputs)
    })
    | Plot.onChange({"start": update_physical_path})
)

# %% [markdown]
# ### Modeling taking steps
#
# The following models attempting to step (constrained by the walls) towards a point with some uncertainty about it.
# %%
@genjax.gen
def step_model(motion_settings, start, control):
    p = (
        genjax.mv_normal_diag(
            start.p + control.ds * start.dp(), motion_settings["p_noise"] * jnp.ones(2)
        )
        @ "p"
    )
    hd = genjax.normal(start.hd + control.dhd, motion_settings["hd_noise"]) @ "hd"
    return physical_step(start.p, p, hd)


# Set the motion settings
degrees = jnp.pi / 180
model_motion_settings = {
    "p_noise": 0.15,
    "hd_noise": 1 * degrees
}

# %% [markdown]
# The reader is especially encouraged, in the following widget, to drag the attempted step beyond a wall, to get a feel for how this model handles the physics.

# %%
N_samples = 50

key, k1, k2 = jax.random.split(key, 3)

def confidence_circle(p, p_noise):
    return Plot.ellipse(
        p,
        r=2.5 * p_noise,
        fill=Plot.constantly("95% confidence region"),
    ) + Plot.color_map({"95% confidence region": "rgba(255,0,0,0.25)"})

def update_confidence_circle(widget, _):
    step = pose_at(widget.state, "step")
    step_vector = step.p - robot_inputs["start"].p

    tilted_start_hd = jnp.atan2(step_vector[1], step_vector[0])
    tilted_start = Pose(robot_inputs["start"].p, tilted_start_hd)

    ds = jnp.linalg.norm(step_vector)
    dhd = (step.hd - tilted_start_hd + jnp.pi) % (2 * jnp.pi) - jnp.pi

    widget.state.update({
        "start": tilted_start.as_dict(),
        "control": {"ds": ds, "dhd": dhd}
    })

    k1, k2 = jax.random.split(jax.random.wrap_key_data(widget.state.k))
    samples = jax.vmap(step_model.propose, in_axes=(0, None))(
        jax.random.split(k1, N_samples),
        (model_motion_settings, tilted_start, Control(ds, dhd)),
    )[2]
    widget.state.update({
        "k": jax.random.key_data(k2),
        "samples": samples.as_dict()
    })

(
    (
        world_plot
        + confidence_circle(js("[$state.step.p]"), model_motion_settings["p_noise"])
        + pose_plots(js("$state.samples"), color=Plot.constantly("samples from the step model"))
        + pose_plots(js("$state.start"), color=Plot.constantly("start pose"))
        + pose_widget("step", robot_inputs["start"], color=Plot.constantly("attempt to step to here"))
        + Plot.color_map({
            "start pose": "black",
            "attempt to step to here": "blue",
            "samples from the step model": "green",
        })
    )
    | Plot.html(js("`control = Control(${$state.control.ds.toFixed(2)}, ${$state.control.dhd.toFixed(2)})`"))
    | Plot.initialState({
        "start": robot_inputs["start"].as_dict(),
        "control": {"ds": 0.0, "dhd": 0.0},
        "k": jax.random.key_data(k1),
        "samples": (
            jax.vmap(step_model.propose, in_axes=(0, None))(
                jax.random.split(k2, N_samples),
                (model_motion_settings, robot_inputs["start"], robot_inputs["controls"][0]),
            )[2].as_dict()
        ),
    }, sync={"k"})
    | Plot.onChange({"step": update_confidence_circle})
)


# %% [markdown]
# ### Modeling a full path
#
# We may succinctly promote a singly-stepping model into a path-stepping model using *generative function combinators*.  In the following code, these transformations take place.
# * `step_model` starts with signature `(motion_settings, start, control) -> step`.
# * `partial_apply(motion_settings)` substitutes the first parameter, to get signature `(start, control) -> step`.
# * `.map(diag)` forms a tuple with two copies of the output, to get signature `(start, control) -> (step, step)`.
# * `.scan()` folds the computation over the second parameter, to get signature `(start, controls) -> (step, steps)`.
#
# Thus `path_model` returns a tuple whose second entry is the sampled path (and whose first entry duplicates the final position).

# %%
@genjax.gen
def path_model(motion_settings):
    return (
        step_model
        .partial_apply(motion_settings)
        .map(diag)
        .scan()(robot_inputs["start"], robot_inputs["controls"])
        @ "steps"
    )


# %% [markdown]
# Note how the model returns a tuple with the terminus of the sampled path, followed by the path itself vectorized into a single `Pose` object.

# %%
key, sub_key = jax.random.split(key)
path_model.propose(sub_key, (model_motion_settings,))[2]


# %% [markdown]
# Here is a single path with confidence circles on each step's draw.

# %%
def plot_path_with_confidence(path, step):
    prev_step = robot_inputs["start"] if step == 0 else path[step - 1]
    return (
        world_plot
        + confidence_circle(
            [prev_step.apply_control(robot_inputs["controls"][step]).p],
            model_motion_settings["p_noise"]
        )
        + [pose_plots(path[i]) for i in range(step)]
        + pose_plots(path[step], color=Plot.constantly("next pose"))
        + Plot.color_map({"previous poses": "black", "next pose": "green"})
    )

key, sample_key = jax.random.split(key)
path = path_model.propose(sample_key, (model_motion_settings,))[2][1]
Plot.Frames(
    [
        plot_path_with_confidence(path, step)
        + Plot.title("Motion model (samples)")
        for step in range(len(path))
    ],
    fps=2,
)

# %% [markdown]
# Here are some independent draws to get an aggregate sense.  Note how the motion noise really piles up!

# %%
N_samples = 12

key, sub_key = jax.random.split(key)
sample_paths = jax.vmap(
    lambda k:
        path_model.propose(k, (model_motion_settings,))[2][1]
)(jax.random.split(sub_key, N_samples))

Plot.html([
    "div.grid.grid-cols-2.gap-4",
    *[walls_plot + pose_plots(path) + {"maxWidth": 300, "aspectRatio": 1} for path in sample_paths]
])


# %% [markdown]
# ### Full model
#
# We fold the sensor model into the motion model to form a "full model", whose traces describe simulations of the entire robot situation as we have described it.

# %%
@genjax.gen
def full_model_kernel(motion_settings, sensor_noise, state, control):
    pose = step_model(motion_settings, state, control) @ "pose"
    sensor_model(pose, sensor_angles, sensor_noise) @ "sensor"
    return pose

@genjax.gen
def full_model(motion_settings, sensor_noise):
    return (
        full_model_kernel
        .partial_apply(motion_settings, sensor_noise)
        .map(diag)
        .scan()(robot_inputs["start"], robot_inputs["controls"])
        @ "steps"
    )


# %% [markdown]
# In the math picture, `full_model` corresponds to a distribution $\text{full}$ over its traces.  Such a trace is identified with of a pair $(z_{0:T}, o_{0:T})$ where $z_{0:T} \sim \text{path}(\ldots)$ and $o_t \sim \text{sensor}(z_t, \ldots)$ for $t=0,\ldots,T$.  The density of this trace is then
# $$\begin{align*}
# P_\text{full}(z_{0:T}, o_{0:T})
# &= P_\text{path}(z_{0:T}) \cdot \prod\nolimits_{t=0}^T P_\text{sensor}(o_t) \\
# &= \big(P_\text{start}(z_0)\ P_\text{sensor}(o_0)\big)
#   \cdot \prod\nolimits_{t=1}^T \big(P_\text{step}(z_t)\ P_\text{sensor}(o_t)\big).
# \end{align*}$$
#

# %%
key, sub_key = jax.random.split(key)
cm, score, retval = full_model.propose(sub_key, (model_motion_settings, model_sensor_noise))
cm


# %% [markdown]
# We now see the emergent tree structure of choice maps.  Under the layer `"steps"` (whose indirection allows us to vary `motion_settings` below), there are addresses `"pose"` and `"sensor"`, which respectively have sub-addresses `"hd", "p"` and `"distance"`.  At each leaf is an array resulting from the `scan` operation, in typical PyTree fashion.

# %% [markdown]
# By this point, visualization is essential.  We will just get a quick picture here, and turn toward a more principled approach immediately thereafter.

# %%
def animate_path_and_sensors(path, readings, frame_key=None):
    return Plot.Frames([
        plot_path_with_confidence(path, step)
        + plot_sensors(pose, readings[step], sensor_angles)
        for step, pose in enumerate(path)
    ], fps=2, key=frame_key)


# %%
animate_path_and_sensors(retval[1], cm["steps", "sensor", "distance"])

# %% [markdown]
# ## From choicemaps to traces
#
# Managing the tuple `(choice map, score, return value)` given to us by `propose` can get unwieldy: just see how the call to `animate_path_and_sensors` above needed to pick and choose from these data.  `Gen` saves us a bunch of trouble by wrapping these data—and more—into a structure called a *trace*.  We pause our reasoning about the robot in order to familiarize ourselves with them.
#
# ### Sampling traces with `simulate`
#
# The `Gen` method `simulate` is called just like `propose` (using a PRNG key plus parameters for the model), and it returns a trace.

# %%
# `simulate` takes the GF plus a tuple of args to pass to it.
key, sub_key = jax.random.split(key)
trace = step_model.simulate(
    sub_key,
    (model_motion_settings, robot_inputs["start"], robot_inputs["controls"][0]),
)
trace


# %% [markdown]
# ### GenJAX API for traces
#
# For starters, the trace contains all the information from `propose`.

# %%
trace.get_choices()
# %%
trace.get_score()

# %%
trace.get_retval()
# %% [markdown]
# One can access from a trace the GF that produced it, along with with model parameters that were supplied.

# %%
trace.get_gen_fn()
# %%
trace.get_args()
# %% [markdown]
# Instead of (the log of) the product of all the primitive choices made in a trace, one can take the product over just a subset using `project`.

# %%
key, sub_key = jax.random.split(key)
selections = [
    genjax.Selection.none(),
    S["p"],
    S["hd"],
    S["p"] | S["hd"]
]
[
    trace.project(k, sel)
    for k, sel in zip(jax.random.split(sub_key, len(selections)), selections)
]

# %% [markdown]
# Since the trace object has a lot going on, we use the Penzai visualization library to render the result. Click on the various nesting arrows to explore the structure.

# %%
pz.ts.display(trace)

# %% [markdown]
# ### Modifying traces
#
# The metaprogramming approach of Gen affords the opportunity to explore alternate stochastic execution histories.  Namely, the `update` method of a trace takes modifications to its arguments and primitive choice values, and returns an accordingly modified trace. It also returns (the log of) the ratio of the updated trace's density to the original trace's density, together with a precise record of the resulting modifications that played out.
#
# One could, for instance, consider just the placement of the first step, and replace its stochastic choice of heading with an updated value. The original trace was typical under the pose prior model, whereas the modified one may be rather less likely. This plot is annotated with log of how much unlikelier, the score ratio:
# %%
key, k1, k2 = jax.random.split(key, 3)
trace = step_model.simulate(
    k1,
    (model_motion_settings, robot_inputs["start"], robot_inputs["controls"][0]),
)
rotated_trace, rotated_trace_weight_diff, _, _ = trace.update(
    k2, C["hd"].set(jnp.pi / 2)
)

(
    world_plot
    + pose_plots(trace.get_retval(), color=Plot.constantly("some pose"))
    + pose_plots(
        rotated_trace.get_retval(), color=Plot.constantly("with heading modified")
    )
    + Plot.color_map({"some pose": "green", "with heading modified": "red"})
    + Plot.title("Modifying a heading")
) | html(f"score ratio: {rotated_trace_weight_diff}")

# %% [markdown]
# It is worth carefully thinking through a trickier instance of this.  Suppose instead, within the full path, we replaced the first step's stochastic choice of heading with some specific value.
# %%
key, k1, k2 = jax.random.split(key, 3)
trace = path_model.simulate(k1, (model_motion_settings,))
rotated_first_step, rotated_first_step_weight_diff, _, _ = trace.update(
    k2, C["steps", 0, "hd"].set(jnp.pi / 2)
)

# %% [markdown]
# Note carefully how, below, the green path has perfectly overdrawn the red path, except for the first pose.  Why has modifying the first pose not modified the rest of the path?  Trace through the execution of the code in detail to understand why.

# %%
(
    world_plot
    + [
        pose_plots(pose, color=Plot.constantly("with heading modified"))
        for pose in rotated_first_step.get_retval()[1]
    ]
    + [
        pose_plots(pose, color=Plot.constantly("some path"))
        for pose in trace.get_retval()[1]
    ]
    + Plot.color_map({"some path": "green", "with heading modified": "red"})
) | html(f"score ratio: {rotated_first_step_weight_diff}")


# %% [markdown]
# ### Visualizing traces

# %% [markdown]
# In addition to the handy data structure inspection `pz.ts.display(trace)` shown above, it is important to develop, ongoing in one's work, visualization code *as a function of the trace*, since all the information is on one place here, and *in the context of the human meaning of the information*.

# %%
def get_path(trace):
    return trace.get_retval()[1]

def get_sensors(trace):
    return trace.get_choices()["steps", "sensor", "distance"]

def animate_full_trace(trace, frame_key=None):
    path = get_path(trace)
    readings = get_sensors(trace)
    return animate_path_and_sensors(
        path, readings, frame_key=frame_key
    )


# %%
key, sub_key = jax.random.split(key)
tr = full_model.simulate(sub_key, (model_motion_settings, model_sensor_noise))
animate_full_trace(tr)

# %% [markdown]
# ## The data
#
# Let us generate some fixed synthetic motion data that, for pedagogical purposes, we will work with as if it were the actual path of the robot.  We will generate two versions, one each with low or high motion deviation.
# %%
motion_settings_low_deviation = {
    "p_noise": 0.05,
    "hd_noise": (1 / 10) * degrees,
}
motion_settings_high_deviation = {
    "p_noise": 0.4,
    "hd_noise": 2 * degrees,
}

key, k_low, k_high = jax.random.split(jax.random.key(0), 3)
trace_low_deviation = full_model.simulate(k_low, (motion_settings_low_deviation, model_sensor_noise))
trace_high_deviation = full_model.simulate(k_high, (motion_settings_high_deviation, model_sensor_noise))

(
    (
        html("low motion-deviation data")
        | animate_full_trace(trace_low_deviation, frame_key="frame")
    ) & (
        html("high motion-deviation data")
        | animate_full_trace(trace_high_deviation, frame_key="frame")
    )
) | Plot.Slider("frame", 0, T, fps=2)
# %% [markdown]
# Since we imagine these data as having been recorded from the real world, keep only their extracted data, *discarding* the traces that produced them.
# %%
# These are what we hope to recover...
path_low_deviation = get_path(trace_low_deviation)
path_high_deviation = get_path(trace_high_deviation)

# ...using these data.
observations_low_deviation = get_sensors(trace_low_deviation)
observations_high_deviation = get_sensors(trace_high_deviation)

constraints_low_deviation = C["steps", "sensor", "distance"].set(observations_low_deviation)
constraints_high_deviation = C["steps", "sensor", "distance"].set(observations_high_deviation)

# %% [markdown]
# We summarize the information available to the robot to determine its location:
# * On the one hand, one has to produce a guess of the start pose plus some controls, which one might integrate to produce an idealized guess of path.
# * On the other hand, one has the sensor data, which, recall, look to it like this:


# %%
empty_world = Plot.new(
    {"margin": 0, "inset": 50, "width": 500, "axis": None, "aspectRatio": 1},
    Plot.domain(world["bounding_box"][0]),
)

(
    Plot.Frames([
        empty_world
        + plot_sensors(Pose(world["center_point"], 0.0), readings, sensor_angles, show_legend=True)
        for readings in observations_low_deviation
    ], key="frame")
    & Plot.Frames([
        empty_world
        + plot_sensors(Pose(world["center_point"], 0.0), readings, sensor_angles, show_legend=True)
        for readings in observations_high_deviation
    ], key="frame")
) | Plot.Slider("frame", 0, T, fps=2)
# %% [markdown]
# ## Inference over robot paths
#
# ### Why we need inference: in a picture
#
# The path obtained by integrating the controls serves as a proposal for the true path, but it is unsatisfactory, especially in the high motion deviation case. The picture gives an intuitive sense of the (poor) fit:
# %%
(
    Plot.Frames([
        world_plot
        + plot_sensors(pose, readings, sensor_angles, show_legend=True)
        for (pose, readings) in zip(path_integrated, observations_low_deviation)
    ], key="frame")
    & Plot.Frames([
        world_plot
        + plot_sensors(pose, readings, sensor_angles, show_legend=True)
        for (pose, readings) in zip(path_integrated, observations_high_deviation)
    ], key="frame")
) | Plot.Slider("frame", 0, T, fps=2)
# %% [markdown]
# It would seem that the fit is reasonable in low motion deviation, but really breaks down in high motion deviation.
#
# We are not limited to visual judgments here: the model can quantitatively assess how good a fit the integrated path is for the data.  In order to do this, we detour to explain how to produce samples from our model that agree with the fixed observation data.

# %% [markdown]
# ### Producing samples with constraints
#
# We have seen how `simulate` performs traced execution of a generative function: as the program runs, it draws stochastic choices from all required primitive distributions, and records them in a choice map.
#
# Given a choice map of *constraints* that declare fixed values of some of the primitive choices, the operation `importance` proposes traces of the generative function that are consistent with these constraints.

# %%
key, k1, k2 = jax.random.split(key, 3)
trace_low, log_weight_low = full_model.importance(
    k1, constraints_low_deviation, (model_motion_settings, model_sensor_noise)
)
trace_high, log_weight_high = full_model.importance(
    k2, constraints_high_deviation, (model_motion_settings, model_sensor_noise)
)

(
    (
        html("fresh path sample", "fixed low motion-deviation sensor data")
        | animate_full_trace(trace_low, frame_key="frame")
        | html(f"log_weight: {log_weight_low}")
    ) & (
        html("fresh path sample", "fixed high motion-deviation sensor data")
        | animate_full_trace(trace_high, frame_key="frame")
        | html(f"log_weight: {log_weight_high}")
    )
) | Plot.Slider("frame", 0, T, fps=2)

# %% [markdown]
# A trace resulting from a call to `importance` is structurally indistinguishable from one drawn from `simulate`.  But there is a key *situational* difference: while `get_score` always returns the frequency with which `simulate` stochastically produces the trace, this value is **no longer equal to** the frequency with which the trace is stochastically produced by `importance`.  This is both true in an obvious and less relevant sense, as well as true in a more subtle and extremely germane sense.
#
# On the superficial level, since all traces produced by `importance` are consistent with the constraints, those traces that are inconsistent with the constraints do not occur at all, and in balance the traces that are consistent with the constraints are more common.
#
# More deeply and importantly, the stochastic choice of the *constraints* under a run of `simulate` might have any density, perhaps very low.  This constraints density contributes as always to the `get_score`, whereas it does not influence the frequency of producing this trace under `importance`.
#
# The ratio of the `get_score` of a trace to the probability density that `importance` would produce it with the given constraints, is called the *importance weight*.  For convenience, (the log of) this quantity is returned by `importance` along with the trace.
#
# We stress the basic invariant:
# $$
# (\text{frequency simulate creates this trace})
# =
# \text{get\_score}(\text{trace})
# =
# (\text{weight from importance})
# \cdot
# (\text{frequency importance creates this trace}).
# $$
# %% [markdown]
# The preceding comments apply to generative functions in wide generality.  We can say even more about our present examples, because further assumptions hold.
# 1. There is no untraced randomness.  Given a *full choice map* for constraints, `importance` is rendered deterministic.  In particular, the importance weight is the `get_score`.
# 2. The generative function was constructed using GenJAX's DSL and primitive distributions.  Under ancestral sampling, `importance` with *empty constraints* reduces to `simulate` plus importance weight $1$.
# 3. Combined, the importance weight is directly computed as the `project` of the trace upon the choice map addresses that were constrained in the call to `importance`.
#
# Thus in our running example, the projection in question is $\prod_{t=0}^T P_\text{sensor}(o_t)$.
# %%
key, sub_key = jax.random.split(key)
log_weight_high - trace_high.project(sub_key, S["steps", "sensor", "distance"])


# %% [markdown]
# ### Why we need inference: in numbers
#
# We return to how the model offers a numerical benchmark for how good a fit the integrated path is.
#
# In words, the integrated path has incongruously low likelihood for the data.  The (log) density of the measurement data, given the integrated path...

# %%
def constraint_from_path(path):
    c_ps = jax.vmap(lambda ix, p: C["steps", ix, "pose", "p"].set(p))(
        jnp.arange(T), path.p
    )
    c_hds = jax.vmap(lambda ix, hd: C["steps", ix, "pose", "hd"].set(hd))(
        jnp.arange(T), path.hd
    )
    return c_ps | c_hds

constraints_path_integrated = constraint_from_path(path_integrated)

key, k1, k2 = jax.random.split(key, 3)
trace_path_integrated_observations_low_deviation, log_weight_low = full_model.importance(
    k1,
    constraints_path_integrated | constraints_low_deviation,
    (model_motion_settings, model_sensor_noise),
)
trace_path_integrated_observations_high_deviation, log_weight_high = full_model.importance(
    k2,
    constraints_path_integrated | constraints_high_deviation,
    (model_motion_settings, model_sensor_noise),
)

(
    (
        html("integrated path from controls", "fixed low motion-deviation sensor data")
        | animate_full_trace(trace_path_integrated_observations_low_deviation, frame_key="frame")
        | html(f"log_weight: {log_weight_low}")
    ) & (
        html("integrated path from controls", "fixed high motion-deviation sensor data")
        | animate_full_trace(trace_path_integrated_observations_high_deviation, frame_key="frame")
        | html(f"log_weight: {log_weight_high}")
    )
) | Plot.Slider("frame", 0, T, fps=2)
# %% [markdown]
# (the `log_weight` reported just above this cell) ...more closely resembles the densities of these data back-fitted onto typical (random) paths of the model, than the (log) densities of data typically produced by the full model run in its natural manner:


# %%
N_samples = 200

key, k1, k2, k3, k4 = jax.random.split(key, 5)

traces_generated_low_deviation, low_weights = jax.vmap(
    full_model.importance, in_axes=(0, None, None)
)(
    jax.random.split(k1, N_samples),
    constraints_low_deviation,
    (model_motion_settings, model_sensor_noise),
)
traces_generated_high_deviation, high_weights = jax.vmap(
    full_model.importance, in_axes=(0, None, None)
)(
    jax.random.split(k2, N_samples),
    constraints_high_deviation,
    (model_motion_settings, model_sensor_noise),
)

traces_simulated = jax.vmap(
    full_model.simulate, in_axes=(0, None)
)(
    jax.random.split(k3, N_samples),
    (model_motion_settings, model_sensor_noise),
)
simulated_weights = jax.vmap(
    lambda trace, k: trace.project(k, S["steps", "sensor", "distance"])
)(traces_simulated, jax.random.split(k4, N_samples))

(
    html("likelihoods of low motion-deviation data")
    | Plot.histogram(values=low_weights, thresholds=10)
) & (
    html("likelihoods of high motion-deviation data")
    | Plot.histogram(values=high_weights, thresholds=10)
) & (
    html("likelihoods of random data")
    | Plot.histogram(values=simulated_weights, thresholds=10)
)


# %% [markdown]
# ## Generic strategies for inference
#
# We now spell out some generic strategies for conditioning the ouputs of a model towards some observed data.  The word "generic" indicates that they make no special intelligent use of the model structure, and their convergence is guaranteed by theorems of a similar nature.  In terms to be defined shortly, they simply take a pair $(Q,f)$ of a proposal and a weight function that implement importance sampling with target $P$.
#
# There is no free lunch in this game: generic inference recipies are inefficient, for example, converging very slowly or needing vast counts of particles, especially in high-dimensional settings.  One of the root problems is that proposals $Q$ may provide arbitrarily bad samples relative to our target $P$; if $Q$ still supports all samples of $P$ with microscopic but nonzero density, then the generic algorithm will converge in the limit, however astronomically slowly.
#
# Rather, efficiency will become possible when we do the *opposite* of generic: exploit what we actually know about the problem in our design of the inference strategy to propose better traces towards our target.  Gen's aim is to provide the right entry points to enact this exploitation.

# %% [markdown]
# ### The posterior distribution and importance sampling
#
# Mathematically, the passage from the prior to the posterior is the operation of conditioning distributions.
#
# Intuitively, the conditional distribution $\text{full}(\cdot | o_{0:T})$ is just the restriction of the joint distribution $\text{full}(z_{0:T}, o_{0:T})$ to where the parameter $o_{0:T}$ is constant, letting $z_{0:T}$ continue to vary.  This restriction no longer has total density equal to $1$, so we must renormalize it.  The normalizing constant must be
# $$
# P_\text{marginal}(o_{0:T})
# := \int P_\text{full}(Z_{0:T}, o_{0:T}) \, dZ_{0:T}
#  = \mathbf{E}_{Z_{0:T} \sim \text{path}}\big[P_\text{full}(Z_{0:T}, o_{0:T})\big].
# $$
# By Fubini's Theorem, this function of $o_{0:T}$ is the density of a probability distribution over observations $o_{0:T}$, called the *marginal distribution*; but we will often have $o_{0:T}$ fixed, and consider it a constant.  Then, finally, the *conditional distribution* $\text{full}(\cdot | o_{0:T})$ is defined to have the normalized density
# $$
# P_\text{full}(z_{0:T} | o_{0:T}) := \frac{P_\text{full}(z_{0:T}, o_{0:T})}{P_\text{marginal}(o_{0:T})}.
# $$
#
# The goal of inference is to produce samples $\text{trace}_{0:T}$ distributed (approximately) according to $\text{full}(\cdot | o_{0:T})$.  The most immediately evident problem with doing inference is that the quantity $P_\text{marginal}(o_{0:T})$ is intractable!

# %% [markdown]
# Define the function $\hat f(z_{0:T})$ of sample values $z_{0:T}$ to be the ratio of probability densities between the posterior distribution $\text{full}(\cdot | o_{0:T})$ that we wish to sample from, and the prior distribution $\text{path}$ that we are presently able to produce samples from.  Manipulating it à la Bayes's Rule gives:
# $$
# \hat f(z_{0:T})
# :=
# \frac{P_\text{full}(z_{0:T} | o_{0:T})}{P_\text{path}(z_{0:T})}
# =
# \frac{P_\text{full}(z_{0:T}, o_{0:T})}{P_\text{marginal}(o_{0:T}) \cdot P_\text{path}(z_{0:T})}
# =
# \frac{\prod_{t=0}^T P_\text{sensor}(o_t)}{P_\text{marginal}(o_{0:T})}.
# $$
# Noting that the intractable quantity
# $$
# Z := P_\text{marginal}(o_{0:T})
# $$
# is constant in $z_{0:T}$, we define the explicitly computable quantity
# $$
# f(z_{0:T}) := Z \cdot \hat f(z_{0:T}) = \prod\nolimits_{t=0}^T P_\text{sensor}(o_t).
# $$
# The right hand side has been written sloppily, but we remind the reader that $P_\text{sensor}(o_t)$ is a product of densities of normal distributions that *does depend* on $z_t$ as well as "sensor" and "world" parameters.
#
# Compare to our previous description of calling `importance` on `full_model` with the observations $o_{0:T}$ as constraints: it produces a trace of the form $(z_{0:T}, o_{0:T})$ where $z_{0:T} \sim \text{path}$ has been drawn from $\text{path}$, together with the weight equal to none other than this $f(z_{0:T})$.

# %% [markdown]
# This reasoning involving `importance` is indicative of the general scenario with conditioning, and fits into the following shape.
#
# We have on hand two distributions, a *target* $P$ from which we would like to (approximately) generate samples, and a *proposal* $Q$ from which we are presently able to generate samples.  We must assume that the proposal is a suitable substitute for the target, in the sense that every possible event under $P$ occurs under $Q$ (mathematically, $P$ is absolutely continuous with respect to $Q$).
#
# Under these hypotheses, there is a well-defined density ratio function $\hat f$ between $P$ and $Q$ (mathematically, the Radon–Nikodym derivative).  If $z$ is a sample drawn from $Q$, then $\hat w = \hat f(z)$ is how much more or less likely $z$ would have been drawn from $P$.  We only require that we are able to compute the *unnormalized* density ratio, that is, some function of the form $f = Z \cdot \hat f$ where $Z > 0$ is constant.
#
# The pair $(Q,f)$ is said to implement *importance sampling* for $P$, and the values of $f$ are called *importance weights*.  Generic inference attempts to use knowledge of $f$ to correct for the difference in behavior between $P$ and $Q$, and thereby use $Q$ to produce samples from (approximately) $P$.
#
# So in our running example, the target $P$ is the posterior distribution on paths $\text{full}(\cdot | o_{0:T})$, the proposal $Q$ is the path prior $\text{path}$, and the importance weight $f$ is the product of the sensor model densities.  We seek a computational model of the first; the second and third are computationally modeled by calling `importance` on `full_model` constrained by the observations $o_{0:T}$.  (The computation of the second, on its own, simplifies to `path_model`.)
#

# %% [markdown]
# ### Sampling / importance resampling
#
# We turn to inference strategies that require only our proposal $Q$ and unnormalized weight function $f$ for the target $P$, *without* forcing us to wrangle any intractable integrals or upper bounds.
#
# Suppose we are given a list of nonnegative numbers, not all zero: $w^1, w^2, \ldots, w^N$.  To *normalize* the numbers means computing $\hat w^i := w^i / \sum_{j=1}^N w^j$.  The normalized list $\hat w^1, \hat w^2, \ldots, \hat w^N$ determines a *categorical distribution* on the indices $1, \ldots, N$, wherein the index $i$ occurs with probability $\hat w^i$.
# Note that for any constant $Z > 0$, the scaled list $Zw^1, Zw^2, \ldots, Zw^N$ leads to the same normalized $\hat w^i$ as well as the same categorical distribution.
#
# When some list of data $z^1, z^2, \ldots, z^N$ have been associated with these respective numbers $w^1, w^2, \ldots, w^N$, then to *importance **re**sample* $M$ values from these data according to these weights means to independently sample indices $a^1, a^2, \ldots, a^M \sim \text{categorical}([\hat w^1, \hat w^2, \ldots, \hat w^N])$ and return the new list of data $z^{a^1}, z^{a^2}, \ldots, z^{a^M}$.  Compare to the function `resample` implemented in the code box below.
#
# The *sampling / importance resampling* (SIR) strategy for inference runs as follows.  Let counts $N > 0$ and $M > 0$ be given.
# 1. Importance sample:  Independently sample $N$ data $z^1, z^2, \ldots, z^N$ from the proposal $Q$, called *particles*.  Compute also their *importance weights* $w^i := f(z^i)$ for $i = 1, \ldots, N$.
# 2. Importance resample:  Independently sample $M$ indices $a^1, a^2, \ldots, a^M \sim \text{categorical}([\hat w^1, \hat w^2, \ldots, \hat w^N])$, where $\hat w^i = w^i / \sum_{j=1}^N w^j$, and return $z^{a^1}, z^{a^2}, \ldots, z^{a^M}$. These sampled particles all inherit the *average weight* $\sum_{j=1}^N w^j / N$.
#
# As $N \to \infty$ with $M$ fixed, the samples produced by this algorithm converge to $M$ independent samples drawn from the target $P$.  This strategy is computationally an improve
# ment over rejection sampling: intead of indefinitely constructing and rejecting samples, we can guarantee to use at least some of them after a fixed time, and we are using the best guesses among these.

# %%
def importance_resample_unjitted(
    key: PRNGKey, constraints: genjax.ChoiceMap, motion_settings, sensor_noise, N: int, K: int
):
    """Produce K importance samples of depth N from the model. That is, K times, we
    generate N importance samples conditioned by the constraints, and categorically
    select one of them."""
    key1, key2 = jax.random.split(key)
    samples, log_weights = jax.vmap(full_model.importance, in_axes=(0, None, None))(
        jax.random.split(key1, N * K), constraints, (motion_settings, sensor_noise)
    )
    winners = jax.vmap(genjax.categorical.propose)(
        jax.random.split(key2, K), (jnp.reshape(log_weights, (K, N)),)
    )[2]
    # indices returned are relative to the start of the K-segment from which they were drawn.
    # globalize the indices by adding back the index of the start of each segment.
    winners += jnp.arange(0, N * K, N)
    selected = jax.tree.map(lambda x: x[winners], samples)
    return selected

importance_resample = jax.jit(importance_resample_unjitted, static_argnums=(4, 5))


def pytree_transpose(list_of_pytrees):
  """
  Converts a list of pytrees of identical structure into a single pytree of lists.
  """
  return jax.tree.map(lambda *xs: jnp.array(list(xs)), *list_of_pytrees)

def plot_inference_result(title, samples_label, posterior_paths, target_path, history_paths=None):
    return (
        html(*title)
        | (
            world_plot
            + (
                [
                    Plot.line(
                        {"x": path.p[:, 0], "y": path.p[:, 1]},
                        curve="linear",
                        opacity=0.05,
                        strokeWidth=2,
                        stroke="red"
                    )
                    for path in history_paths
                ] if history_paths else []
            )
            + [
                Plot.line(
                    {"x": path.p[:, 0], "y": path.p[:, 1]},
                    curve="linear",
                    opacity=0.2,
                    strokeWidth=2,
                    stroke="green"
                )
                for path in posterior_paths
            ]
            + pose_plots(
                target_path, fill=Plot.constantly("path to be inferred"), opacity = 0.5, strokeWidth=2
            )
            + Plot.color_map({
                samples_label: "green",
                "path to be inferred": "black",
            } | (
               {"culled paths": "red"} if history_paths else {}
            ))
        )
    )


# %%
N_presamples = 2000
N_samples = 20

key, k1, k2 = jax.random.split(key, 3)
low_posterior = importance_resample(
    k1, constraints_low_deviation, model_motion_settings, model_sensor_noise, N_presamples, N_samples
)
high_posterior = importance_resample(
    k2, constraints_high_deviation, model_motion_settings, model_sensor_noise, N_presamples, N_samples
)

plot_inference_result(
    ("importance resampling on low motion-deviation data",),
    "importance resamples",
    jax.vmap(get_path)(low_posterior),
    path_low_deviation
) & plot_inference_result(
    ("importance resampling on high motion-deviation data",),
    "importance resamples",
    jax.vmap(get_path)(high_posterior),
    path_high_deviation
)
# %% [markdown]
# Let's pause a moment to examine this chart. If the robot had no sensors, it would have no alternative but to estimate its position by integrating the control inputs to produce the integrated path in gray. In the low deviation setting, *the sensor data combined with importance sampling is enough* to give accurate results in the low deviation setting.
# But in the high deviation setting, the loose nature of the paths in the blue posterior indicate that the robot has not discovered its true position by using importance sampling with the noisy sensor data. In the high deviation setting, more refined inference technique will be required.
#
# Let's approach the problem step by step instead of trying to infer the whole path at once.
# The technique we will use is called Sequential Importance Sampling or a
# [Particle Filter](https://en.wikipedia.org/wiki/Particle_filter). It works like this.
#
# When we designed the step model for the robot, we arranged things so that the model
# could be used with `scan`: the model takes a *state* and a *control input* to produce
# a new *state*. Imagine at some time step $t$ that we use importance sampling with this
# model at a pose $\mathbf{z}_t$ and control input $\mathbf{u}_t$, scored with respect to the
# sensor observations $\mathbf{y}_t$ observed at that time. We will get a weighted collection
# of possible updated poses $\mathbf{z}_t^N$ and weights $w^N$.
#
# The particle filter "winnows" this set by replacing it with $N$ weighted selections
# *with replacement* from this collection. This may select better candidates several
# times, and is likely to drop poor candidates from the collection. We can arrange to
# do this at each time step with a little preparation: we start by "cloning" our idea
# of the robot's initial position into an N vector and this becomes the initial particle
# collection. At each step, we generate an importance sample and winnow it.
#
# This can also be done as a scan. Our previous attempt used `scan` to produce candidate
# paths from start to end, and these were scored for importance using all of the sensor
# readings at once. The results were better than guesses, but not accurate, in the
# high deviation setting.
#
# The technique we will use here discards steps with low posterior density at each step, and
# reinforces steps with high posterior density, allowing better particles to proportionately
# search more of the probability space while discarding unpromising particles.
#
# The following class attempts to generatlize this idea:

# %%
StateT = TypeVar("StateT")
ControlT = TypeVar("ControlT")

class SISwithRejuvenation(Generic[StateT, ControlT]):
    """
    Given:
     - a functional wrapper for the importance method of a generative function
     - an initial state of type StateT, which should be a PyTree $z_0$
     - a vector of control inputs, also a PyTree $u_i, of shape $(T, \\ldots)$
     - an array of observations $y_i$, also of shape $(T, \\ldots)$
    perform the inference technique known as Sequential Importance Sampling.

    The signature of the GFI importance method is
        key -> constraint -> args -> (trace, weight)
    For importance sampling, this is vmapped over key to get
        [keys] -> constraint -> args -> ([trace], [weight])
    The functional wrapper's purpose is to maneuver the state and control
    inputs into whatever argument shape the underlying model is expecting,
    and to turn the observation at step $t$ into a choicemap asserting
    that constraint.

    You may also supply an SMCP3 rejuvenation function, whose signature is
        key -> Trace[StateT] ->

    After the object is constructed, SIS can be performed at any importance
    depth with the `run` method, which will perform the following steps:

     - inflate the initial value to a vector of size N of identical initial
       values
     - vmap over N keys generated from the supplied key
     - each vmap cell will scan over the control inputs and observations

    Between each step, categorical sampling with replacement is formed to
    create a particle filter. Favorable importance draws are likely to
    be replicated, and unfavorable ones discarded. The resampled vector of
    states is sent the the next step, while the values drawn from the
    importance sample and the indices chosen are emitted from the scan step,
    where, at the end of the process, they will be available as matrices
    of shape (N, T).
    """


    def __init__(
        self,
        init: StateT,
        controls: ControlT,
        observations: Array,
        importance: Callable[
            [PRNGKey, StateT, ControlT, Array], tuple[genjax.Trace[StateT], float]
        ],
        rejuvenate: Callable[
            [PRNGKey, genjax.Trace[StateT], Array, StateT, ControlT], tuple[genjax.Trace[StateT], float]
        ] | None = None,
    ):
        self.importance = jax.jit(importance)
        self.rejuvenate = jax.jit(rejuvenate) if rejuvenate else None
        self.init = init
        self.controls = controls
        self.observations = observations


    class Result(Generic[StateT]):
        """This object contains all of the information generated by the SIS scan,
        and offers some convenient methods to reconstruct the paths explored
        (`flood_fill`) or ultimately chosen (`backtrack`).
        """

        def __init__(
            self, end: StateT, samples: genjax.Trace[StateT], indices: IntArray, rejuvenated: genjax.Trace[StateT]
        ):
            self.end = end
            self.samples = samples.get_retval()
            self.indices = indices
            self.rejuvenated = rejuvenated.get_retval()
            self.N = len(end)
            self.T = len(self.rejuvenated)

        def flood_fill(self) -> list[list[StateT]]:
            complete_paths = []
            active_paths = self.N * [[]]
            for i in range(self.T):
                new_active_paths = self.N * [None]
                for (j, count) in enumerate(jnp.bincount(self.indices[i], length=self.N)):
                    if count == 0:
                        complete_paths.append(active_paths[j] + [self.samples[i][j]])
                    new_active_paths[j] = active_paths[self.indices[i][j]] + [self.rejuvenated[i][j]]
                active_paths = new_active_paths
            return complete_paths + active_paths

        def backtrack(self) -> list[list[StateT]]:
            paths = [[p] for p in self.end]
            for i in reversed(range(self.T - 1)):
                for j in range(self.N):
                    paths[j].insert(0, self.rejuvenated[i][self.indices[i + 1][j]])
            return paths


    def run(self, key: PRNGKey, N: int) -> dict:
        def step(state, update):
            particles, log_weights = state
            key, control, observation = update
            ks = jax.random.split(key, (3, N))
            samples, log_weight_increments = jax.vmap(self.importance, in_axes=(0, 0, None, None))(
                ks[0], particles, control, observation
            )
            indices = jax.vmap(genjax.categorical.propose, in_axes=(0, None))(
                ks[1], (log_weights + log_weight_increments,)
            )[2]
            (resamples, antecedents) = jax.tree.map(lambda v: v[indices], (samples, particles))
            if self.rejuvenate:
                rejuvenated, new_log_weights = jax.vmap(self.rejuvenate, in_axes=(0, 0, 0, None, None))(
                    ks[2],
                    resamples,
                    antecedents,
                    control,
                    observation
                )
            else:
                rejuvenated, new_log_weights = resamples, jnp.zeros(log_weights.shape)
            return (rejuvenated.get_retval(), new_log_weights), (samples, indices, rejuvenated)

        init_array = jax.tree.map(
            lambda a: jnp.broadcast_to(a, (N,) + a.shape), self.init
        )
        (end, _), (samples, indices, rejuvenated) = jax.lax.scan(
            step,
            (init_array, jnp.zeros(N)),
            (
                jax.random.split(key, len(self.controls)),
                self.controls,
                self.observations,
            ),
        )
        return SISwithRejuvenation.Result(end, samples, indices, rejuvenated)

# %%
def localization_sis(motion_settings, sensor_noise, observations):
    return SISwithRejuvenation(
        robot_inputs["start"],
        robot_inputs["controls"],
        observations,
        lambda key, pose, control, observation: full_model_kernel.importance(
            key,
            C["sensor", "distance"].set(observation),
            (motion_settings, sensor_noise, pose, control),
        ),
    )


# %% [markdown]
# Below, the paths that died in the sequential resampling are shown in red.

# %%
key, k1, k2 = jax.random.split(key, 3)

N_particles = 20
sis_result_low = localization_sis(
    model_motion_settings, model_sensor_noise, observations_low_deviation
).run(k1, N_particles)
sis_result_high = localization_sis(
    model_motion_settings, model_sensor_noise, observations_high_deviation
).run(k2, N_particles)

plot_inference_result(
    ("SIS on low motion-deviation data",),
    "sequential importance resamples",
    [pytree_transpose(path) for path in sis_result_low.backtrack()],
    path_low_deviation,
    history_paths=[pytree_transpose(path) for path in sis_result_low.flood_fill()]
) & plot_inference_result(
    ("SIS on high motion-deviation data",),
    "sequential importance resamples",
    [pytree_transpose(path) for path in sis_result_high.backtrack()],
    path_high_deviation,
    history_paths=[pytree_transpose(path) for path in sis_result_high.flood_fill()]
)
# %% [markdown]
# As already noted, after winnowing poor quality particles and replacing them with copies of the better ones, particle diversity suffers.  The technique of *rejuvenation* is to perturb the results of resampling to restore this diversity.  At the same time, it can perform operations that improve the sample quality.  In this case, we search a small grid of points near the most recent step for improvement in the score.
#
# While we perturb the particles, we update the weight using the *SMCP3* rule.

# %%
# This is the general SMCP3 algorithm in the case where there is no Jacobian term.
def run_SMCP3_step(fwd_proposal, bwd_proposal, key, sample, proposal_args):
    k1, k2 = jax.random.split(key, 2)

    _, fwd_proposal_weight, (fwd_update, bwd_choices) = fwd_proposal.propose(k1, (sample, proposal_args))

    new_sample, model_weight_diff, _, _ = sample.update(k2, fwd_update)

    bwd_proposal_weight, _ = bwd_proposal.assess(bwd_choices, (new_sample, proposal_args))

    new_log_weight = model_weight_diff + bwd_proposal_weight - fwd_proposal_weight
    return new_sample, new_log_weight

# Forward proposal searches a nearby grid around the sample,
# and returns an importance-resampled member.
# The joint density (= the density from the full model) serves as
# the unnormalized posterior density over steps.
@genjax.gen
def grid_fwd_proposal(sample, args):
    base_grid, observation, full_model_args = args
    observation_cm = C["sensor", "distance"].set(observation)

    log_weights = jax.vmap(
        lambda p, hd:
            full_model_kernel.assess(
                observation_cm
                | C["pose", "p"].set(p + sample.get_retval().p)
                | C["pose", "hd"].set(hd + sample.get_retval().hd),
                full_model_args
            )[0]
    )(*base_grid)
    fwd_index = genjax.categorical(log_weights) @ "fwd_index"

    return (
        (
            C["pose", "p"].set(base_grid[0][fwd_index] + sample.get_retval().p)
            | C["pose", "hd"].set(base_grid[1][fwd_index] + sample.get_retval().hd)
        ),
        C["bwd_index"].set(len(log_weights) - 1 - fwd_index)
    )

# Backwards proposal simply guesses according to the prior over steps, nothing fancier.
@genjax.gen
def grid_bwd_proposal(new_sample, args):
    base_grid, _, full_model_args = args
    step_model_args = (full_model_args[0], full_model_args[2], full_model_args[3])

    log_weights = jax.vmap(
        lambda p, hd:
            step_model.assess(
                C["p"].set(p + new_sample.get_retval().p)
                | C["hd"].set(hd + new_sample.get_retval().hd),
                step_model_args
            )[0]
    )(*base_grid)

    _ = genjax.categorical(log_weights) @ "bwd_index"
    # Since the backward proposal is only used for assessing the above choice,
    # no further computation is necessary.


# %%
def localization_sis_plus_grid_rejuv(motion_settings, sensor_noise, M_grid, N_grid, observations):
    base_grid = make_poses_grid_array(
        jnp.array([-M_grid / 2, M_grid / 2]).T,
        N_grid
    )
    return SISwithRejuvenation(
        robot_inputs["start"],
        robot_inputs["controls"],
        observations,
        importance=lambda key, pose, control, observation: full_model_kernel.importance(
            key,
            C["sensor", "distance"].set(observation),
            (motion_settings, sensor_noise, pose, control),
        ),
        rejuvenate=lambda key, sample, pose, control, observation: run_SMCP3_step(
            grid_fwd_proposal,
            grid_bwd_proposal,
            key,
            sample,
            (base_grid, observation, (motion_settings, sensor_noise, pose, control))
        ),
    )


# %% [markdown]
# The following plots compare the performance of this local grid search rejuvenation scheme (first) versus vanilla sequential importance sampling (second).

# %%
N_particles = 20
M_grid = jnp.array([1.0, 1.0, (3 / 10) * degrees])
N_grid = jnp.array([15, 15, 15])

key, k1, k2 = jax.random.split(key, 3)
sis_result = localization_sis(
    model_motion_settings, model_sensor_noise, observations_high_deviation
).run(k1, N_particles)
smcp3_result = localization_sis_plus_grid_rejuv(
    model_motion_settings, model_sensor_noise, M_grid, N_grid, observations_high_deviation
).run(k2, N_particles)

plot_inference_result(
    ("SIS without rejuvenation", "high motion-deviation data"),
    "samples",
    [pytree_transpose(path) for path in sis_result.backtrack()],
    path_high_deviation,
    history_paths=[pytree_transpose(path) for path in sis_result.flood_fill()]
) & plot_inference_result(
    ("SIS with SMCP3 grid rejuvenation", "high motion-deviation data"),
    "samples",
    [pytree_transpose(path) for path in smcp3_result.backtrack()],
    path_high_deviation,
    history_paths=[pytree_transpose(path) for path in smcp3_result.flood_fill()]
)
