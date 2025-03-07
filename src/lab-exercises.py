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
# # Exercises on the Localization Tutorial
#
# ## Setup
#
# Here are two large notebook cells.  The first cell declares all the reusable components from the tutorial (modeling and inference gadgets, and plotting abstractions), making them globally available while performing no significant computation.  The second cell gathers the main visualization recipes in a series of commented blocks; each block can be run independently.
#
# After running the first of these, skip below to the following sections.
# %%
# Global includes

import json
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import genjax
from genjax import ChoiceMapBuilder as C
from genjax.typing import Array, FloatArray, PRNGKey, IntArray
from penzai import pz
from typing import TypeVar, Generic, Callable
from genstudio.plot import js

html = Plot.Hiccup


# Map data

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

world = load_world("world.json")

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


# Poses

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

def random_pose(k):
    p_array = jax.random.uniform(k, shape=(3,),
        minval=world["bounding_box"][:, 0],
        maxval=world["bounding_box"][:, 1])
    return Pose(p_array[0:2], p_array[2])

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


# Ideal sensors

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

sensor_settings = {
    "fov": 2 * jnp.pi * (2 / 3),
    "num_angles": 41,
    "box_size": world["box_size"],
}

def sensor_distance(pose, walls, box_size):
    d = jnp.min(jax.vmap(distance, in_axes=(None, 0))(pose, walls))
    # Capping to a finite value avoids issues below.
    return jnp.where(jnp.isinf(d), 2 * box_size, d)

def make_sensor_angles(sensor_settings):
    na = sensor_settings["num_angles"]
    return sensor_settings["fov"] * (jnp.arange(na) - ((na - 1) / 2)) / (na - 1)

sensor_angles = make_sensor_angles(sensor_settings)

def ideal_sensor(pose):
    return jax.vmap(
        lambda angle: sensor_distance(pose.rotate(angle), world["walls"], sensor_settings["box_size"])
    )(sensor_angles)

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

def pose_at(state, label):
    pose_dict = getattr(state, label)
    return Pose(jnp.array(pose_dict["p"]), jnp.array(pose_dict["hd"]))

def update_ideal_sensors(widget, label):
    widget.state.update({
        (label + "_readings"): ideal_sensor(pose_at(widget.state, label))
    })


# Noisy sensors

sensor_settings["s_noise"] = 0.1

@genjax.gen
def sensor_model_one(pose, angle, s_noise):
    return (
        genjax.normal(
            sensor_distance(pose.rotate(angle), world["walls"], sensor_settings["box_size"]),
            s_noise,
        )
        @ "distance"
    )

sensor_model = sensor_model_one.vmap(in_axes=(None, 0, None))

def noisy_sensor(key, pose, s_noise):
    return sensor_model.propose(key, (pose, sensor_angles, s_noise))[2]

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


# Pose priors

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


# Joint model

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
        sensor = sensor_model(pose, sensor_angles, model_noise) @ "sensor"  # noqa: F841
    return jax.jit(
        lambda pose:
            joint_model.assess(
                C["pose"].set(cm_builder(pose)) | C["sensor", "distance"].set(readings),
                ()
            )[0]
    )


# "Camera widget" code

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
        | noise_slider("world_noise", "World/data noise = ", sensor_settings["s_noise"])
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
        | noise_slider("model_noise", "Model/inference noise = ", sensor_settings["s_noise"])
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


# Grid utils

def make_grid(bounds, ns):
    return [dim.reshape(-1) for dim in jnp.meshgrid(*(jnp.linspace(*bound, num=n) for (bound, n) in zip(bounds, ns)))]

def make_poses_grid_array(bounds, ns):
    grid_xs, grid_ys, grid_hds = make_grid(bounds, ns)
    return jnp.array([grid_xs, grid_ys]).T, grid_hds

def make_poses_grid(bounds, ns):
    return Pose(*make_poses_grid_array(bounds, ns))


# Robot programs

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

world["bounce"] = 0.1
robot_inputs, T = load_robot_program("robot_program.json")


# Integrating controls / applying robot programs

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

path_integrated = integrate_controls_physical(robot_inputs)


# Step and path models

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

degrees = jnp.pi / 180
default_motion_settings = {
    "p_noise": 0.15,
    "hd_noise": 1 * degrees
}

@genjax.gen
def path_model(motion_settings):
    return (
        step_model
        .partial_apply(motion_settings)
        .map(diag)
        .scan()(robot_inputs["start"], robot_inputs["controls"])
        @ "steps"
    )

def confidence_circle(p, p_noise):
    return Plot.ellipse(
        p,
        r=2.5 * p_noise,
        fill=Plot.constantly("95% confidence region"),
    ) + Plot.color_map({"95% confidence region": "rgba(255,0,0,0.25)"})

def plot_path_with_confidence(path, step):
    prev_step = robot_inputs["start"] if step == 0 else path[step - 1]
    return (
        world_plot
        + confidence_circle(
            [prev_step.apply_control(robot_inputs["controls"][step]).p],
            default_motion_settings["p_noise"]
        )
        + [pose_plots(path[i]) for i in range(step)]
        + pose_plots(path[step], color=Plot.constantly("next pose"))
        + Plot.color_map({"previous poses": "black", "next pose": "green"})
    )

def animate_path_and_sensors(path, readings, motion_settings, frame_key=None):
    return Plot.Frames([
        plot_path_with_confidence(path, step)
        + plot_sensors(pose, readings[step], sensor_angles)
        for step, pose in enumerate(path)
    ], fps=2, key=frame_key)


# Full model

@genjax.gen
def full_model_kernel(motion_settings, s_noise, state, control):
    pose = step_model(motion_settings, state, control) @ "pose"
    sensor_model(pose, sensor_angles, s_noise) @ "sensor"
    return pose

@genjax.gen
def full_model(motion_settings, s_noise):
    return (
        full_model_kernel
        .partial_apply(motion_settings)
        .partial_apply(s_noise)
        .map(diag)
        .scan()(robot_inputs["start"], robot_inputs["controls"])
        @ "steps"
    )


# THE DATA

motion_settings_low_deviation = {
    "p_noise": 0.05,
    "hd_noise": (1 / 10) * degrees,
}
motion_settings_high_deviation = {
    "p_noise": 0.5,
    "hd_noise": 3 * degrees,
}

key = jax.random.key(0)
key, k_low, k_high = jax.random.split(key, 3)
trace_low_deviation = full_model.simulate(k_low, (motion_settings_low_deviation, sensor_settings["s_noise"]))
trace_high_deviation = full_model.simulate(k_high, (motion_settings_high_deviation, sensor_settings["s_noise"]))

def get_path(trace):
    return trace.get_retval()[1]

def get_sensors(trace):
    return trace.get_choices()["steps", "sensor", "distance"]

def animate_full_trace(trace, frame_key=None):
    path = get_path(trace)
    readings = get_sensors(trace)
    motion_settings = trace.get_args()[0]
    return animate_path_and_sensors(
        path, readings, motion_settings, frame_key=frame_key
    )

def constraint_from_path(path):
    c_ps = jax.vmap(lambda ix, p: C["steps", ix, "pose", "p"].set(p))(
        jnp.arange(T), path.p
    )
    c_hds = jax.vmap(lambda ix, hd: C["steps", ix, "pose", "hd"].set(hd))(
        jnp.arange(T), path.hd
    )
    return c_ps | c_hds

path_low_deviation = get_path(trace_low_deviation)
path_high_deviation = get_path(trace_high_deviation)
observations_low_deviation = get_sensors(trace_low_deviation)
observations_high_deviation = get_sensors(trace_high_deviation)
constraints_low_deviation = C["steps", "sensor", "distance"].set(observations_low_deviation)
constraints_high_deviation = C["steps", "sensor", "distance"].set(observations_high_deviation)


# Whole-path importance resampling

def importance_resample_unjitted(
    key: PRNGKey, constraints: genjax.ChoiceMap, motion_settings, s_noise, N: int, K: int
):
    """Produce K importance samples of depth N from the model. That is, K times, we
    generate N importance samples conditioned by the constraints, and categorically
    select one of them."""
    key1, key2 = jax.random.split(key)
    samples, log_weights = jax.vmap(full_model.importance, in_axes=(0, None, None))(
        jax.random.split(key1, N * K), constraints, (motion_settings, s_noise)
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


# Sequential importance resampling

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

def localization_sis(motion_settings, s_noise, observations):
    return SISwithRejuvenation(
        robot_inputs["start"],
        robot_inputs["controls"],
        observations,
        lambda key, pose, control, observation: full_model_kernel.importance(
            key,
            C["sensor", "distance"].set(observation),
            (motion_settings, s_noise, pose, control),
        ),
    )


# SMCP3

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

def localization_sis_plus_grid_rejuv(motion_settings, s_noise, M_grid, N_grid, observations):
    base_grid = make_poses_grid_array(
        jnp.array([M_grid / 2, M_grid / 2]).T,
        N_grid
    )
    return SISwithRejuvenation(
        robot_inputs["start"],
        robot_inputs["controls"],
        observations,
        importance=lambda key, pose, control, observation: full_model_kernel.importance(
            key,
            C["sensor", "distance"].set(observation),
            (motion_settings, s_noise, pose, control),
        ),
        rejuvenate=lambda key, sample, pose, control, observation: run_SMCP3_step(
            grid_fwd_proposal,
            grid_bwd_proposal,
            key,
            sample,
            (base_grid, observation, (motion_settings, s_noise, pose, control))
        ),
    )


# %% [markdown]
# Here are the graphics gadgets.

# %%
# # World plot

# (
#     world_plot
#     + {"title": "Given data"}
# )


# # Pose plot

# some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))
# Plot.html("Click-drag on pose to change location.  Shift-click-drag on pose to change heading.") | (
#     world_plot
#     + pose_widget("pose", some_pose, color="blue")
# ) | Plot.html(js("`pose = Pose([${$state.pose.p.map((x) => x.toFixed(2))}], ${$state.pose.hd.toFixed(2)})`"))

# some_poses = jax.vmap(random_pose)(jax.random.split(key, 20))
# (
#     world_plot
#     + pose_plots(some_poses, color="green")
#     + {"title": "Some poses"}
# )


# # Ideal sensor plot

# some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))
# (
#     (
#         world_plot
#         + plot_sensors(js("$state.pose"), js("$state.pose_readings"), sensor_angles, show_legend=True)
#         + pose_widget("pose", some_pose, color="blue")
#     )
#     | Plot.html(js("`pose = Pose([${$state.pose.p.map((x) => x.toFixed(2))}], ${$state.pose.hd.toFixed(2)})`"))
#     | Plot.initialState({
#         "pose_readings": ideal_sensor(some_pose)
#     })
#     | Plot.onChange({
#         "pose": lambda widget, _: update_ideal_sensors(widget, "pose")
#     })
# )

# key, sub_key = jax.random.split(key)
# some_poses = jax.vmap(random_pose)(jax.random.split(sub_key, 20))
# some_readings = jax.vmap(ideal_sensor)(some_poses)
# Plot.Frames([
#     (
#         world_plot
#         + plot_sensors(pose, some_readings[i], sensor_angles, show_legend=True)
#         + pose_plots(pose)
#     )
#     for i, pose in enumerate(some_poses)
# ], fps=2)


# # Noisy sensor plot

# key, k1, k2 = jax.random.split(key, 3)
# some_pose = Pose(jnp.array([6.0, 15.0]), jnp.array(0.0))
# def on_slider_change(widget, _):
#     update_noisy_sensors(widget, "pose", "noise_slider")
# (
#     (
#         world_plot
#         + plot_sensors(js("$state.pose"), js("$state.pose_readings"), sensor_angles)
#         + pose_widget("pose", some_pose, color="blue")
#     )
#     | noise_slider("noise_slider", "Sensor noise =", sensor_settings["s_noise"])
#     | Plot.html(js("`pose = Pose([${$state.pose.p.map((x) => x.toFixed(2))}], ${$state.pose.hd.toFixed(2)})`"))
#     | Plot.initialState({
#         "k": jax.random.key_data(k1),
#         "pose_readings": noisy_sensor(k2, some_pose, sensor_settings["s_noise"])
#     }, sync={"k"})
#     | Plot.onChange({"pose": on_slider_change, "noise_slider": on_slider_change})
# )


# # Guess-the-pose demo

# key, k1, k2, k3 = jax.random.split(key, 4)
# guess_pose = Pose(jnp.array([2.0, 16.0]), jnp.array(0.0))
# target_pose = Pose(jnp.array([15.0, 4.0]), jnp.array(-1.6))
# def likelihood_function(cm, pose, s_noise):
#     return sensor_model.assess(cm, (pose, sensor_angles, s_noise))[0]
# def on_guess_pose_chage(widget, _):
#     update_ideal_sensors(widget, "guess")
#     widget.state.update({"likelihood":
#         likelihood_function(
#             C["distance"].set(widget.state.target_readings),
#             pose_at(widget.state, "guess"),
#             sensor_settings["s_noise"]
#         )
#     })
# def on_target_pose_chage(widget, _):
#     update_noisy_sensors(widget, "target", "noise_slider")
#     widget.state.update({"likelihood":
#         likelihood_function(
#             C["distance"].set(widget.state.target_readings),
#             pose_at(widget.state, "guess"),
#             sensor_settings["s_noise"]
#         )
#     })
# (
#     Plot.Grid(
#         (
#             world_plot
#             + plot_sensors(js("$state.guess"), js("$state.target_readings"), sensor_angles)
#             + pose_widget("guess", guess_pose, color="blue")
#             + Plot.cond(js("$state.show_target_pose"),
#                 pose_widget("target", target_pose, color="gold"))
#         ),
#         (
#             Plot.rectY(
#                 Plot.js("""
#                 const data = [];
#                 for (let i = 0; i < $state.guess_readings.length; i++) {
#                     data.push({
#                         "sensor index": i - 0.15,
#                         "distance": $state.guess_readings[i],
#                         "group": "wall distances from guess pose"
#                     });
#                     data.push({
#                         "sensor index": i + 0.15,
#                         "distance": $state.target_readings[i],
#                         "group": "sensor readings from hidden pose"
#                     });
#                 }
#                 return data;
#                 """, expression=False),
#                 x="sensor index",
#                 y="distance",
#                 fill="group",
#                 interval=0.5
#             )
#             + Plot.domainY([0, 15])
#             + {"height": 300, "marginBottom": 50}
#             + Plot.color_map({
#                 "wall distances from guess pose": "blue",
#                 "sensor readings from hidden pose": "gold"
#             })
#             + Plot.colorLegend()
#             + {"legend": {"anchor": "middle", "x": 0.5, "y": 1.2}}
#             | [
#                 "div",
#                 {"class": "text-lg mt-2 text-center w-full"},
#                 Plot.js("'log likelihood (greater is better): ' + $state.likelihood.toFixed(2)")
#             ]
#         ),
#         cols=2
#     )
#     | noise_slider("noise_slider", "Sensor noise =", sensor_settings["s_noise"])
#     | (
#         Plot.html([
#             "label",
#             {"class": "flex items-center gap-2 cursor-pointer"},
#             [
#                 "input",
#                 {
#                     "type": "checkbox",
#                     "checked": js("$state.show_target_pose"),
#                     "onChange": js("(e) => $state.show_target_pose = e.target.checked")
#                 }
#             ],
#             "show target pose"
#         ])
#         & Plot.html(js("`guess = Pose([${$state.guess.p.map((x) => x.toFixed(2))}], ${$state.guess.hd.toFixed(2)})`"))
#         & Plot.html(js("`target = Pose([${$state.target.p.map((x) => x.toFixed(2))}], ${$state.target.hd.toFixed(2)})`"))
#     )
#     | Plot.initialState(
#         {
#             "k": jax.random.key_data(k1),
#             "guess_readings": ideal_sensor(guess_pose),
#             "target_readings": (initial_target_readings := noisy_sensor(k3, target_pose, sensor_settings["s_noise"])),
#             "likelihood": likelihood_function(C["distance"].set(initial_target_readings), guess_pose, sensor_settings["s_noise"]),
#             "show_target_pose": False,
#         }, sync={"k", "target_readings"})
#     | Plot.onChange({
#             "guess": on_guess_pose_chage,
#             "target": on_target_pose_chage,
#             "noise_slider": on_target_pose_chage,
#     })
# )


# # Pose prior plots

# key, sub_key = jax.random.split(key)
# some_poses = jax.vmap(lambda k: whole_map_prior.simulate(k, ()))(jax.random.split(sub_key, 100)).get_retval()
# (
#     world_plot
#     + pose_plots(some_poses, color="green")
#     + {"title": "Some poses"}
# )

# key, sub_key = jax.random.split(key)
# some_poses = jax.vmap(lambda k: two_room_prior.simulate(k, ()))(jax.random.split(sub_key, 100)).get_retval()
# (
#     world_plot
#     + pose_plots(some_poses, color="green")
#     + {"title": "Some poses"}
# )

# key, sub_key = jax.random.split(key)
# some_poses = jax.vmap(lambda k: localized_prior.simulate(k, ()))(jax.random.split(sub_key, 100)).get_retval()
# (
#     world_plot
#     + pose_plots(some_poses, color="green")
#     + {"title": "Some poses"}
# )


# # Grid search widget

# N_grid = jnp.array([50, 50, 20])
# N_keep = 1000  # keep the top this many out of the total `jnp.prod(N_grid)` of them
# key, sub_key = jax.random.split(key)
# camera_pose = Pose(jnp.array([2.0, 16.0]), jnp.array(0.0))
# def grid_search_handler(widget, k, readings):
#     model_noise = float(getattr(widget.state, "model_noise"))
#     jitted_posterior = make_posterior_density_fn(widget.state.prior, readings, model_noise)
#     grid_poses = make_poses_grid(world["bounding_box"], N_grid)
#     posterior_densities = jax.vmap(jitted_posterior)(grid_poses)
#     best = jnp.argsort(posterior_densities, descending=True)[0:N_keep]
#     widget.state.update({
#         "grid_poses": grid_poses[best].as_dict(),
#         "best": grid_poses[best[0]].as_dict()
#     })
# camera_widget(
#     sub_key,
#     camera_pose,
#     "grid search",
#     grid_search_handler,
#     result_plots=(
#         pose_plots(js("$state.grid_poses"), color="green", opacity=jnp.arange(1.0, 0.0, -1/N_keep))
#         + pose_plots(js("$state.best"), color="purple")
#     ),
#     bottom_elements=(
#         Plot.html(
#             # For some reason `toFixed` very stubbonrly malfunctions in the following line:
#             Plot.js("""$state.target_exists ?
#                                 `best = Pose([${$state.best.p.map((x) => x.toFixed(2))}], ${$state.best.hd.toFixed(2)})` : ''""")
#         )
#     ),
#     initial_state={
#         "grid_poses": {"p": [], "hd": []},
#         "best": {"p": None, "hd": None},
#     },
# )


# # Grid approximation sampler

# N_grid = jnp.array([50, 50, 20])
# N_samples = 100
# key, sub_key = jax.random.split(key)
# camera_pose = Pose(jnp.array([15.13, 14.16]), jnp.array(1.5))
# def grid_approximation_handler(widget, k, readings):
#     model_noise = float(getattr(widget.state, "model_noise"))
#     jitted_posterior = make_posterior_density_fn(widget.state.prior, readings, model_noise)
#     grid_poses = make_poses_grid(world["bounding_box"], N_grid)
#     posterior_densities = jax.vmap(jitted_posterior)(grid_poses)
#     def grid_sample_one(k):
#         return grid_poses[genjax.categorical.propose(k, (posterior_densities,))[2]]
#     grid_samples = jax.vmap(grid_sample_one)(jax.random.split(k, N_samples))
#     widget.state.update({
#         "sample_poses": grid_samples,
#     })
# camera_widget(
#     sub_key,
#     camera_pose,
#     "grid sampler",
#     grid_approximation_handler,
#     result_plots=pose_plots(js("$state.sample_poses"), color="green"),
#     initial_state={"sample_poses": {"p": [], "hd": []}},
# )


# # Importance resampling widget

# N_presamples = 1000
# N_samples = 100
# key, sub_key = jax.random.split(key)
# camera_pose = Pose(jnp.array([15.13, 14.16]), jnp.array(1.5))
# def importance_resampling_handler(widget, k, readings):
#     model_noise = float(getattr(widget.state, "model_noise"))
#     jitted_posterior = make_posterior_density_fn(widget.state.prior, readings, model_noise)
#     def importance_resample_one(k):
#         k1, k2 = jax.random.split(k)
#         presamples = jax.vmap(random_pose)(jax.random.split(k1, N_presamples))
#         posterior_densities = jax.vmap(jitted_posterior)(presamples)
#         return presamples[genjax.categorical.propose(k2, (posterior_densities,))[2]]
#     grid_samples = jax.vmap(importance_resample_one)(jax.random.split(k, N_samples))
#     widget.state.update({
#         "sample_poses": grid_samples,
#     })
# camera_widget(
#     sub_key,
#     camera_pose,
#     "importance resampler",
#     importance_resampling_handler,
#     result_plots=pose_plots(js("$state.sample_poses"), color="green"),
#     initial_state={"sample_poses": {"p": [], "hd": []}},
# )


# # Markov chain Monte Carlo widget

# N_MH_steps = 1000
# N_samples = 100
# key, sub_key = jax.random.split(key)
# camera_pose = Pose(jnp.array([15.13, 14.16]), jnp.array(1.5))
# def MCMC_handler(widget, k, readings):
#     model_noise = float(getattr(widget.state, "model_noise"))
#     jitted_posterior = make_posterior_density_fn(widget.state.prior, readings, model_noise)
#     def do_MH_step(pose_posterior_density, k):
#         pose, posterior_density = pose_posterior_density
#         k1, k2 = jax.random.split(k)
#         p_hd = pose.as_array()
#         delta = jnp.array([0.5, 0.5, 0.1])
#         mins = jnp.maximum(p_hd - delta, world["bounding_box"][:, 0])
#         maxs = jnp.minimum(p_hd + delta, world["bounding_box"][:, 1])
#         new_p_hd = jax.random.uniform(k1, shape=(3,), minval=mins, maxval=maxs)
#         new_pose = Pose(new_p_hd[0:2], new_p_hd[2])
#         new_posterior = jitted_posterior(new_pose)
#         accept = (jnp.log(genjax.uniform.propose(k2, ())[2]) <= new_posterior - posterior_density)
#         return (
#             jax.tree.map(
#                 lambda x, y: jnp.where(accept, x, y),
#                 (new_pose, posterior_density),
#                 (pose, posterior_density)
#             ),
#             None
#         )
#     def sample_MH_one(k):
#         k1, k2 = jax.random.split(k)
#         start_pose = random_pose(k1)
#         start_posterior = jitted_posterior(start_pose)
#         return jax.lax.scan(do_MH_step, (start_pose, start_posterior), jax.random.split(k2, N_MH_steps))[0][0]
#     grid_samples = jax.vmap(sample_MH_one)(jax.random.split(k, N_samples))
#     widget.state.update({
#         "sample_poses": grid_samples,
#     })
# camera_widget(
#     sub_key,
#     camera_pose,
#     "MCMC trajectories",
#     MCMC_handler,
#     result_plots=pose_plots(js("$state.sample_poses"), color="green"),
#     initial_state={"sample_poses": {"p": [], "hd": []}},
# )


# # Robot motion

# def update_unphysical_path(widget, _):
#     start = pose_at(widget.state, "start")
#     widget.state.update({
#         "path": integrate_controls_unphysical(robot_inputs | {"start": start})
#     })
# (
#     (
#         world_plot
#         + pose_plots(js("$state.path"), color=Plot.constantly("path from integrating controls (UNphysical)"))
#         + pose_widget("start", robot_inputs["start"], color=Plot.constantly("start pose"))
#         + Plot.color_map({"start pose": "blue", "path from integrating controls (UNphysical)": "green"})
#     )
#     | Plot.html(js("`start = Pose([${$state.start.p.map((x) => x.toFixed(2))}], ${$state.start.hd.toFixed(2)})`"))
#     | Plot.initialState({
#         "path": integrate_controls_unphysical(robot_inputs)
#     })
#     | Plot.onChange({"start": update_unphysical_path})
# )

# def update_physical_path(widget, _):
#     start = pose_at(widget.state, "start")
#     widget.state.update({
#         "path": integrate_controls_physical(robot_inputs | {"start": start})
#     })
# (
#     (
#         world_plot
#         + pose_plots(js("$state.path"), color=Plot.constantly("path from integrating controls (physical)"))
#         + pose_widget("start", robot_inputs["start"], color=Plot.constantly("start pose"))
#         + Plot.color_map({"start pose": "blue", "path from integrating controls (physical)": "green"})
#     )
#     | Plot.html(js("`start = Pose([${$state.start.p.map((x) => x.toFixed(2))}], ${$state.start.hd.toFixed(2)})`"))
#     | Plot.initialState({
#         "path": integrate_controls_physical(robot_inputs)
#     })
#     | Plot.onChange({"start": update_physical_path})
# )


# # Step model

# N_samples = 50
# key, k1, k2 = jax.random.split(key, 3)
# def update_confidence_circle(widget, _):
#     step = pose_at(widget.state, "step")
#     step_vector = step.p - robot_inputs["start"].p
#     tilted_start_hd = jnp.atan2(step_vector[1], step_vector[0])
#     tilted_start = Pose(robot_inputs["start"].p, tilted_start_hd)
#     ds = jnp.linalg.norm(step_vector)
#     dhd = (step.hd - tilted_start_hd + jnp.pi) % (2 * jnp.pi) - jnp.pi
#     widget.state.update({
#         "start": tilted_start.as_dict(),
#         "control": {"ds": ds, "dhd": dhd}
#     })
#     k1, k2 = jax.random.split(jax.random.wrap_key_data(widget.state.k))
#     samples = jax.vmap(step_model.propose, in_axes=(0, None))(
#         jax.random.split(k1, N_samples),
#         (default_motion_settings, tilted_start, Control(ds, dhd)),
#     )[2]
#     widget.state.update({
#         "k": jax.random.key_data(k2),
#         "samples": samples.as_dict()
#     })
# (
#     (
#         world_plot
#         + confidence_circle(js("[$state.step.p]"), default_motion_settings["p_noise"])
#         + pose_plots(js("$state.samples"), color=Plot.constantly("samples from the step model"))
#         + pose_plots(js("$state.start"), color=Plot.constantly("start pose"))
#         + pose_widget("step", robot_inputs["start"], color=Plot.constantly("attempt to step to here"))
#         + Plot.color_map({
#             "start pose": "black",
#             "attempt to step to here": "blue",
#             "samples from the step model": "green",
#         })
#     )
#     | Plot.html(js("`control = Control(${$state.control.ds.toFixed(2)}, ${$state.control.dhd.toFixed(2)})`"))
#     | Plot.initialState({
#         "start": robot_inputs["start"].as_dict(),
#         "control": {"ds": 0.0, "dhd": 0.0},
#         "k": jax.random.key_data(k1),
#         "samples": (
#             jax.vmap(step_model.propose, in_axes=(0, None))(
#                 jax.random.split(k2, N_samples),
#                 (default_motion_settings, robot_inputs["start"], robot_inputs["controls"][0]),
#             )[2].as_dict()
#         ),
#     }, sync={"k"})
#     | Plot.onChange({"step": update_confidence_circle})
# )


# # Path model

# key, sample_key = jax.random.split(key)
# path = path_model.propose(sample_key, (default_motion_settings,))[2][1]
# Plot.Frames(
#     [
#         plot_path_with_confidence(path, step)
#         + Plot.title("Motion model (samples)")
#         for step in range(len(path))
#     ],
#     fps=2,
# )

# N_samples = 12
# key, sub_key = jax.random.split(key)
# sample_paths = jax.vmap(
#     lambda k:
#         path_model.propose(k, (default_motion_settings,))[2][1]
# )(jax.random.split(sub_key, N_samples))
# Plot.html([
#     "div.grid.grid-cols-2.gap-4",
#     *[walls_plot + pose_plots(path) + {"maxWidth": 300, "aspectRatio": 1} for path in sample_paths]
# ])


# # Full model

# key, sub_key = jax.random.split(key)
# cm, _, retval = full_model.propose(sub_key, (default_motion_settings, sensor_settings["s_noise"]))
# animate_path_and_sensors(retval[1], cm["steps", "sensor", "distance"], default_motion_settings)


# # Updating traces

# key, k1, k2 = jax.random.split(key, 3)
# trace = step_model.simulate(
#     k1,
#     (default_motion_settings, robot_inputs["start"], robot_inputs["controls"][0]),
# )
# rotated_trace, rotated_trace_weight_diff, _, _ = trace.update(
#     k2, C["hd"].set(jnp.pi / 2)
# )
# (
#     world_plot
#     + pose_plots(trace.get_retval(), color=Plot.constantly("some pose"))
#     + pose_plots(
#         rotated_trace.get_retval(), color=Plot.constantly("with heading modified")
#     )
#     + Plot.color_map({"some pose": "green", "with heading modified": "red"})
#     + Plot.title("Modifying a heading")
# ) | html(f"score ratio: {rotated_trace_weight_diff}")

# key, k1, k2 = jax.random.split(key, 3)
# trace = path_model.simulate(k1, (default_motion_settings,))
# rotated_first_step, rotated_first_step_weight_diff, _, _ = trace.update(
#     k2, C["steps", 0, "hd"].set(jnp.pi / 2)
# )
# (
#     world_plot
#     + [
#         pose_plots(pose, color=Plot.constantly("with heading modified"))
#         for pose in rotated_first_step.get_retval()[1]
#     ]
#     + [
#         pose_plots(pose, color=Plot.constantly("some path"))
#         for pose in trace.get_retval()[1]
#     ]
#     + Plot.color_map({"some path": "green", "with heading modified": "red"})
# ) | html(f"score ratio: {rotated_first_step_weight_diff}")


# # Animating full traces

# key, sub_key = jax.random.split(key)
# tr = full_model.simulate(sub_key, (default_motion_settings, sensor_settings["s_noise"]))
# animate_full_trace(tr)

# (
#     (
#         html("low motion-deviation data")
#         | animate_full_trace(trace_low_deviation, frame_key="frame")
#     ) & (
#         html("high motion-deviation data")
#         | animate_full_trace(trace_high_deviation, frame_key="frame")
#     )
# ) | Plot.Slider("frame", 0, T, fps=2)


# # Making traces with constraints

# key, k1, k2 = jax.random.split(key, 3)
# trace_low, log_weight_low = full_model.importance(
#     k1, constraints_low_deviation, (default_motion_settings, sensor_settings["s_noise"])
# )
# trace_high, log_weight_high = full_model.importance(
#     k2, constraints_high_deviation, (default_motion_settings, sensor_settings["s_noise"])
# )
# (
#     (
#         html("fresh path sample", "fixed low motion-deviation sensor data")
#         | animate_full_trace(trace_low, frame_key="frame")
#         | html(f"log_weight: {log_weight_low}")
#     ) & (
#         html("fresh path sample", "fixed high motion-deviation sensor data")
#         | animate_full_trace(trace_high, frame_key="frame")
#         | html(f"log_weight: {log_weight_high}")
#     )
# ) | Plot.Slider("frame", 0, T, fps=2)


# # Whole-path importance resampling

# N_presamples = 2000
# N_samples = 20
# key, k1, k2 = jax.random.split(key, 3)
# low_posterior = importance_resample(
#     k1, constraints_low_deviation, motion_settings_low_deviation, sensor_settings["s_noise"], N_presamples, N_samples
# )
# high_posterior = importance_resample(
#     k2, constraints_high_deviation, motion_settings_high_deviation, sensor_settings["s_noise"], N_presamples, N_samples
# )
# plot_inference_result(
#     ("importance resampling on low motion-deviation data",),
#     "importance resamples",
#     jax.vmap(get_path)(low_posterior),
#     path_low_deviation
# ) & plot_inference_result(
#     ("importance resampling on high motion-deviation data",),
#     "importance resamples",
#     jax.vmap(get_path)(high_posterior),
#     path_high_deviation
# )


# # Sequential importance resampling

# key, k1, k2 = jax.random.split(key, 3)
# N_particles = 20
# sis_result_low = localization_sis(
#     motion_settings_low_deviation, sensor_settings["s_noise"], observations_low_deviation
# ).run(k1, N_particles)
# sis_result_high = localization_sis(
#     motion_settings_high_deviation, sensor_settings["s_noise"], observations_high_deviation
# ).run(k2, N_particles)
# plot_inference_result(
#     ("SIS on low motion-deviation data",),
#     "sequential importance resamples",
#     [pytree_transpose(path) for path in sis_result_low.backtrack()],
#     path_low_deviation,
#     history_paths=[pytree_transpose(path) for path in sis_result_low.flood_fill()]
# ) & plot_inference_result(
#     ("SIS on high motion-deviation data",),
#     "sequential importance resamples",
#     [pytree_transpose(path) for path in sis_result_high.backtrack()],
#     path_high_deviation,
#     history_paths=[pytree_transpose(path) for path in sis_result_high.flood_fill()]
# )


# # SMCP3

# N_particles = 20
# M_grid = jnp.array([0.5, 0.5, (3 / 10) * degrees])
# N_grid = jnp.array([15, 15, 15])
# key, k1, k2 = jax.random.split(key, 3)
# sis_result = localization_sis(
#     motion_settings_high_deviation, sensor_settings["s_noise"], observations_high_deviation
# ).run(k1, N_particles)
# smcp3_result = localization_sis_plus_grid_rejuv(
#     motion_settings_high_deviation, sensor_settings["s_noise"], M_grid, N_grid, observations_high_deviation
# ).run(k2, N_particles)
# plot_inference_result(
#     ("SIS without rejuvenation", "high motion-deviation data"),
#     "samples",
#     [pytree_transpose(path) for path in sis_result.backtrack()],
#     path_high_deviation,
#     history_paths=[pytree_transpose(path) for path in sis_result.flood_fill()]
# ) & plot_inference_result(
#     ("SIS with SMCP3 grid rejuvenation", "high motion-deviation data"),
#     "samples",
#     [pytree_transpose(path) for path in smcp3_result.backtrack()],
#     path_high_deviation,
#     history_paths=[pytree_transpose(path) for path in smcp3_result.flood_fill()]
# )
