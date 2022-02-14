import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import concatenate as np_concatenate
from numpy import vectorize as np_vectorize
from numpy import arange as np_arange
from numpy import array as np_array
from matplotlib import colors
import matplotlib.font_manager as font_manager
from matplotlib import rcParams

# For plotting policies: direction of triangles for actions
DELTAS_A_0 = (0.337, 0.7, 0.657, 0.7, 0.5, 0.1)
DELTAS_A_1 = (0.3, 0.3, 0.3, 0.7, 0.9, 0.5)
DELTAS_A_2 = (0.337, 0.3, 0.657, 0.3, 0.5, 0.9)
DELTAS_A_3 = (0.7, 0.337, 0.7, 0.657, 0.1, 0.5)
DELTAS_DICT = {0: DELTAS_A_0, 1: DELTAS_A_1, 2: DELTAS_A_2, 3: DELTAS_A_3}


def set_report_style():
    # Global default plotting parameters
    plt.style.use("seaborn-whitegrid")  # ('fast') #('fivethirtyeight') #('bmh')
    rcParams['axes.labelsize'] = 15
    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13
    rcParams['legend.fontsize'] = 18
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = False


COLOR_VALUE = "red"  # "orangered"
COLOUR_POLICY = 'darkorange'

FONT_GRID = {'family': 'serif',  # 'serif',
             'weight': 'normal',  # "bold",  #
             'size': 7}
FONT_LEGEND = {'family': 'serif',  # 'serif',
               'weight': "bold",    #'normal',  #
               'size': 18}


class AgentEmptyObject(object):
    pass


class GoalEmptyObject(object):
    pass


class LavaEmptyObject(object):
    pass


class AgentLegendObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width - 6, handlebox.height + 10
        patch = mpatches.Rectangle([x0, y0 - 4], width, height, facecolor='deepskyblue',
                                   edgecolor='white', lw=1, transform=handlebox.get_transform())
        handlebox.add_artist(patch)

        return patch


class GoalLegendObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width - 6, handlebox.height + 10
        patch = mpatches.Rectangle([x0, y0 - 4], width, height, facecolor='limegreen',
                                   edgecolor='white', lw=1, transform=handlebox.get_transform())
        handlebox.add_artist(patch)

        return patch


class LavaLegendObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width - 6, handlebox.height + 10
        patch = mpatches.Rectangle([x0, y0 - 4], width, height, facecolor='red',
                                   edgecolor='white', lw=1, transform=handlebox.get_transform())
        handlebox.add_artist(patch)

        return patch


def plot_gridworld_value(V, env, policy=None, true_model=False, experience_lvl=None, robust_estimation=None, counts=None,
                         save_path=None, title=None):
    fig_d_0 = env.ncol + 3
    fig_d_1 = env.nrow + 2
    str_format = "%.3f"
    plt.figure(figsize=(fig_d_0, fig_d_1))

    c = plt.pcolormesh(V, cmap="gray", edgecolors='white', linewidth=1)
    if counts is None:
        counts = np_zeros(V.shape)
    if policy is None:
        # Need probabilities from the environment to get pollicy that the agent will follow
        policy = np_zeros((env.nS, env.nA))

    empty_value_function = True if V.sum().sum().sum() == 0 else False
    for y in range(V.shape[0]):
        for x in range(V.shape[1]):
            if not empty_value_function:
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    str_format % V[y, x],
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=COLOR_VALUE,
                    fontdict={'size': 9},
                )
                plt.text(
                    x + 0.85,
                    y + 0.85,
                    "%d" % counts[y, x],
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="firebrick",
                    fontdict=FONT_GRID,
                )
            if policy is not None:
                s = env.make_state_index(y, x)
                # Plotting policy: action 0
                plt.text(
                    x + 0.5,
                    y + 0.2,
                    (str_format % policy[s, 0]).lstrip('0'),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=COLOUR_POLICY,
                    fontdict=FONT_GRID,
                )
                # Action 1
                plt.text(
                    x + 0.8,
                    y + 0.5,
                    (str_format % policy[s, 1]).lstrip('0'),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=COLOUR_POLICY,
                    fontdict=FONT_GRID,
                )
                # Action 2
                plt.text(
                    x + 0.5,
                    y + 0.8,
                    (str_format % policy[s, 2]).lstrip('0'),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=COLOUR_POLICY,
                    fontdict=FONT_GRID,
                )
                # Action 3
                plt.text(
                    x + 0.2,
                    y + 0.5,
                    (str_format % policy[s, 3]).lstrip('0'),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=COLOUR_POLICY,
                    fontdict=FONT_GRID,
                )

    if title is None:
        title = f"W_{env.wind_rate_west if hasattr(env,'wind_rate_west') else None}_S_{env.wind_rate_south if hasattr(env,'wind_rate_south') else None}" \
                f"_TrueEnv_{true_model if hasattr(env,'true_model') else None}_Exp_{experience_lvl if hasattr(env,'experience_lvl') else None}" \
                f"_Robust_{robust_estimation if hasattr(env,'robust_estimation') else None}"
    plt.title(title)
    plt.colorbar(c)
    # In the array, first row = 0 is on top
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(os.path.join(save_path, title+'.png'))
    else:
        plt.show()

    plt.close()


def plot_environment_state(env, desc, policy=None, starting_states=None):
    fig_d_0 = env.ncol + 3
    fig_d_1 = env.nrow
    fig = plt.figure(figsize=(fig_d_0, fig_d_1))

    ax = plt.subplot(1, 1, 1)
    pos = ax.get_position()
    pos.x0 = 0.1
    ax.set_position(pos)

    # Get ANSI codes of the strings
    myfunc_vec = np_vectorize(ord)
    grid_vals = myfunc_vec(desc)
    # Colours will be relative to some value
    grid_vals[grid_vals == ord(b'L')] = 2
    grid_vals[grid_vals == ord(b'G')] = 1
    grid_vals[grid_vals == ord(b'A')] = 0  # 3
    grid_vals[grid_vals == ord(b'F')] = 0
    grid_vals = np_concatenate([grid_vals, -1 * np_ones((grid_vals.shape[0], 1))], axis=1)
    grid_vals = np_concatenate([-1 * np_ones((grid_vals.shape[0], 1)), grid_vals], axis=1)
    grid_vals = np_concatenate([-1 * np_ones((1, grid_vals.shape[1])), grid_vals], axis=0)
    grid_vals = np_concatenate([grid_vals, -1 * np_ones((1, grid_vals.shape[1]))], axis=0)

    cmap = colors.ListedColormap(['gray', 'gainsboro', 'limegreen', 'red', 'deepskyblue'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 20]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    c = plt.pcolormesh(grid_vals, edgecolors='white', linewidth=0.2, cmap=cmap, norm=norm)

    ## Get rid off axis ticks
    ax.set_yticks(np_arange(grid_vals.shape[0]) + 0.5, minor=False, )
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    # ax.set_yticklabels(np.arange(grid_vals.shape[0]))
    ax.set_yticklabels([])
    ax.set_xticks(np_arange(grid_vals.shape[1]) + 0.5, minor=False)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    # ax.set_xticklabels(np.arange(grid_vals.shape[1]))
    ax.set_xticklabels([])

    if policy is not None:
        # Mutable argument :(
        ax = plot_policy(ax, env, policy, starting_states)
    else:
        ## We are just plotting the agent
        xy_agent_sprite = env.make_coordinates(env.s)
        xy_agent_sprite = (xy_agent_sprite[0] + 1, xy_agent_sprite[1] + 1)
        agent_sprite = plt.Polygon(((0.5 + xy_agent_sprite[1], 0.1 + xy_agent_sprite[0]),
                                    (0.9 + xy_agent_sprite[1], 0.5 + xy_agent_sprite[0]),
                                    (0.5 + xy_agent_sprite[1], 0.9 + xy_agent_sprite[0]),
                                    (0.1 + xy_agent_sprite[1], 0.5 + xy_agent_sprite[0])),
                                    fc="cornflowerblue", lw=2)  # "cornflowerblue"
        ax.add_artist(agent_sprite)

    plt.legend([AgentEmptyObject(), GoalEmptyObject(), LavaEmptyObject()],
               ['Agent', 'Goal', 'Lava'],
               handler_map={AgentEmptyObject: AgentLegendObjectHandler(),
                            GoalEmptyObject: GoalLegendObjectHandler(),
                            LavaEmptyObject: LavaLegendObjectHandler()},
               bbox_to_anchor=(0.0, 0.0, 0., .95), frameon=False, prop=font_manager.FontProperties(**FONT_LEGEND))

    plt.gca().invert_yaxis()
    plt.show()

    plt.close()


def plot_policy(ax, env, policy, starting_states):
    for y in range(env.shape[0]):
        for x in range(env.shape[1]):
            state_index = env.make_state_index(y, x)
            if state_index not in env.terminal_states:
                xy_agent_sprite = (x+1, y+1)
                if sum(policy[state_index] == 1):
                    action = policy[state_index].argmax()
                    deltas_a = DELTAS_DICT[action]
                    if state_index in starting_states:
                        agent_sprite = plt.Polygon(((deltas_a[0] + xy_agent_sprite[0], deltas_a[1] + xy_agent_sprite[1]),
                                                    (deltas_a[2] + xy_agent_sprite[0], deltas_a[3] + xy_agent_sprite[1]),
                                                    (deltas_a[4] + xy_agent_sprite[0], deltas_a[5] + xy_agent_sprite[1])),
                                                    fc="orange", alpha=0.9)
                        ax.add_artist(agent_sprite)
                    else:
                        agent_sprite = plt.Polygon(((deltas_a[0] + xy_agent_sprite[0], deltas_a[1] + xy_agent_sprite[1]),
                                                    (deltas_a[2] + xy_agent_sprite[0], deltas_a[3] + xy_agent_sprite[1]),
                                                    (deltas_a[4] + xy_agent_sprite[0], deltas_a[5] + xy_agent_sprite[1])),
                                                   fc="cornflowerblue", alpha=0.9)
                        ax.add_artist(agent_sprite)
                elif sum(policy[state_index] > 1):
                    raise Exception("Policy: actions' probabilities do not sum to 1.")
