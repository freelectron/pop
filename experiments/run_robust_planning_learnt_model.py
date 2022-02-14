from models.model_gridworld_windy import ModelGridworld
from simulators.gridworld_battery import BatteryGridworldEnv
from models.model_gridworld_battery import ModelBatteryGridworld
from simulators.gridworld_windy import GridworldEnv
from utils.plotting import plot_gridworld_value
from agents.agent_dp import AgentDP
from agents.agent_mcts import AgentMCTS
from agents.agent_rmcts_batched import AgentRMCTSBatched
from utils.model_space import *
from agents.agent_mcts import evalute_mcts


def generate_experience_random_walk(env, n_train_episodes, **kwargs):
    """
    Learn a model for different number of experience tuples
    Args:
        env: gridworld environment
        n_train_episodes (int): number of episodes to run
    Returns:
        list, list of tuples of the form (x,y, action, reward, next_state)
    """
    experience = []
    max_langth_flag = False
    while True:
        # s_w = np.random.choice(env.starting_states, 1)[0]
        s_w = np.random.choice(range(env.nS), 1)[0]

        print(s_w)
        print(len(experience))
        print()

        state, _ = env.build(start_state=s_w, seed=None)
        state = env.make_state_index(state[0], state[1])
        t = 0
        done = False
        while not done:
            # Usually, we need an algorithm that takes in the current state
            a = np.random.choice([0, 1, 2, 3])
            (x, y), reward, done, _ = env.perform_step(a)
            state_prime = env.make_state_index(x, y)
            # Record experience
            experience.append((state, a, reward, state_prime))
            state = state_prime
            t += 1

            # Make it numbre of experience tuples
            if len(experience) >= n_train_episodes:
                max_langth_flag = True
                break

        if max_langth_flag:
            return experience

    return experience


def generate_experience_per_sa(env, tuples_per_state_action=10, **kwargs):
    """
    Learn a model for different number of experience tuples
    Args:
        env: gridworld environment
        n_train_episodes (int): number of episodes to run
    Returns:
        list, list of tuples of the form (x,y, action, reward, next_state)
    """
    experience = []
    max_langth_flag = False

    for state in range(env.nS):
        if state not in env.goal_states:
            for a in range(env.nA):
                for experience_tuple in range(tuples_per_state_action):
                    state_init = env.reset(start_state=state)
                    assert state == state_init
                    (x, y), reward, done, _ = env.perform_step(a)
                    state_prime = env.make_state_index(x, y)
                    experience.append((state, a, reward, state_prime))

    return experience


def learn_model(env, experience, T=None, seed=None):
    """
    Args:
        env: true model, used to accurately make the reward function (as it is assumed known)
        experience (list): list of tuples with transitions
        T (numpy.array): matrix of shape S x A x nT x S which we create tranistion dict from
    """
    # The same type of environment
    if isinstance(env, GridworldEnv):
        env_class = ModelGridworld
    elif isinstance(env, BatteryGridworldEnv):
        env_class = ModelBatteryGridworld
    else:
        raise Exception("Unkown environment type.")

    model = env_class(
        experience_tuples=experience,
        T=T,
        wind_states=env.wind_south_states + env.wind_west_states,
        reward_wind=env.reward_wind,
        shape=env.shape,
        seed=seed)
    # If we are learning from experience
    print("Experience length", len(experience))
    if len(experience) > 0:
        print("Creating new transition matrix")
        model.estimate_transition_matrix()
    model.create_transition_dict()
    # Hard pass the reward function
    model.R = env.generate_reward_matrix()

    return model


def run_robust_value_iteration(experience_volumes=[400], wind_rate_model=(0.8, 0.0), reward_wind=-2,
                               agent_constructor=AgentDP, n_experiments=1, seed=None):
    """"
    Perform learnt model planning (LMP).
    Args:
        experience_volumes (list): number of training episodes from which experience is conencted
        agent_constructor (agents): agent class that you want to use for solving MDPs
        seed (int/None): seed to use for creatiom of simulators
    """
    # Create true model
    env = GridworldEnv(shape=[5, 5],  reward_wind=reward_wind, seed=seed)  # BatteryGridworldEnv(shape=[5, 6], seed=seed)  #
    env.build(wind_rate=wind_rate_model)
    P = env.generate_transition_matrix()
    T_true = env.generate_transition_matrix()
    # Which agent to use to solve for true environment
    agent = agent_constructor(env)

    #
    # Working with the true-Environment
    #
    # Usually you want to run the agent untill convergence, now we just run for some predefined number of episodes
    policy_true, v_true = agent.train_model_based(env=env)
    # plot_gridworld_value(v_true.reshape(env.shape), env, policy=policy_true, true_model=True)
    # Possibly more efficient custom solution
    if agent_constructor == AgentDP:
        env.generate_reward_matrix()
        v_new = agent.train_model_based_matrix(env)
        #plot_gridworld_value(v_new.reshape(env.shape), env, true_model=True)

    # Holds number of corect actions
    correct_actions_per_experience_est = []
    correct_actions_per_experience_robust = []

    # Plan for each experiene volume
    for n_train_episodes in experience_volumes:

        policy_est_experiments, policy_robust_experiments = np.zeros((1, env.nS, env.nA)), np.zeros((1, env.nS, env.nA))
        values_est_experiments, values_robust_experiments = np.zeros((1, env.nS)), np.zeros((1, env.nS))

        for i_experiment in range(n_experiments):

            print(f"Experiment #{i_experiment}")

            # experience = generate_experience_random_walk(env, n_train_episodes)
            tuples_per_sa = n_train_episodes // ((env.nS - 2) * 4)
            experience = generate_experience_per_sa(env, tuples_per_sa)

            model = learn_model(env, experience, seed=seed)
            T_estimated = model.generate_transition_matrix()

            # Calculate visitation counts to display later
            state_visit_counts = []
            for s in range(env.nS):
                s_count = sum([model.experience_dict[s][a]['count'].sum() for a in range(env.nA)])
                state_visit_counts.append(s_count)
            state_visit_counts = np.array(state_visit_counts)

            #
            # Non-robust Planning
            #
            # Which agent to use to solve to plan in estimated model
            agent = agent_constructor(model)
            # Usually you want to run the agent untill convergence, now we just run for some predefined number of episodes
            policy_est, v_est = agent.train_model_based(env=model)
            policy_est_experiments = np.vstack([policy_est_experiments, policy_est[np.newaxis, :, :]])
            values_est_experiments = np.vstack([values_est_experiments, v_est[np.newaxis, :]])

            #
            # Robust Planning
            #
            # Create epsilon matrix which has different espilons for different (s,a)'s
            # epsilon_robust = alpha = 0.1
            T_estimated_min = np.copy(T_estimated)
            # For each (s, a) do the optimization
            D = env.create_distances_matrix(range(len(v_est)))
            for s in range(model.nS):
                for a in range(model.nA):
                    p_0_sa = T_estimated[s, a, :, :].squeeze()
                    # Make epsilon be proportional to the number of observations
                    epsilon_robust = 0.4 if sum(model.experience_dict[s][a]['count']) < 30 else 0.05
                    r_sa = env.R.squeeze()[s, a, :]
                    p_sa_min = wasserstein_worstcase_distribution_analytical(p_0_sa, v_est, epsilon_robust, D)
                    T_estimated_min[s, a, 0, :] = p_sa_min

            #
            # Recording results
            #
            # Create the (robust) environment model
            model_worst_case = learn_model(env, [], T=T_estimated_min)
            policy_robust, v_robust = agent.train_model_based(env=model_worst_case)
            # Record
            policy_robust_experiments = np.vstack([policy_robust_experiments, policy_robust[np.newaxis, :, :]])
            values_robust_experiments = np.vstack([values_robust_experiments, v_robust[np.newaxis, :]])

        # Estimated policy and value function
        policy_est = policy_est_experiments[1:, :, :].sum(axis=0) / n_experiments
        v_est = values_est_experiments[1:, :].mean(axis=0)
        # plot_gridworld_value(v_est.reshape(env.shape), model, policy=policy_est,
        #                      experience_lvl=n_train_episodes, robust_estimation=0,
        #                      counts=state_visit_counts.reshape(env.shape))

        # Robust policy and value function
        policy_robust = policy_robust_experiments[1:, :, :].sum(axis=0) / n_experiments
        v_robust = values_robust_experiments[1:, :].mean(axis=0)
        # plot_gridworld_value(v_robust.reshape(env.shape), model, policy=policy_robust,
        #                      experience_lvl=n_train_episodes, robust_estimation=1,
        #                      counts=state_visit_counts.reshape(env.shape))

        # Collect the results so we can plot them later
        policy_true_mask = np.zeros(policy_est.shape)
        starting_states = [2, 6, 10]
        optimal_robust_actions = [1, 2]
        policy_true_mask[starting_states[:], optimal_robust_actions[0]: optimal_robust_actions[1] + 1] = 1

        # Record whether the policy was correct
        correct_actions_per_experience_robust.append((policy_true_mask * policy_robust).sum())
        correct_actions_per_experience_est.append((policy_true_mask * policy_est).sum())

    return correct_actions_per_experience_est, correct_actions_per_experience_robust


def run_rmcts_batched(experience_volumes=[4000],
                      agent_nonrobust_constructor=AgentMCTS,
                      agent_robust_constructor=AgentRMCTSBatched,
                      agent_params=dict(uct_tree_policy=True, num_rollouts=8000, num_batches=2, horizon=2,
                                        discount_factor=.99, epsilon_robust=0.4, uct_exploration_weight=2),
                      n_experiments=1, seed=None):
    """"
    Perform learnt model planning (LMP).
    Args:
        experience_volumes (list): number of training episodes from which experience is conencted
        agent_nonrobust_constructor (agents): agent class that you want to use for solving MDPs
        seed (int/None): seed to use for creatiom of simulators
    """
    # Create true model: Gridworld
    wind_rate_model = (0.8, 0.0)
    env = GridworldEnv([5, 5], wind_rate_model=wind_rate_model, reward_wind=-2, seed=seed)
    # Create true model: BatteryGridworld
    # up_down_stochasticity = (0.75, 0.75)
    # env = BatteryGridworldEnv(shape=[5, 7], wind_rate_model=up_down_stochasticity, seed=seed, reward_wind=-1)

    _, _ = env.build(start_state=13, seed=seed)
    P = env.generate_transition_matrix()
    T_true = env.generate_transition_matrix()
    R = env.generate_reward_matrix()
    # Which agent to use to solve for true environment
    agent_nonrobust = agent_nonrobust_constructor(env, **agent_params)

    #
    # Working with the true-Environment
    #
    # Regular evalute
    policy_true, v_true = agent_nonrobust.evaluate(n_mcts_runs=1, eval_range=None)  # range(1, env.nS-1))
    policy_true = policy_true.sum(axis=0) / n_experiments
    v_true = v_true.mean(axis=0)
    plot_gridworld_value(v_true.reshape(env.shape), env, policy=policy_true, true_model=True)

    # Plan for each experiene volume
    for tuples_per_state_action in experience_volumes:
        # Store results from runs with different experience
        policy_est_experiments, policy_robust_experiments = np.zeros((1, env.nS, env.nA)), np.zeros((1, env.nS, env.nA))
        values_est_experiments, values_robust_experiments = np.zeros((1, env.nS)), np.zeros((1, env.nS))

        # Plan for each experiene volume
        for i_experiment in range(n_experiments):

            # experience = generate_experience_per_sa(env, tuples_per_state_action=tuples_per_state_action)
            experience = generate_experience_random_walk(env, tuples_per_state_action)
            model = learn_model(env, experience, seed=seed)
            T_estimated = model.generate_transition_matrix()

            # Calculate visitation counts to display later
            state_visit_counts = []
            for s in range(env.nS):
                s_count = sum([model.experience_dict[s][a]['count'].sum() for a in range(env.nA)])
                state_visit_counts.append(s_count)
            state_visit_counts = np.array(state_visit_counts)

            #
            # Non-robust Planning
            #
            # Usually you want to run the agent untill convergence, now we just run for some predefined number of episodes
            agent_nonrobust = agent_nonrobust_constructor(model, **agent_params)
            policy_est, v_est = agent_nonrobust.evaluate(n_mcts_runs=1, eval_range=None)  # range(1, env.nS-1))
            # Record the results
            policy_est_experiments = np.vstack([policy_est_experiments, policy_est])
            values_est_experiments = np.vstack([values_est_experiments, v_est])

            #
            # Robust Planning
            #
            agent_robust = agent_robust_constructor(env=model, **agent_params)
            policy_robust, v_robust = agent_robust.evaluate(n_mcts_runs=1, eval_range=None)  # range(1, env.nS-1))
            # Record the results
            policy_robust_experiments = np.vstack([policy_robust_experiments, policy_robust])
            values_robust_experiments = np.vstack([values_robust_experiments, v_robust])

            print(f"EPSILON {agent_robust.epsilon_robust}")

        policy_est = policy_est_experiments[1:, :, :].sum(axis=0) / n_experiments
        v_est = values_est_experiments[1:, :].mean(axis=0)
        # Normalization of v's to record the relative difference (not really needed)
        plot_gridworld_value(v_est.reshape(env.shape), model, policy=policy_est,
                             experience_lvl=tuples_per_state_action, robust_estimation=0,
                             counts=state_visit_counts.reshape(env.shape))

        policy_robust = policy_robust_experiments[1:, :, :].sum(axis=0) / n_experiments
        v_robust = values_robust_experiments[1:, :].mean(axis=0)
        plot_gridworld_value(v_robust.reshape(env.shape), model, policy=policy_robust,
                             experience_lvl=tuples_per_state_action, robust_estimation=1,
                             counts=state_visit_counts.reshape(env.shape))


if __name__ == "__main__":
    """
    Planning with learnt models.
    """
    import time
    start = time.time()

    # Scalable due to quick wasserstein worst-case model calculation
    from scipy.stats import entropy
    from scipy.optimize import fsolve, brute, minimize_scalar

    def get_nu_left(e_delta, nu_big=False):
        """
        Get nu_left given entropy delta. We are missing Newton's quadratic convergence but with.
        e_delta should always be given positive.
        If nu_big is True, then nu selected from [0.5:1] else from [0:0.5].
        """
        # Under certain conditions: wind_rate = p_sa
        ent_delta = lambda wind_rate: abs(entropy([wind_rate, 1 - wind_rate], base=2) - e_delta)

        res0 = minimize_scalar(ent_delta, bounds=(0.0, 0.5), method='bounded')

        res1 = minimize_scalar(ent_delta, bounds=(0.5, 1), method='bounded')

        return res0.x, res1.x

    # Store results
    results_grid = np.zeros(shape=(1, 4))

    step_delta_entropy = 0.2
    step_delta_action = 1.
    # First entropy gap, later action gap
    g = np.mgrid[-1.0:1.1:step_delta_entropy, 0:4.:step_delta_action]
    e_a_mesh = list(zip(*(x.flat for x in g)))

    for entropy_delta, action_gap in e_a_mesh:
        if entropy_delta < 0:
            # Safe action is better
            wind_rate_s, wind_rate_b = get_nu_left(-entropy_delta, nu_big=False)
        else:
            # Risky action is better
            nu_left_true_s, nu_left_true_b = get_nu_left(entropy_delta, nu_big=False)

        for wind_rate in [wind_rate_b]:  # [wind_rate_s, wind_rate_b]:
            if entropy_delta < 0:
                reward_wind = (-action_gap - 0.17 - 0.78 * wind_rate) / wind_rate
            else:
                reward_wind = (action_gap - 0.17 - 0.78 * wind_rate) / wind_rate

            # Create true model
            env = GridworldEnv(shape=[5, 5], reward_wind=reward_wind,
                               seed=None)  # BatteryGridworldEnv(shape=[5, 6], seed=seed)  #
            env.build(wind_rate=(wind_rate, 0))
            P = env.generate_transition_matrix()
            T_true = env.generate_transition_matrix()
            # Which agent to use to solve for true environment
            agent = AgentDP(env)
            #
            # Working with the true-Environment
            #
            # Usually you want to run the agent untill convergence, now we just run for some predefined number of episodes
            # policy_true, v_true = agent.train_model_based(env=env)
            # plot_gridworld_value(v_true.reshape(env.shape), env, policy=policy_true, true_model=True)

            # Q-learning solution
            from agents.agent_qlearning import AgentQLearning
            agent = AgentQLearning(env)
            # env.generate_reward_matrix()
            # policy, v = agent.train_model_based(env, num_episodes=10, discount_factor=0.99, alpha=0.01, epsilon=0.1,
            #               seed=None, double_qlearning=False)
            # plot_gridworld_value(v_new.reshape(env.shape), env, true_model=True)

            #
            # Run the main comparisment
            #
            correct_actions_nonrobust, correct_actions_robust = run_robust_value_iteration(wind_rate_model=(wind_rate, 0),
                                                                                           reward_wind=reward_wind,
                                                                                           n_experiments=10)
            #run_rmcts_batched()

            x = action_gap
            y = entropy_delta
            z_non_robust = correct_actions_nonrobust[0]
            z_robust = correct_actions_robust[0]

            result_row = np.array([x, y, z_non_robust, z_robust])
            results_grid = np.vstack([results_grid, result_row])

    # Plot rewards
    x = results_grid[1:, 0]
    y = results_grid[1:, 1]
    z_non_robust = results_grid[1:, 2]
    z_robust = results_grid[1:, 3]

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use('Qt5Agg')
    plt.style.use("seaborn-whitegrid")  # ('fast') #('fivethirtyeight') #('bmh')

    fig = plt.figure(figsize=(14, 10))

    # Plot just entrpy gap and reward
    # plt.plot(x, z_non_robust)
    # plt.show()
    # plt.plot(y, z_robust)
    # plt.show()

    color = 'red'
    color_robust = 'green'

    ax = fig.gca(projection="3d")
    surf1 = ax.plot_trisurf(x, y, z_non_robust, linewidth=0.2, antialiased=True, alpha=0.8, label='non-robust',
                            color=color)
    surf1._facecolors2d = surf1._facecolors3d
    surf1._edgecolors2d = surf1._edgecolors3d
    surf2 = ax.plot_trisurf(x, y, z_robust, linewidth=0.2, antialiased=True, alpha=0.8, label='robust',
                            color=color_robust)
    surf2._facecolors2d = surf2._facecolors3d
    surf2._edgecolors2d = surf2._edgecolors3d

    ax.tick_params(axis='x', which='major', pad=10.0, rotation=-20)
    ax.tick_params(axis='y', which='major', pad=0.0, rotation=-0)
    ax.tick_params(axis='z', which='major', pad=10.0)

    ax.view_init(15, 20)

    plt.title("Average Regret", fontsize=20)
    ax.set_xlabel("Action Gap", labelpad=30)
    ax.set_ylabel("Entropy Delta", labelpad=25)
    ax.set_zlabel("Regret", labelpad=27)

    plt.legend(loc='upper right', bbox_to_anchor=(0.9, .85), ncol=1, fancybox=True, shadow=True)
    # fig.tight_layout()
    plt.show()

    end = time.time()
    print(f"{(end - start) // 60} mins {(end - start) % 60} secs")

