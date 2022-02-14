import numpy as np
import gym
import random
import time

policy_to_action = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}
GAMMA = 0.99


def epsilon_greedy(a, env, eps=0.1):
    p = np.random.random()
    if p < 1 - eps:  # exploit
        return a
    else:  # explore
        return np.random.randint(0, env.nA)


def play_game(env, policy, EPSILON):
    s = env.reset()
    a = epsilon_greedy(policy[s], env, eps=EPSILON)

    # reward belong to one state and action before
    state_action_reward = [(s, a, 0)]
    while True:
        s, r, terminated, _ = env.step(a)
        if terminated:
            state_action_reward.append((s, None, r))
            break
        else:
            a = epsilon_greedy(policy[s], env, eps=EPSILON)
            state_action_reward.append((s, a, r))
    G = 0
    state_action_return = []
    first = True
    for s, a, r in reversed(state_action_reward):
        if first:
            first = False
        else:
            state_action_return.append((s, a, G))

        G = r + GAMMA * G
        state_action_return.reverse()
    return state_action_return


def monte_carlo(env, EPSILON=0.5, N_EPISODES=10000):
    policy = np.random.choice(env.nA, env.nS)
    Q = {}
    visit = {}
    for s in range(env.nS):
        Q[s] = {}
        visit[s] = {}
        for a in range(env.nA):
            Q[s][a] = 0
            visit[s][a] = 0

    deltas = []  # keep track of learning curve
    for i in range(N_EPISODES):
        # epsilon decreasing
        esp = max(0, EPSILON - i / N_EPISODES)
        state_action_return = play_game(env, policy, esp)
        seen_state_action = set()
        biggest_change = 0
        for s, a, G in state_action_return:
            if (s, a) not in seen_state_action:
                visit[s][a] += 1
                oldq = Q[s][a]
                # incremental mean
                Q[s][a] = Q[s][a] + (G - Q[s][a]) / visit[s][a]
                seen_state_action.add((s, a))
                biggest_change = max(biggest_change, np.abs(oldq - Q[s][a]))
        deltas.append(biggest_change)

        # update policy
        for s in Q.keys():
            best_a = None
            best_G = float('-inf')
            for a, G in Q[s].items():
                if G > best_G:
                    best_G = G
                    best_a = a
            policy[s] = best_a
    V = []
    for s in Q.keys():
        best_G = float('-inf')
        for _, G in Q[s].items():
            if G > best_G:
                best_G = G
        V.append(best_G)
    return V, policy, deltas


# game = gym.make('FrozenLake-v0')
# env = game.env
from simulators.gridword_frozen_lake import FrozenLakeEnv
env = FrozenLakeEnv()
start = time.time()
values_0, policy_0, Delta = monte_carlo(env, EPSILON=0.5, N_EPISODES=100000)
print('TIME TAKEN {} seconds'.format(time.time() - start))
print(policy_0)
print()
print(values_0)
print()

########################################################################################################################
########################################################################################################################
########################################################################################################################


""" Solving FrozenLake8x8 from OpenAI using Value Iteration
    Author: Diganta Kalita  (digankate26@gmail.com) """


def value_iteration(env, max_iterations=100000, lmbda=0.9):
    stateValue = [0 for i in range(env.nS)]
    newStateValue = stateValue.copy()
    for i in range(max_iterations):
        for state in range(env.nS):
            action_values = []
            for action in range(env.nA):
                state_value = 0
                for i in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][i]
                    state_action_value = prob * (reward + lmbda * stateValue[next_state])
                    state_value += state_action_value
                action_values.append(state_value)  # the value of each action
                best_action = np.argmax(np.asarray(action_values))  # choose the action which gives the maximum value
                newStateValue[state] = action_values[best_action]  # update the value of the state
        if i > 1000:
            if sum(stateValue) - sum(newStateValue) < 1e-04:  # if there is negligible difference break the loop
                break
                print(i)
        else:
            stateValue = newStateValue.copy()
    return stateValue


def get_policy(env, stateValue, lmbda=0.9):
    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_values = []
        for action in range(env.nA):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, r, _ = env.P[state][action][i]
                action_value += prob * (r + lmbda * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy


def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:

            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(np.mean(steps_list)))
    print('And you fell in the hole {:.2f} % of the times'.format((misses / episodes) * 100))
    print('----------------------------------------------')


# env = gym.make('FrozenLake-v0')
from simulators.gridword_frozen_lake import FrozenLakeEnv
env = FrozenLakeEnv()

values_1 = value_iteration(env, lmbda=0.99, max_iterations=10000)
policy_1 = get_policy(env, values_1)
get_score(env, policy_1, episodes=100000)
print(policy_1)
print()
print(values_1)
print()

########################################################################################################################
########################################################################################################################
########################################################################################################################


"""Brute force"""

import numpy
import time
import gym

"""
    Args:
    poicy [S,A] shaped matrix representing policy.
    env. OpenAi gym env.v.
      env.P represents the transition propablities of the env
      env.P[s][a] is a list of transition tuples 
      env.nS = is a number of states
      env.nA is a number of actions
    gamma: discount factor
    render: boolean to turn rendering on/off 
"""


def execute(env, policy, gamma=1.0, render=False):
    start = env.reset()
    totalReward = 0
    stepIndex = 0
    while True:
        if render:
            env.render()
        start, reward, done, _ = env.step(int(policy[start]))
        totalReward += (gamma ** stepIndex * reward)
        stepIndex += 1
        if done:
            break
    return totalReward


# Evaluation
def evaluatePolicy(env, policy, gamma=1.0, n=100):
    scores = [execute(env, policy, gamma, False) for _ in range(n)]
    return numpy.mean(scores)


# choosing a policy given a value-function
def calculatePolicy(v, gamma=1.0):
    policy = numpy.zeros(env.nS)
    for s in range(env.nS):
        q_sa = numpy.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = numpy.argmax(q_sa)
    return policy


# Value Iteration Algorithm
def valueIteration(env, gamma=1.0):
    value = numpy.zeros(env.nS)
    max_iterations = 10000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = numpy.copy(value)
        for s in range(env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            value[s] = max(q_sa)
        if (numpy.sum(numpy.fabs(prev_v - value)) <= eps):
            print("Value-Iteration converged at $ %d" % (i + 1))
            break
    return value


gamma = 0.99
from simulators.gridword_frozen_lake import FrozenLakeEnv
env = FrozenLakeEnv()

optimalValue = valueIteration(env, gamma)
startTime = time.time()
policy = calculatePolicy(optimalValue, gamma)
policy_score = evaluatePolicy(env, policy, gamma, 1000)
endTime = time.time()
print("Best score = %0.2f. Time taken = %4.4f seconds" % (numpy.mean(policy_score), endTime - startTime))