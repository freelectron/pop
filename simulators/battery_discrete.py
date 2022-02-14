import numpy as np
from gym import spaces

from simulators.battery import BatterySystem
from utils.distribution import categorical_sample


ACTION_CHARGE_MAPPING = {
    0: -1,  # DISCHARGE
    1: 1,  # CHARGE
    2: 0,  # IDLE
}
# Buckets used to ditgitize
PRICE_BUCKETS = [-1000, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
# Buckets used to convert from pred_bucket to pred_(mid)_price
PRICE_BUCKETS_MID = [-25] + list(range(-15, 115, 10)) + [1000]
# Add epsilon later so that if battary 100 , we have lvl 10, instead of 11 (see how np.digitize works)
ENERGY_LVLS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 + 1e-3]
# Produced after (imba) feature engineering
PATH_MARKET_FEATURES_PREDS = {
    "X": "./storage/imba_hindcast_storage/proba_2019_01_to_04/X.parquet",
    "Y": "./storage/imba_hindcast_storage/proba_2019_01_to_04/Y.parquet",
    "Y_prices": "./storage/imba_hindcast_storage/proba_2019_01_to_04/Y_prices.parquet",
    # Prediction for future PTU's
    "Y_afn_preds_pte0": "./storage/imba_hindcast_storage/proba_2019_01_to_04/Y_preds_afnemen.parquet",
    "Y_inv_preds_pte0":  "./storage/imba_hindcast_storage/proba_2019_01_to_04/Y_preds_invoeden.parquet",
    "Y_afn_preds_pte1": "./storage/nl_imba_hind/full_hindcast/afnemen_preds_pte1.pq",
    "Y_inv_preds_pte1": "./storage/nl_imba_hind/full_hindcast/invoeden_preds_pte1.pq",
    "Y_afn_preds_pte2": "./storage/nl_imba_hind/full_hindcast/afnemen_preds_pte2.pq",
    "Y_inv_preds_pte2": "./storage/nl_imba_hind/full_hindcast/invoeden_preds_pte2.pq",
    "Y_afn_preds_pte3": "./storage/nl_imba_hind/full_hindcast/afnemen_preds_pte3.pq",
    "Y_inv_preds_pte3": "./storage/nl_imba_hind/full_hindcast/invoeden_preds_pte3.pq",
    "Y_afn_preds_pte4": "./storage/nl_imba_hind/full_hindcast/afnemen_preds_pte4.pq",
    "Y_inv_preds_pte4": "./storage/nl_imba_hind/full_hindcast/invoeden_preds_pte4.pq",
    "Y_afn_preds_pte5": "./storage/nl_imba_hind/full_hindcast/afnemen_preds_pte5.pq",
    "Y_inv_preds_pte5": "./storage/nl_imba_hind/full_hindcast/invoeden_preds_pte5.pq",
    "Y_afn_preds_pte6": "./storage/nl_imba_hind/full_hindcast/afnemen_preds_pte6.pq",
    "Y_inv_preds_pte6": "./storage/nl_imba_hind/full_hindcast/invoeden_preds_pte6.pq",
}


class DiscreteBatterySystem(BatterySystem):
    """
    State representation: VECTOR
    Simulate the behaviour of a simple battery system.
    I unite battery features (SoC, switching costs) with market features (imbalance price). There are two options
    to get market features: simulate them or load historical prices.
    """
    def __init__(
            self,
            bat_stor,
            bat_cap,
            soc_min,
            soc_max,
            eff_ac_dc,
            deg_per_cycle=0,
            max_cycles_day=False,
            time_step_delta=1,
            price_buckets_mid=PRICE_BUCKETS_MID,
            energy_lvls=ENERGY_LVLS,
            historic_prices_period=("0001-01-01 00:00:00", "2525-01-01 00:00:00"),
            run_period=("0001-01-01 00:00:00", "2525-01-01 00:00:00"),
            historic_feats_path=PATH_MARKET_FEATURES_PREDS,
            historic_preds_path=PATH_MARKET_FEATURES_PREDS,
    ):
        """
        Initialize battery system with different parameters
        Args:
            time_step_delta (int): number of minutes that will pass after a decision is made
            historic_prices_period (tuple): start and end datetime to use to load hystoric data
        """
        # Get mother class methods and variables
        super(DiscreteBatterySystem, self).__init__(bat_stor,
                                                    bat_cap,
                                                    soc_min,
                                                    soc_max,
                                                    eff_ac_dc,
                                                    deg_per_cycle,
                                                    max_cycles_day,
                                                    time_step_delta,
                                                    price_buckets_mid,
                                                    energy_lvls,
                                                    historic_prices_period,
                                                    run_period,
                                                    historic_feats_path,
                                                    historic_preds_path)

        # How many SoC and price levels, ideally 4 * 15 soc lvls
        self.card_soc = 4
        self.card_price = len(PRICE_BUCKETS) - 1
        # Thus, number of states
        self.nS = self.card_price * self.card_soc
        self.nA = len(ACTION_CHARGE_MAPPING.keys())
        # Number of PTU's in the future to consider
        self.nT = 7
        # Determine charging amount based on the chosen cardinality
        self.step_charge_amount = self.SoC_max / self.card_soc
        # To create tabular setting we need transition matrix and reward matrix
        self.T = np.zeros((self.nS, self.nA, self.nT, self.nS), dtype="float64")
        self.R = None
        # Next state matrix
        self.Q = None
        # Matrix of shape S x A x 'cardinatlity row feature (price)'
        self.structure_matrix = self.structure_matrix = np.zeros((self.nS, self.nA, 1, self.nS), dtype="int")

        # Initial state of the battery [market and physical features], discreteized so env.s is a state index
        self.s = None
        # Set 3 dimensional action space as discrete: charge, discharge, or idle
        self.action_space = spaces.Discrete(3)
        # Define state space
        # TODO: how to determine state_shape automatically?
        self.state_shape = 3
        # TODO: adjust this to be discrete Box-es
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_shape,), dtype=np.float64)
        # Give environment a name
        self.env_id = "Discrete-Battery-System"

    def create_tranistion_structure(self):
        """
        Based on the parameters of this discrete battery.
        """
        # Will suit as a mask
        for soc_lvl in range(self.card_soc):
            for price_bucket in range(self.card_price):
                state = price_bucket + soc_lvl * self.card_price
                for action in range(self.nA):
                    # Find reachble states from one (state, action)-pair
                    # Discharge
                    if (action == 0) and (not soc_lvl == 0):
                        rs = [(soc_lvl - 1) * self.card_price + i for i in range(self.card_price)]
                    # Charge
                    elif (action == 1) and (not soc_lvl == self.card_soc - 1):
                        rs = [(soc_lvl + 1) * self.card_price + i for i in range(self.card_price)]
                    # Idle OR we cannot charge anymore OR we cannot discharge anymore
                    else:
                        rs = [soc_lvl * self.card_price + i for i in range(self.card_price)]
                    self.structure_matrix[state, action, 0, rs] = 1

        return self.structure_matrix

    def generate_transition_matrix(self):
        """
        Create a transition matrix for the current datetime and nT time steps ahead.
        I can create a transition matrix based on Afnemen or Invoeden probabilities.
        """
        for t in range(len(self.preds_inv_ptes)):
            p_sa = self.preds_afn_ptes[t].loc[self.datetime_current].iloc[:-1].values
            trans_probs = p_sa.reshape((1, -1)).repeat(self.nS * self.nA, axis=0).reshape((self.nS, self.nA, 1, self.card_price))
            # Place the transition probabilities based on the mask
            np.place(self.T[:, :, t, :], self.structure_matrix > 0, trans_probs)

        return self.T.copy()

    def generate_reward_matrix(self):
        self.R = np.zeros((self.nS, self.nA, self.nS))
        # Discharge
        rewards_sa0 = self.step_charge_amount * np.array([np.array(self.price_buckets_mid)[i].astype(int) -
                                                                       self.mean_bought_price for i in
                                                                       range(self.card_price)])
        rewards_sa0 = rewards_sa0.reshape((1, -1)).repeat(self.nS, axis=0).reshape((self.nS, self.card_price))
        # Correct the first few states because we cannot discharge from them
        np.place(self.R[self.card_price:, 0, :], self.structure_matrix[self.card_price:, 0, 0, :] > 0, rewards_sa0[self.card_price:, :])
        # Charge
        rewards_sa1 = self.step_charge_amount * np.array([self.df_episode.loc[:, "afnemen"].mean() -
                                                               np.array(self.price_buckets_mid)[i].astype(int)
                                                               for i in range(self.card_price)])
        rewards_sa1 = rewards_sa1.reshape((1, -1)).repeat(self.nS, axis=0).reshape((self.nS, self.card_price)) * 2
        # Correct for the last rows we can not charge anymore
        np.place(self.R[:-self.card_price, 1, :], self.structure_matrix[:-self.card_price, 1, 0, :] > 0, rewards_sa1[:-self.card_price, :])

        return self.R.copy()

    def is_terminal(self, state, t):
        """
        Terminal state is the state in the last PTU.
        In our case, idicator of terminal is that we are in the last PTU.
        Args:
            t (int): tree deptt or horizon (in simulation).
        """
        return t >= self.nT-1

    def transition(self, s, a, t=0):
        """
        Note: rename to sample_transition() ?
        """
        d = self.T[s, a, t, :]
        # Sample next state
        # ToDo: categorical_sample, check correctness
        s_p = categorical_sample(d, np.random)
        r = self.R[s, a, s_p]
        # In our environment terminal means we dont have predictions anymore
        done = self.is_terminal(None, t)

        return s_p, r, done

    def reset(self, current_energy_stored=None, datetime_dt=None, episode_len_ptu=96, test_time=False):
        super(DiscreteBatterySystem, self).reset(current_energy_stored, datetime_dt, episode_len_ptu, test_time)
        # todo: how to handle not equal SoC levels?
        self.s = int(self.bat_stored / self.SoC_max * self.card_soc)

        return self.state

    def step(self, action):
        # todo: when to call self.s , before or after update?
        self.state, reward, done, _ = super(DiscreteBatterySystem, self).step(action)
        self.s = int(self.bat_stored / self.SoC_max * self.card_soc)

        return self.state, reward, done, _


if __name__ == "__main__":
    # USE THIS PARAMS !
    bat_stor = 1.0
    bat_cap = 1.0
    SoC_min = 0.0
    SoC_max = 1.0
    eff_AC_DC = 1.0
    bat_SoC = SoC_max
    deg_per_cycle = 0
    max_cycles_day = 50

    env = DiscreteBatterySystem(
        bat_stor,
        bat_cap,
        SoC_min,
        SoC_max,
        eff_AC_DC,
        deg_per_cycle,
        max_cycles_day,
        historic_prices_period=("2019-02-01 00:00:00", "2019-03-01 00:00:00"),
        # Period to train on
        run_period=("2019-02-01 00:00:00", "2019-03-01 00:00:00"))

    env.reset()
    env.create_tranistion_structure()

    from agents.agent_mcts import AgentMCTS
    agent_mcts = AgentMCTS(env.action_space, num_rollouts=250, horizon=100, discount_factor=0.8)

    # while True:
    #     env.generate_transition_matrix()
    #     env.generate_reward_matrix()
    #     action = agent_mcts.select_action(env, False)
    #     next_state = env.step(action)
    #     # todo: define stepping out point

    env.step(1)
    env.step(2)
    env.step(0)

    print("Debugging.")

    ################## Testing ##########################
    # Legacy, to check correctness
    #####################################################
