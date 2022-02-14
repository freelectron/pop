from abc import ABC
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Iterable, Optional, Dict

import numpy as np
import pandas as pd
import gym
from gym import spaces


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
PATH_MARKET_FEATURES_PREDS = { # proba_2019_01_to_04
    "X": "./storage/imba_hindcast_storage/test_start_2019_test_96_hind_4/X.parquet",
    "Y": "./storage/imba_hindcast_storage/test_start_2019_test_96_hind_4/Y.parquet",
    "Y_prices": "./storage/imba_hindcast_storage/test_start_2019_test_96_hind_4/Y_prices.parquet",
    # Prediction for future PTU's
    "Y_afn_preds_pte0": "./storage/imba_hindcast_storage/test_start_2019_test_96_hind_4/Y_preds_afnemen.parquet",
    "Y_inv_preds_pte0": "./storage/imba_hindcast_storage/test_start_2019_test_96_hind_4/Y_preds_invoeden.parquet",
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
# Features without Nans
# POLICY_MARKET_FEATURES = ["apx_price_eur_mwh", "minute_weight_mean_price",
#                           "minute_calc_price_up_feat_shift_pte_1", "minute_calc_price_down_feat_shift_pte_1",
#                           "minute_calc_nv_feat_sum_pte_1",
#                           "minute_balans_delta", "minute_balans_delta_igcc_feat_rolling_sum_2",
#                           "minute_afregelen_feat_diff_lag_1",
#                           "minute_opregel_sum", "minute_opregelen_reserve",
#                           "minute_afregel_sum", "minute_afregelen_reserve"]
POLICY_MARKET_FEATURES = ["minute_weight_mean_price",
                          "minute_calc_price_up_feat_shift_pte_1", "minute_calc_price_down_feat_shift_pte_1",
                          "minute_calc_nv_feat_sum_pte_1"]


class BatterySystem(gym.core.Env, ABC):
    """
    State representation: VECTOR
    Simulate the behaviour of a simple battery system.
    I unite battery features (SoC, switching costs) with market features (imbalance price). There are two options
    to get market features: simulate them or load historical prices.
    """
    PARAMETERS = {}

    @staticmethod
    def calc_profit(df_episode, time_step_delta):
        """
        Calculate profit.
        """
        # Evaluate the agent in money terms
        charge_delta = df_episode.SoC.shift(-time_step_delta) - df_episode.SoC
        # We charged
        pos_charge_delta = charge_delta[charge_delta > 0]
        # We discharged
        neg_charge_delta = charge_delta[charge_delta < 0]
        # We spent on charging
        costs_charge = -pos_charge_delta * df_episode.loc[pos_charge_delta.index].afnemen
        # Negative delta is we discharged thus should be converted to revenue
        profits_discharge = -neg_charge_delta * df_episode.loc[neg_charge_delta.index].invoeden
        profits_discharge.name = 'profit_loss'
        costs_charge.name = 'profit_loss'
        df = pd.DataFrame(index=df_episode.index)
        df.loc[costs_charge.index, 'profit_loss'] = costs_charge
        df.loc[profits_discharge.index, 'profit_loss'] = profits_discharge
        df['cum_sum'] = df.profit_loss.cumsum()
        profit_loss = df.cum_sum.ffill().fillna(0)

        return profit_loss

    @staticmethod
    def get_mid_price(df_preds_afn, df_preds_inv, df_episode_index, price_buckets):
        """
        Convert bucket predictions to mid-bucket prices.
        Buckets start from zero.
        """
        inv_pred_price = np.array(price_buckets)[np.array(df_preds_inv.loc[df_episode_index, "pred_buck"].astype(int))]
        afn_pred_price = np.array(price_buckets)[np.array(df_preds_afn.loc[df_episode_index, "pred_buck"].astype(int))]

        return afn_pred_price, inv_pred_price

    def get_market_features_labels(self, start_date, end_date, data_path):
        """
        Reads X.pq dataframe that contains features used to preict price.
        Data should be stored as parquet files.
        """
        X = pd.read_parquet(data_path.get('X')).loc[start_date:end_date]
        Y = pd.read_parquet(data_path.get('Y')).loc[start_date:end_date]
        Y_prices = pd.read_parquet(data_path.get('Y_prices')).loc[start_date:end_date]

        return X.join(Y).join(Y_prices)

    def get_imba_predictions(self, start_date, end_date, data_path):
        """
        Get the price forecasts for invoden and afenmen from a file.
        """
        preds_afn_ptes = dict()
        preds_inv_ptes = dict()
        # See which ptu and label it is 
        for name_df in data_path.keys():
            if 'preds' in name_df:
                if 'afn' in name_df:
                    # Always have the last character be a ptu number 
                    pte = int(name_df[-1])  
                    preds_afn_ptes[pte] = pd.read_parquet(data_path.get(name_df)).loc[start_date:end_date]
                elif 'inv' in name_df:
                    pte = int(name_df[-1])
                    preds_inv_ptes[pte] = pd.read_parquet(data_path.get(name_df)).loc[start_date:end_date]      
                else:
                    raise KeyError("Could not parse the dataframe's name to distinguish label (anf/inv) or/and PTU number.")

        return preds_afn_ptes, preds_inv_ptes

    def _degradation_update(self, charge):
        """
        Update all parameters that degrade. Degradataion is a linear cost funcion now.
        After 6000 cycles a battery's max storage is descreased by 20%.
        Thus, degradation per charge amount is (0.2 / 6000) * (charge_amount/(2*Soc_max))

        In his notes, Luuk had: 0.2 / 6000 * (bat_stor / 60.)
        """
        # Update SoC_max for degradation
        self.SoC_max = (
                self.SoC_max
                - self.SoC_max * (abs(charge) / self.full_eq_cycle) * self.deg_per_cycle)
        # Update full_eq_cycles for degradation
        self.full_eq_cycle = 2.0 * (self.SoC_max - self.SoC_min)
        # Update total cycles
        self.total_cycles += abs(charge) / self.full_eq_cycle
        # Update total cycles
        self.cycles_day += abs(charge) / self.full_eq_cycle

    def stored_energy_update(self, dis_charge, dis_charge_duration):
        """
        Charge or discharge battery based on signal:
           (Dis)charge: -1 is discharge & +1 is charge, time in seconds
        Automatically does degradation.
        Args:
            dis_charge_duration (int): how many seconds we are (dis)-charging
            dis_charge (int): (Dis)charge: -1 is discharge & +1 is charge,

        Returns:
            (dis)charge amount output/input (before efficiency loss)
        """
        # TODO: find out how efficiency loss is calculated exactly?
        # Discharge
        if dis_charge < 0:
            max_discharge = self.bat_stored - self.SoC_min
            discharge_bat = max(
                dis_charge * max_discharge,
                self.bat_cap
                / self.eff_AC_DC ** 0.5
                * dis_charge
                * dis_charge_duration
                / 3600.0)
            self.bat_stored += discharge_bat
            self._degradation_update(discharge_bat)
            discharge_out = discharge_bat * self.eff_AC_DC ** 0.5
            return discharge_out
        # Charge
        elif dis_charge > 0:
            max_charge = self.SoC_max - self.bat_stored
            charge_bat = min(
                dis_charge * max_charge,
                self.eff_AC_DC ** 0.5
                * self.bat_cap
                * dis_charge
                * dis_charge_duration
                / 3600.0)
            self.bat_stored += charge_bat
            self._degradation_update(charge_bat)
            charge_in = charge_bat / self.eff_AC_DC ** 0.5
            return charge_in
        else:
            return 0

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
            historic_preds_path=PATH_MARKET_FEATURES_PREDS):
        """
        Initialize battery system with different parameters

        Args:
            time_step_delta (int): number of minutes that will pass after a decision is made
            historic_prices_period (tuple): start and end datetime to use to load hystoric data
        """
        # Current time
        self.datetime_current = datetime.strptime(historic_prices_period[0], "%Y-%m-%d %H:%M:%S") \
            if (historic_prices_period[0] is not None) else datetime.strptime("0001-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        # When our episode finishes
        self.datetime_end = None
        # Step through time
        self.time_step_delta = time_step_delta
        # Arbitrary hour
        self.hour = None
        # Current minute
        self.minute = None
        # If we use historic prices or not
        self.history_data_dt_start = datetime.strptime(historic_prices_period[0], "%Y-%m-%d %H:%M:%S")
        self.history_data_dt_end = datetime.strptime(historic_prices_period[1], "%Y-%m-%d %H:%M:%S")
        # Train on this period
        self.run_dt_start = datetime.strptime(run_period[0], "%Y-%m-%d %H:%M:%S")
        self.run_dt_end = datetime.strptime(run_period[1], "%Y-%m-%d %H:%M:%S")
        # Dataframe for each episode
        self.df_episode = None
        # Battery stored energy MWh, must be updated at every (dis)charge
        self.bat_stored = bat_stor
        # Battery charge capacity MW - updated after degradation cycle?
        self.bat_cap = bat_cap
        # SoC minimum MWh
        self.SoC_min = soc_min
        # SoC maximum MWh, must be updated after degradation cycle
        self.SoC_max = soc_max
        # Initial SoC maximum MWh
        self.SoC_max_initial = soc_max
        # Get discrete energy levels (battery charge in percentage)
        self.energy_lvls = energy_lvls
        # Round trip eff: AC -> DC and DC -> AC
        self.eff_AC_DC = eff_ac_dc
        # Amount of cycles required to decrease SoC_max with 20% / 20%
        self.deg_per_cycle = deg_per_cycle
        # Maximum cycles per day
        self.max_cycles_day = max_cycles_day
        # Amount of cycles per day
        self.cycles_day = 0
        # Total amount of cycles
        self.total_cycles = 0
        # Full cycle is charge & discharge (MWh); must be updated after degradation cycle
        self.full_eq_cycle = 2.0 * (self.SoC_max - self.SoC_min)
        # What was price within a day
        self.price_function = None
        # If we have discrete price
        self.price_buckets_mid = price_buckets_mid
        # What our forecast was
        self.price_forecast = None
        # Sorted by price (lower to higher) hours within CE duration (first several are  optimal charging hours)
        self.optimal_ch_hours = None
        # Average price bought/sold
        self.mean_bought_price = None
        self.mean_sold_price = None
        # Put market features in a dataframe where indexes are datetime stamps
        self.historic_X_Y = historic_feats_path
        self.historic_Y_preds = historic_preds_path
        if (None or "0001-01-01 00:00:00") not in historic_prices_period:
            self.df = self.get_market_features_labels(start_date=self.history_data_dt_start,
                                                      end_date=self.history_data_dt_end,
                                                      data_path=self.historic_X_Y)
            self.preds_afn_ptes, self.preds_inv_ptes = self.get_imba_predictions(start_date=self.history_data_dt_start,
                                                                                       end_date=self.history_data_dt_end,
                                                                                       data_path=self.historic_Y_preds)
            # Set current time to the intitial historical datetime
            self.datetime_current = self.history_data_dt_start
        else:
            raise Exception("No date range for market features was given.")

        # Get mother class methods and variables
        super().__init__()
        # Initial state of the battery [market and physical features)
        self.state = None
        # Set 3 dimensional action space as discrete: charge, discharge, or idle
        self.action_space = spaces.Discrete(3)
        # Define state space
        # TODO: how to determine state_shape automatically?
        self.state_shape = (36 + 2,)  # (2 + 2 + len(POLICY_MARKET_FEATURES),)  # (len(POLICY_MARKET_FEATURES) + 1,)  #
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.state_shape, dtype=np.float64)
        # Give environment a name
        self.env_id = "Battery-System"

    def reset(self, current_energy_stored=None, datetime_dt=None, episode_len_ptu=96, test_time=False):
        """
        Start simulation with current_energy_stored as initial battery charge.
        Args:
            current_energy_stored (float): energy stored now
            episode_len_ptu (int): how many ptu an episode is
            test_time (bool): True indicates that we are at test time and do not need to account for end
                              of historic data, we keep an eye on it ourselves
        """
        self.datetime_current = datetime_dt or self.run_dt_start
        # Set episode length with the number of PTU's
        self.datetime_end = self.datetime_current + timedelta(minutes=episode_len_ptu * 15)
        # If we reached the end of historical data, reset to the beginning
        if (self.datetime_end >= self.run_dt_end - timedelta(minutes=episode_len_ptu * 15)) and (not test_time):
            self.datetime_current = self.run_dt_start
            self.datetime_end = self.datetime_current + timedelta(minutes=episode_len_ptu * 15)
        self.hour = self.datetime_current.hour
        self.minute = self.datetime_current.minute
        # Get/init battery features
        self.bat_stored = current_energy_stored or self.bat_stored
        # Create dataframes TimeIndices for an episode
        self.df_episode = pd.DataFrame(
            index=pd.date_range(start=self.datetime_current, end=self.datetime_end, freq="1T"))
        # Fill in feature for an episode, generate if we have nowhere to retreive features from
        if self.df is None:
            raise Exception("Can not create features for an episode as the dataframe with market features was not created.")
        else:
            # Market features
            self.df_episode[POLICY_MARKET_FEATURES] = self.df.loc[self.df_episode.index, POLICY_MARKET_FEATURES]
            # Needed for the reward function
            self.df_episode["invoeden"] = self.df.loc[self.df_episode.index, "settled_invoeden"]
            self.df_episode["afnemen"] = self.df.loc[self.df_episode.index, "settled_afnemen"]
            # Convert buckets to point predictictions: predictions bucket -> mid-price of the bucket
            afn_pred_price, inv_pred_price = self.get_mid_price(self.preds_afn_ptes[0], self.preds_inv_ptes[0],
                                                                self.df_episode.index, self.price_buckets_mid)
            self.df_episode["invoeden_pred_price_pte0"] = inv_pred_price
            self.df_episode["afnemen_pred_price_pte0"] = afn_pred_price
            # Get the predictions bucket
            self.df_episode["afnemen_pred_buck_pte0"] = self.preds_afn_ptes[0].loc[self.df_episode.index, "pred_buck"]
            self.df_episode["invoeden_pred_buck_pte0"] = self.preds_inv_ptes[0].loc[self.df_episode.index, "pred_buck"]
            # Track state of charge for plotting
            self.df_episode["SoC"] = np.nan
            self.df_episode.loc[self.datetime_current, "SoC"] = self.bat_stored
            # Record actions
            self.df_episode["action"] = np.nan
        # Initial mean bought/sold prices : Needed for reward fucntion
        self.mean_bought_price = self.df_episode["invoeden"].median()
        self.mean_sold_price = self.df_episode["afnemen"].median()

        self.state = np.concatenate([np.array([self.bat_stored, self.mean_bought_price, self.datetime_current.minute % 15, self.datetime_current.hour]),
                                     self.df_episode.loc[self.datetime_current, POLICY_MARKET_FEATURES].values,
                                     self.df_episode.loc[self.datetime_current, ["invoeden_pred_price_pte0", "afnemen_pred_price_pte0"]].values,
                                     self.preds_afn_ptes[0].loc[self.datetime_current, self.preds_afn_ptes[0].columns != "pred_buck"].values,
                                     self.preds_inv_ptes[0].loc[self.datetime_current, self.preds_inv_ptes[0].columns != "pred_buck"].values])

        return self.state

    def calculate_reward(self, action, charge_amount):
        """
        Assume the agent will now that it needs to accumulate negative rewards before being able to discharge.

        Note: we do the update on battery storage before calculating reward.
        """
        # Discharge
        if action == 0:
            reward = abs(charge_amount) * (
                    self.df_episode.loc[self.datetime_current, "invoeden"] - self.mean_bought_price)
                    # if (self.bat_stored + abs(charge_amount)) / self.SoC_max > 0.0 else -1
        # Charge
        elif action == 1:
            reward = abs(charge_amount) * (self.df_episode.loc[:, "afnemen"].mean() - self.df_episode.loc[self.datetime_current, "afnemen"]) * 2
            # self.mean_bought_price += abs(charge_amount) * (self.df_episode.loc[self.datetime_current, "afnemen"] - self.mean_bought_price)
        # Idle
        elif action == 2:
            reward = 0.15
        else:
            raise Exception("Unknown action.")

        return reward / self.time_step_delta
    
    def step(self, action):
        """
        Make a step in time. Get reward based on the action. See if the episode has ended.
        """
        # Record the action
        self.df_episode.loc[self.datetime_current, "action"] = action

        done = False
        # Change battery features.
        charge = ACTION_CHARGE_MAPPING[action]
        charge_amount = self.stored_energy_update(charge, self.time_step_delta * 60)

        # Calculate the reward for this time s (not for s')
        reward = self.calculate_reward(action, charge_amount)

        # Update timestamp
        self.datetime_current = self.datetime_current + timedelta(
            minutes=self.time_step_delta
        )
        # Tracking state of charge
        self.df_episode.loc[self.datetime_current, "SoC"] = self.bat_stored
        # We reached the end of an episode (no more prices availble)
        if self.datetime_current >= self.datetime_end:
            done = True

        self.state = np.concatenate([np.array([self.bat_stored, self.mean_bought_price, self.datetime_current.minute % 15, self.datetime_current.hour]),
                                     self.df_episode.loc[self.datetime_current, POLICY_MARKET_FEATURES].values,
                                     self.df_episode.loc[self.datetime_current, ["invoeden_pred_price_pte0", "afnemen_pred_price_pte0"]].values,
                                     self.preds_afn_ptes[0].loc[self.datetime_current, self.preds_afn_ptes[0].columns != "pred_buck"].values,
                                     self.preds_inv_ptes[0].loc[self.datetime_current, self.preds_inv_ptes[0].columns != "pred_buck"].values])

        return self.state, reward, done, 0


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

    # todo: now predictions for PTU1-6 is available from 2019-02-01 till 2019-10-01, make more data
    DATA_RANGE_PERIOD = ("2019-02-01 00:00:00", "2019-10-01 00:00:00")
    TRAIN_PERIOD = ("2019-02-01 00:00:00", "2019-09-01 00:00:00")

    env = BatterySystem(
        bat_stor,
        bat_cap,
        SoC_min,
        SoC_max,
        eff_AC_DC,
        deg_per_cycle,
        max_cycles_day,
        historic_prices_period=DATA_RANGE_PERIOD,
        run_period=TRAIN_PERIOD,
    )

    env.reset()
    env.step(1)
    env.step(2)
    env.step(0)

    print("Debugging.")

    ########################################################################
    # Used to dump future predictions so we can load them with a nice format
    ########################################################################

    # for i in range(1, 7):
    #     test = pd.read_parquet(f"/home/romaks/Studie/Thesis/pop/storage/nl_imba_hind/full_hindcast/results_pte{i}.pq")
    #     afn = pd.DataFrame(
    #         test['quarter_actual_afnemen_price_eurpmw_bucketized'].tolist(),
    #         columns=[x for x in range(len(PRICE_BUCKETS_MID) - 1)],
    #         index=test.index)
    #     afn['pred_bucket'] = afn.idxmax(axis=1)
    #     afn.columns = afn.columns.astype(str)
    #     afn.to_parquet(f"/home/romaks/Studie/Thesis/pop/storage/nl_imba_hind/full_hindcast/afnemen_preds_pte{i}.pq")
    #     inv = pd.DataFrame(
    #         test['quarter_actual_invoeden_price_eurpmw_bucketized'].tolist(),
    #         columns=[x for x in range(len(PRICE_BUCKETS_MID) - 1)],
    #         index=test.index)
    #     inv['pred_bucket'] = inv.idxmax(axis=1)
    #     inv.columns = inv.columns.astype(str)
    #     inv.to_parquet(f"/home/romaks/Studie/Thesis/pop/storage/nl_imba_hind/full_hindcast/invoeden_preds_pte{i}.pq")
