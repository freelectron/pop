from abc import ABC
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Dict

import numpy as np
import pandas as pd
import gym
from gym import spaces

from utils.imba.tennet import fetch_imba_1, fetch_imba_merit, fetch_imba_settled


ACTION_CHARGE_MAPPING = {
    0: -1,  # DISCHARGE
    1: 1,  # CHARGE
    2: 0,  # IDLE
}
PRICE_BUCKETS = [-1000, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
# Add epsilon later so that if battary 100 , we have lvl 10, instead of 11 (see how np.digitize works)
ENERGY_LVLS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 + 1e-3]

# Testing data for two months
HISTRORIC_PREDS_TESTING = {
    "afnemen": "./data/imba_small/imba_preds_2018_afnemen.csv",
    "invoeden": "./data/imba_small/imba_preds_2018_invoeden.csv",
}
# FEATS also contain targets (settled inv/afn)
HISTRORIC_FEATS_TESTING = {
    "minute": "./data/imba_small/tennet_deltas_2018.csv",
    "merit": "./data/imba_small/tennet_merit_2018.csv",
    "settled": "./data/imba_small/tennet_settled_2018.csv",
}


class BatterySystem(gym.core.Env, ABC):
    """
    State representation: VECTOR

    Simulate the behaviour of a simple battery system.

    I unite battery features (SoC, switching costs) with market features (imbalance price). There are two options
    to get market features: simulate them or load historical prices.
    """

    PARAMETERS = {}

    @staticmethod
    def discretize_array(array_nums, bucket_specs):
        """
        Discretizes each element in the `array_nums` according to the bucket specs.
        If bucks specs [-1000, -20, -10, 0, 10] and the array is [-22,-15,-2,0], the results is np.array([1, 2, 3, 4]).

        Note: No zero bucket (buckets start with 1).

        Args:
            array_nums (numpy.array or list): array with conitnuous elemets.
            bucket_specs (list): contains all the buckets. There are  len(list)-1 buckets.

        Returns:
            np.array, array bucketized
        """
        return np.digitize(array_nums, bucket_specs)

    @staticmethod
    def get_discrete_array(mode_specs=None, n_ptus=None, price_lowest=20):
        """
        Used to simulate prices for a charging episode. Prices are still continuous.

        Get a mixture of two Gaussians defined by `modes`.
        Arguments:
            n_ptus (int): 24 because generating price for a day.
            mode_specs (list): of lists that specify the normal distribution's peak (mean), std, and weight applied to it when mixing.
            price_lowest (int/float): lowest price for the period
        """
        if mode_specs is None:
            mode_specs = [(8, 3, 1), (16, 2, 0.5)]

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

        arr = []
        x_values = np.linspace(0, n_ptus, n_ptus)
        for mu, sig, weight in mode_specs:
            arr.append(weight * gaussian(x_values, mu, sig))
        # N model price (see Appendix how it looks like)
        summed = np.array(sum(arr))

        # Get the price for the whole day
        return (summed + price_lowest) * (1 + summed)

    @staticmethod
    def get_random_discrete_array(n_modes=2, n_ptus=4 * 60):
        """
        Used to simulate prices for a charging episode.

        Get a random mixture of `n_modes` Gaussians.
        """
        # Sample randomly two modes
        modes = np.trunc(np.random.uniform(0, 1, n_modes) * n_ptus)
        # 7 is arbitrary
        stds = np.trunc(np.random.uniform(0, 1, n_modes) * n_ptus / (n_ptus / 35))
        # This is only for two means
        r = np.random.random(n_modes)
        weights = r / sum(r)  # r
        mode_specs = zip(modes, stds, weights)

        return BatterySystem.get_discrete_array(list(mode_specs), n_ptus=n_ptus)

    @staticmethod
    def get_n_lowest(arr, n):
        """
        Get indices for n lowest elements in arr.
        """
        arr = arr.copy()
        indices = list()
        for i in range(n):
            ind = arr.argmin()
            indices.append(ind)
            arr[ind] = 9999999

        return indices

    @staticmethod
    def pick_n_lowest(arr, n):
        return np.sort(arr)[:n]

    def get_imba_data(
        self,
        start_date: datetime,
        end_date: datetime,
        days_ago_limit: int = 365 * 4,
        fake_data: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Loads tennet feautures and settled prices (targets).

        Args:
            start_date: When beginning all data should be
            end_date: End time
            days_ago_limit: How many days ago to limit the start_date (to avoid huge API call)
            fake_data: dictionary with specified paths to local/fake data
                    {'minute': fake_csv_minute,'merit': fake_csv_merit, 'settled': fake_csv_settled, 'apx': fake_apx_minute}

        Returns:
            Dataframe with data from all sources we require

        Raises:
            Warning if start_date is longer than days_ago_limit ago
        """
        period = (start_date, end_date)
        # We need to retrieve relative to now
        start_days_ago = (datetime.now() - start_date).days
        start_date = start_date.replace(tzinfo=timezone.utc)
        end_date = end_date.replace(tzinfo=timezone.utc)

        if start_days_ago > days_ago_limit:
            raise Warning(
                f"start_date is longer than {days_ago_limit} days ago. Increase days_ago_limit"
            )

        # Add two days just to be sure we cover the start date
        if fake_data:
            df_tennet_1 = fetch_imba_1(*period, fake_csv=fake_data.get("minute"))
            df_tennet_merit = fetch_imba_merit(*period, fake_csv=fake_data.get("merit"))
            df_tennet_settled = fetch_imba_settled(
                *period, fake_csv=fake_data.get("settled")
            )
        else:
            df_tennet_1 = fetch_imba_1(*period, fake_csv=fake_data)
            df_tennet_merit = fetch_imba_merit(*period, fake_csv=fake_data)
            df_tennet_settled = fetch_imba_settled(*period, fake_csv=fake_data)

        # Make new dataframe with minute index
        df = pd.DataFrame(
            index=pd.date_range(
                df_tennet_1.index.min(), df_tennet_1.index.max(), freq="1T"
            )
        )
        # Join all after upsampling to 1-minute and forward filling
        df = (
            df.join(df_tennet_1.asfreq("T"))
            .join(df_tennet_settled.asfreq("T", method="ffill"))
            .join(df_tennet_merit.asfreq("T", method="ffill"))
        )

        return df

    def get_imba_forecast(
        self,
        start_date,
        end_date,
        fake_data=HISTRORIC_PREDS_TESTING,
        days_ago_limit=None,
    ):
        """
        Get the price forecasts for invoden and afenmen from a file.
        """
        #  I load predictions that are already UTC
        df_preds_afn = pd.read_csv(fake_data["afnemen"], decimal=",", index_col=0)
        df_preds_inv = pd.read_csv(fake_data["invoeden"], decimal=",", index_col=0)
        # So far we have been doing everything in UTC
        df_preds_afn.index = pd.to_datetime(df_preds_afn.index, utc=True).tz_convert(
            "Europe/Amsterdam"
        )  # format='%Y-%m-%d %H:%M:%S'
        df_preds_inv.index = pd.to_datetime(df_preds_inv.index, utc=True).tz_convert(
            "Europe/Amsterdam"
        )
        # Get desired freq
        df_preds_afn.index.freq = "1T"
        df_preds_inv.index.freq = "1T"
        # Convert to UTC
        df_preds_afn.index = df_preds_afn.index.tz_convert(None)
        df_preds_inv.index = df_preds_inv.index.tz_convert(None)

        return df_preds_afn.loc[start_date:end_date], df_preds_inv.loc[start_date:end_date]

    def _degradation_update(self, charge):
        """
        Update all parameters that degrade. Degradataion is a linear cost funcion now.
        After 6000 cycles a battery's max storage is descreased by 20 %.
        Thus, degradation per charge amount is (0.2 / 6000) * (charge_amount/(2*Soc_max))

        In his notes, Luuk had: 0.2 / 6000 * (bat_stor / 60.)
        """
        # Update SoC_max for degradation
        self.SoC_max = (
            self.SoC_max
            - self.SoC_max * (abs(charge) / self.full_eq_cycle) * self.deg_per_cycle
        )
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
                / 3600.0,
            )
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
                / 3600.0,
            )
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
        time_step_delta=15,
        normalizer=1.0,
        discrete_states=False,
        price_buckets=PRICE_BUCKETS,
        energy_lvls=ENERGY_LVLS,
        historic_prices_period=("0001-01-01 00:00:00", "2525-01-01 00:00:00"),
        historic_feats_path=HISTRORIC_FEATS_TESTING,
        historic_preds_path=HISTRORIC_PREDS_TESTING,
    ):
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
        self.history_data_dt_start = datetime.strptime(
            historic_prices_period[0], "%Y-%m-%d %H:%M:%S"
        )
        self.history_data_dt_end = datetime.strptime(
            historic_prices_period[1], "%Y-%m-%d %H:%M:%S"
        )
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

        # How many PTU's ahead we can predict: prediction moving window
        self.prediction_window_ptu = 4
        # Controls that we can not charge more tha n_ch_pte
        self.can_charge = True
        # Amount of noise to add to the price to produce price forecast
        self.amount_noise = 4
        # What was price within a day
        self.price_function = None
        # If we have discrete price
        self.price_buckets = price_buckets
        # What our forecast was
        self.price_forecast = None
        # Sorted by price (lower to higher) hours within CE duration (first several are  optimal charging hours)
        self.optimal_ch_hours = None
        # Get normalized inverted prices
        self.price_range_ch_rate_fix = None
        # Flag whether to give price buckets to state representation
        self.discrete_states = discrete_states
        self.normalizer = normalizer
        # Average price bought/sold
        self.mean_bought_price = None
        self.mean_sold_price = None
        # Update rates for them
        self.alpha_mean_sold_price = 0.01
        self.alpha_mean_bought_price = 0.01

        # Put market features in a dataframe where indexes are datetime stamps
        self.historic_X = historic_feats_path
        self.historic_Y = historic_preds_path
        if (None or "0001-01-01 00:00:00") not in historic_prices_period:
            # Histroical real data
            self.df = self.get_imba_data(
                start_date=self.history_data_dt_start,
                end_date=self.history_data_dt_end,
                fake_data=self.historic_X,
            )
            self.df_forecast_afn, self.df_forecast_inv = self.get_imba_forecast(
                start_date=self.history_data_dt_start,
                end_date=self.history_data_dt_end,
                fake_data=self.historic_Y,
            )
            # Set current time to the intitial historical datetime
            self.datetime_current = self.history_data_dt_start
        else:
            # If no historical data is given, generate prices on the flywut
            self.df = None

        # Get mother class methods and variables
        super().__init__()
        # Initial state of the battery [market and physical features)
        self.state = None
        # Set 3 dimensional action space as discrete: charge, discharge, or idle
        self.action_space = spaces.Discrete(3)
        # Define state space
        # TODO: how to determine state_shape automatically?
        self.state_shape = (2,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.state_shape, dtype=np.float64
        )

        # Give environment a name
        self.env_id = "Battery-System"

    def reset(self, current_energy_stored=None, datetime_dt=None, episode_len_ptu=96):
        """
        Start simulation with current_energy_stored as initial battery charge.

        Args:
            current_energy_stored (float): energy stored now
            datetime_str (string: to specify what date and time of day we are in (used in market simulation or
                                             to retrieve historical prices. FORMAT: %y-%m-%d %H:%M:%S
            episode_len_ptu (int): how many ptu an episode is
        """
        self.datetime_current = datetime_dt or self.history_data_dt_start
        # Set episode length with the number of PTU's
        self.datetime_end = self.datetime_current + timedelta(minutes=episode_len_ptu * 15)
        # If we reached the end of historical data, reset to the beginning
        if self.datetime_end >= self.history_data_dt_end - timedelta(days=3):
            self.datetime_current = self.history_data_dt_start
            self.datetime_end = self.datetime_current + timedelta(minutes=episode_len_ptu * 15)

        self.hour = self.datetime_current.hour
        self.minute = self.datetime_current.minute
        # Get/init battery features
        self.bat_stored = current_energy_stored or self.bat_stored

        # Create dataframes TimeIndices for an episode
        self.df_episode = pd.DataFrame(
            index=pd.date_range(
                start=self.datetime_current, end=self.datetime_end, freq="1T"
            )
        )
        # Fill in feature for an episode, generate if we have nowhere to retreive features from
        if self.df is None:
            self.price_function = self.get_discrete_array(
                n_ptus=episode_len_ptu
            )  # self.get_random_discrete_array(n_ptus=4*self.time_step_delta) #
            if self.discrete_states:
                self.price_forecast = self.discretize_array(
                    self.price_function, self.price_buckets
                )
            else:
                self.price_forecast = self.price_function
            # Get the forecasted prices
            self.optimal_ch_hours = self.hour + np.array(
                self.get_n_lowest(
                    self.price_function[
                        self.hour : self.hour + self.prediction_window_ptu
                    ],
                    self.prediction_window_ptu,
                )
            )
            # Fill in the dataframe
            self.df_episode["invoeden"] = self.price_function
            self.df_episode["afnemen"] = self.price_function
            # Get the forecasts: can be point values or bucketized (argmax)
            self.df_episode["invoeden_pred"] = self.price_forecast
            self.df_episode["afnemen_pred"] = self.price_forecast
            # Track state of charge for plotting
            self.df_episode["SoC"] = np.nan
            self.df_episode.loc[self.datetime_current, "SoC"] = self.bat_stored
            # Record actions
            self.df_episode["action"] = np.nan
        else:
            self.df_episode["invoeden"] = self.df.loc[self.df_episode.index, "settled_invoeden"]
            self.df_episode["afnemen"] = self.df.loc[self.df_episode.index, "settled_afnemen"]
            # Convert buckets to point predictictions: predictions bucket -> mid-price of the bucket
            inv_pred_price = (np.array(self.price_buckets)[self.df_forecast_inv.loc[self.df_episode.index, "pred_buck"].values]
                              + np.array(self.price_buckets)[self.df_forecast_inv.loc[self.df_episode.index, "pred_buck"].values-1]) / 2.0
            afn_pred_price = (np.array(self.price_buckets)[self.df_forecast_afn.loc[self.df_episode.index, "pred_buck"].values]
                              + np.array(self.price_buckets)[self.df_forecast_afn.loc[self.df_episode.index, "pred_buck"].values-1]) / 2.0
            self.df_episode["invoeden_pred"] = inv_pred_price
            self.df_episode["afnemen_pred"] = afn_pred_price
            # Get the predictions bucket
            self.df_episode["invoeden_pred_buck"] = self.df_forecast_inv.loc[self.df_episode.index, "pred_buck"]
            self.df_episode["afnemen_pred_buck"] = self.df_forecast_afn.loc[self.df_episode.index, "pred_buck"]
            # Track state of charge for plotting
            self.df_episode["SoC"] = np.nan
            self.df_episode.loc[self.datetime_current, "SoC"] = self.bat_stored
            # Record actions
            self.df_episode["action"] = np.nan

        # Initial mean bought/sold prices
        self.mean_bought_price = self.df_episode["invoeden"].mean()
        self.mean_sold_price = self.df_episode["afnemen"].mean()

        # Create state vector
        bat_lvl = (
            self.bat_stored
            if not self.discrete_states
            else np.digitize(self.bat_stored / self.SoC_max, self.energy_lvls)
        )
        self.state = np.concatenate(
            (
                np.array([bat_lvl]),
                np.array([self.df_episode.loc[self.datetime_current, "invoeden_pred"]]),
            )
        )
        return self.state

    def calculate_reward(self, action, charge_amount):
        """
        Assume the agent will now that it needs to accumulate negative rewards before being able to discharge.

        Note: we do the update on battery storage before calculating reward.
        """
        # Discharge
        if action == 0:
            reward = abs(charge_amount) * (self.df_episode.loc[self.datetime_current, "invoeden"] - self.mean_bought_price)
            self.mean_sold_price += self.alpha_mean_sold_price * self.df_episode.loc[self.datetime_current, "invoeden"] * (abs(charge_amount) > 0)
        # Charge
        elif action == 1:
            reward = abs(charge_amount) * (self.mean_sold_price - self.df_episode.loc[self.datetime_current, "afnemen"])
            self.mean_bought_price += self.alpha_mean_bought_price * self.df_episode.loc[self.datetime_current, "afnemen"] * (abs(charge_amount) > 0)
        # Idle
        elif action == 2:
            reward = 0.1
        else:
            raise Exception("Unknown action/charge amount.")

        return reward

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

        ############## Testing Correctness ##############
        self.charge_amount = charge_amount
        if charge_amount < 0 and action != 0:
            raise Exception("ERROR")
        if charge_amount > 0 and action != 1:
            raise Exception("ERROR")
        if action == 2 and charge_amount != 0:
            raise Exception("ERROR")

        if self.bat_stored > self.SoC_max:
            raise Exception("ERROR")
        if self.bat_stored < 0:
            raise Exception("ERROR")
        #################################################

        # Update timestamp
        self.datetime_current = self.datetime_current + timedelta(
            minutes=self.time_step_delta
        )
        # Tracking state of charge
        self.df_episode.loc[self.datetime_current, "SoC"] = self.bat_stored
        # We reached the end of an episode (no more prices availble)
        if self.datetime_current >= self.datetime_end:
            done = True

        # Create state vector
        bat_lvl = (
            self.bat_stored
            if not self.discrete_states
            else np.digitize(self.bat_stored / self.SoC_max, self.energy_lvls)
        )
        self.state = np.concatenate(
            (
                np.array([bat_lvl]),
                np.array([self.df_episode.loc[self.datetime_current, "invoeden_pred"]]),
            )
        )

        return self.state, reward, done, 0


if __name__ == "__main__":

    bat_stor = 0.0
    bat_cap = 10.0
    SoC_min = 0.0
    SoC_max = 10000.0
    eff_AC_DC = 1.0
    bat_SoC = SoC_max
    deg_per_cycle = 0
    max_cycles_day = 100

    env = BatterySystem(
        bat_stor,
        bat_cap,
        SoC_min,
        SoC_max,
        eff_AC_DC,
        deg_per_cycle,
        max_cycles_day,
        historic_prices_period=("2018-01-01 00:00:00", "2018-03-01 00:00:00"),
    )
    env.reset()
    env.step(1)
    env.step(2)
    env.step(0)

    print("Debugging.")

    ################## Testing ##########################
    #####################################################
