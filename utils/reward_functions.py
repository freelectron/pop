#################################
# All ideas for reward functions
#################################

# Order: from ASSUMED best to worst


def calculate_reward(self, action, charge_amount):
    """
    Assume the agent will now that it needs to accumulate negative rewards before being able to discharge.

    Note: we do the update on battery storage before calculating reward.
    """
    # Discharge
    if action == 0:
        reward = abs(charge_amount) * (
                self.df_episode.loc[self.datetime_current, "invoeden"] - self.mean_bought_price)
        self.mean_sold_price += abs(charge_amount) * (self.df_episode.loc[
                                                          self.datetime_current, "invoeden"] - self.mean_sold_price)
    # Charge
    elif action == 1:
        reward = abs(charge_amount) * (self.mean_sold_price - self.df_episode.loc[self.datetime_current, "afnemen"])
        self.mean_bought_price += abs(charge_amount) * (self.df_episode.loc[
                                                            self.datetime_current, "afnemen"] - self.mean_bought_price)
    # Idle
    elif action == 2:
        reward = 0.01
    else:
        raise Exception("Unknown action/charge amount.")


def calculate_reward_mean(self, action, charge_amount):
    """
    Assume the agent will now that it needs to accumulate negative rewards before being able to discharge.
    """
    # Discharge
    if action == 0:
        # Add charge amount because we do the update on battery storage before calculating reward
        reward = abs(charge_amount) * self.df_episode.loc[self.datetime_current, "invoeden"] / self.df.loc[:,
                                                                                               "settled_invoeden"].mean()
        return reward
    # Charge
    elif action == 1:
        reward = - abs(charge_amount) * self.df_episode.loc[self.datetime_current, "afnemen"] / self.df.loc[:,
                                                                                                "settled_afnemen"].mean()
        return reward
    # Idle
    elif action == 2:
        return 0.0
    else:
        raise Exception("Unknown action/charge amount.")


def calculate_reward_historic_mean(self, action, charge_amount):
    """
    Rewaed agent on based on the mean price for the whole period loaded.

    Note: only works with historical prices.
    """
    # Discharge
    if action == 0:
        # Add charge amount because we do the update on battery storage before calculating reward
        reward = abs(charge_amount) * (
                    self.df_episode.loc[self.datetime_current, "invoeden"] - self.df.loc[:, "settled_invoeden"].mean()) \
            if (self.bat_stored + abs(charge_amount)) / self.SoC_max > 0.0 else -10
        return reward
    # Charge
    elif action == 1:
        reward = abs(charge_amount) * (
                    self.df.loc[:, "settled_afnemen"].mean() - self.df_episode.loc[self.datetime_current, "afnemen"]) \
            if (self.bat_stored - charge_amount) / self.SoC_max < 1.0 else -10
        return reward
    # Idle
    elif action == 2:
        return 0
    else:
        raise Exception("Unknown action/charge amount.")


def calculate_reward_episodic_mean(self, action, charge_amount):
    """
    Rewaed agent on based on the mean price of the episode.

    TODO: Complicated logic? error prone, simplify?
    """
    # Discharge
    if action == 0:
        # Add charge amount because we do the update on battery storage before calculating reward
        reward = (
            (
                abs(charge_amount)
                * (
                        self.df_episode.loc[self.datetime_current, "invoeden"]
                        - self.df_episode.loc[:, "invoeden"].mean()
                )
            )
            if ((self.bat_stored + abs(charge_amount)) / self.SoC_max > 0.0)
            else -10
        )
        return reward
    # Charge
    elif action == 1:
        reward = (
            (
                abs(charge_amount)
                * (
                        self.df_episode.loc[:, "afnemen"].mean()
                        - self.df_episode.loc[self.datetime_current, "afnemen"]
                )
            )
            if ((self.bat_stored - charge_amount) / self.SoC_max < 1.0)
            else -10
        )
        return reward
    # Idle
    elif action == 2:
        return 0
    else:
        raise Exception("Unknown action/charge amount.")


