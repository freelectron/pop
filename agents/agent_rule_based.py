import pandas as pd


class AgentRB:
    """
    Rules-based. Rules-based policy for steerting a battery.
    Incorporates only rule-based agent which creates thresholds on mid-price and becomes more aggressive with
    higher SoC.
    """

    @staticmethod
    def _create_base_threshold(df):
        """
        Couple the threshold at which we charge/discharge to the merit order mid-price.
        Args:
            df (pandas.DataFrame): [frequency of timeindex: 1T] contains all TenneT merit order feauters
        """
        # merit_posmin and merit_minmin should not have NaNs
        return df.minute_mid_prijs_opregelen, df.minute_mid_prijs_opregelen

    def __init__(self, env, buffer, price_delta_inv=30, price_delta_afn=10, exp_th_delta_soc=True, **kwargs):
        """
        This agent cannot be trained using experience replay - needs one episode.
        """
        self.env = env
        self.price_buckets = env.price_buckets_mid

        # If we want to make price dependent on state of charge
        self.exp_th_delta_soc = exp_th_delta_soc

        # Used to determine how high/low of mid_price we want to buy sell.
        self.price_threshold_delta_inv = price_delta_inv
        self.price_threshold_delta_afn = price_delta_afn

        # Threshold prices are determined by mid_price +/- threshold_delta
        th_base_price_afr, th_base_price_opr = AgentRB._create_base_threshold(env.df)
        self.th_price_afn = pd.Series(index=self.env.df.index, data=th_base_price_afr)
        self.th_price_inv = pd.Series(index=self.env.df.index, data=th_base_price_opr)

        # Store current predictions
        self.inv_pred_price = None
        self.afn_pred_price = None

        self.agent_id = f"rule_based_mid_price"

    def get_epsilon(self, iteration=None):
        """ Function does nothing. Here just to adjust to the style."""
        return None

    # TODO: SoC dependent pricing. Exponeital linking to SoC and th_price. Never want extremes in SoC (no empty or totally full battery).

    def select_action(self, state, dt_state, epsilon=None, batch_size=1, sequence_length=1):
        """
        Assume access to all information available at that time from the simulator (env).
        Select action based on point predictions.
        NOTE: agnet is created with environment. All prices/features for evaluation period
        should be loaded to that environment.
        Args:
            state (pandas.DataFrame): row with features
            dt_state (datetime.datetime): timestamp which of the decision
            epsilon (float): not used
            batch_size (int): not used
            sequence_length (int): not used
        """
        # Check if we are working with bucket predictions
        inv_pred_price = self.env.df_episode.loc[dt_state, "invoeden_pred_price_pte0"]
        afn_pred_price = self.env.df_episode.loc[dt_state, "afnemen_pred_price_pte0"]

        ## ======= Testing/Debug =======
        self.inv_pred_price = inv_pred_price
        self.afn_pred_price = afn_pred_price
        ## =============================

        # epx

        # Discharge
        if inv_pred_price > self.th_price_inv[dt_state] + (1-self.exp_th_delta_soc) * self.price_threshold_delta_inv +\
                self.price_threshold_delta_inv**(self.env.bat_stored) * self.exp_th_delta_soc:
            return 0
        # Charge
        elif afn_pred_price < self.th_price_afn[dt_state] + (1-self.exp_th_delta_soc) * self.price_threshold_delta_afn +\
                self.price_threshold_delta_afn**(self.env.bat_stored) * self.exp_th_delta_soc:
            return 1
        # Idle
        else:
            return 2

    def train(self, batch_size, **kwargs):
        """
        Does evaluation of the rule-based policy.
        TODO: perform search on which thresholds to use
        """
        return 0


if __name__ == "__main__":
    pass
