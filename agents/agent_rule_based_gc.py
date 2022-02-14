import pandas as pd


THR_AFN = 10
THR_INV = 90


class AgentRB:
    """
    Rules-based. Rules-based policy for steerting a battery.

    TODO: code policies that GC uses.
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

    def __init__(self, env, buffer, mid_merit_price_flag=True, th_afn=THR_AFN, th_inv=THR_INV, **kwargs):
        """
        This agent cannot be trained using experience replay - needs one episode.
        """
        self.env = env
        # self.price_buckets = env.price_buckets

        # Do we want to track mid merit price for thresholding
        self.mid_price_merit_flag = mid_merit_price_flag
        # Used to determine how high/low of mid_price we want to buy sell.
        self.price_threshold_delta_inv = 50
        self.price_threshold_delta_afn = 10

        # Set thresholds on when to charge or discharge
        self.th_price_afn = pd.Series(index=self.env.df.index, data=th_afn)
        self.th_price_inv = pd.Series(index=self.env.df.index, data=th_inv)
        self.agent_id = f"rule_based_th_gc"

        # Store current predictions
        self.inv_pred_price = None
        self.afn_pred_price = None

    def get_epsilon(self, iteration=None):
        """ Function does nothing. Here just to adjust to the style."""
        return None

    # TODO: SoC dependent pricing. Exponeital linking to SoC and th_price. Never want extremes in SoC (no empty or totally full battery).

    def select_action(self, state, dt_state, **kwargs):
        """
        Assume access to all information available at that time from the simulator (env).
        Select action based on point predictions.

        NOTE: agnet is created with environment. All prices/features for evaluation period
        should be loaded to that environment.

        Args:
            state (pandas.DataFrame): row with features
            dt_state (datetime.datetime):
        """
        # Check if we are working with bucket predictions
        inv_pred_price = self.env.df_episode.loc[dt_state, "invoeden_pred_price"]
        afn_pred_price = self.env.df_episode.loc[dt_state, "afnemen_pred_price"]

        ## ======= Testing/Debug =======
        self.inv_pred_price = inv_pred_price
        self.afn_pred_price = afn_pred_price
        ## =============================

        # Discharge
        if inv_pred_price > self.th_price_inv[dt_state]:
            return 0
        # Charge
        elif afn_pred_price < self.th_price_afn[dt_state]:
            return 1
        # Idle
        else:
            return 2

    def train(self, batch_size, **kwargs):
        """
        Does evaluation of the rule-based policy.

        TODO: perform optimization on which thresholds to use
        """
        return 0


if __name__ == "__main__":
    pass
