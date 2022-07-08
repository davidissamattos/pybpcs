from pandas.core.frame import DataFrame
import pytest
from pybpcs import bpc
from tests.testdata import *


class TestBt:
    def test_bt(self):
        m1 = bpc(
            data = data_bt(),
            player0 = 'player0',
            player1 = 'player1',
            result_column = 'y',
            model_type = 'bt',
            solve_ties = 'random',
            win_score = 'higher',
            iter = 1000,
            warmup = 300,
            show_chain_messages = False,
            seed = 8484
        )
        m1.fit()
        m1.summary()