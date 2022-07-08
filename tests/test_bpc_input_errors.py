from pandas.core.frame import DataFrame
import pytest
from pybpcs import bpc
from tests.testdata import *


class TestInputErrors:
    #has y but missing socres
    def test_missing_one_score_and_y(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_btscores(),
                player0="player0",
                player1="player1",
                player1_score="score1",
                model_type="bt",
                solve_ties="random",
                win_score="higher",
            )

    #should have both scores or y
    def test_missing_both_scores_and_y(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_btscores(),
                player0="player0",
                player1="player1",
                model_type="bt",
                solve_ties="random",
                win_score="higher",
            )

    #input should be a dataframe
    def test_input_not_a_dataframe(self):
        with pytest.raises(ValueError):
            bpc(
                data=[1, 2, 3],
                player0="player0",
                player1="player1",
                model_type="bt",
                solve_ties="random",
                win_score="higher",
            )

    #contains only 0 1 or 2 in the results column
    def test_results_values(self):
        d = {
            "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
            "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
            "y": [0, 0, 1, 0, 3, 0, 0, 0, 1],
        }
        with pytest.raises(ValueError):
            bpc(
                data=DataFrame(d),
                player0="player0",
                player1="player1",
                model_type="bt",
                solve_ties="random",
                win_score="higher",
            )

    #specify z1 but do not specify th emodel with order effect
    def test_ordereffect1(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_btorder(),
                player0="player0",
                player1="player1",
                result_column="y",
                z_player1="z1",
                model_type="bt",  # model is not bt-ordereffect
            )
    #specify model bt-ordereffect but not z1
    def test_ordereffect2(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_btorder(),
                player0="player0",
                player1="player1",
                result_column="y",
                model_type="bt-ordereffect",
            )

    #error if z_player1 has values different than 0 and 1
    def test_ordereffect3(self):
        d = {
            "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
            "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
            "y": [0, 0, 1, 0, 0, 0, 0, 0, 1],
            "z1": [1, 1, 0, 2, 1, 0, 0, 0, 0],
        }
        with pytest.raises(ValueError):
            bpc(
                data=DataFrame(d),
                player0="player0",
                player1="player1",
                z_player1="z1",
                result_column="y",
                model_type="bt-ordereffect",
            )
    #has clusters but model is not bt-U
    def test_randomeffects1(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_btU(),
                player0="player0",
                player1="player1",
                cluster=['cluster1'],
                result_column="y",
                model_type="bt",
            )
    #does not have clusters but model is bt-U
    def test_randomeffects2(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_btU(),
                player0="player0",
                player1="player1",
                result_column="y",
                model_type="bt-U",
            )
    #more than 3 clusters is not supported
    def test_randomeffects3(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_btU(),
                player0="player0",
                player1="player1",
                cluster=['cluster1', 'cluster2', 'cluster3','cluster4'],
                result_column="y",
                model_type="bt-U",
            )

    
    #no predictors but specify generalized
    def test_generalizedmodels1(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_bt(),
                player0="player0",
                player1="player1",
                result_column="y",
                model_type="bt-generalized",
            )

    #with predictors but specify no -generalized
    def test_generalizedmodels2(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_bt(),
                player0="player0",
                player1="player1",
                result_column="y",
                predictors=data_generalized_predictors(),
                model_type="bt",
            )
    #predictors is not a dataframe
    def test_generalizedmodels3(self):
        with pytest.raises(ValueError):
            bpc(
                data=data_bt(),
                player0="player0",
                player1="player1",
                result_column="y",
                predictors=[1,23,4,5],
                model_type="bt-generalized",
            )

    # def test_subjectpredictors(self):
    #     with pytest.raises(ValueError):
        

    # def test_davidson(self):
    #     with pytest.raises(ValueError):
