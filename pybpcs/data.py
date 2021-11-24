import pandas as pd
from importlib import resources


def tennis_agresti():
    with resources.path("pybpcs.csv", "tennis_agresti.csv") as df:
        return pd.read_csv(df)


def data_bt():
    d = {
        "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
        "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
        "y": [0, 0, 1, 0, 0, 0, 0, 0, 1],
    }
    return pd.DataFrame(d)


def data_btscores():
    d = {
        "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
        "score0": [3, 2, 1, 3, 3, 4, 3, 2, 0],
        "score1": [1, 0, 2, 0, 2, 3, 2, 1, 1],
        "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
    }
    return pd.DataFrame(d)


def data_davidson():
    d = {
        "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
        "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
        "y": [0, 0, 2, 0, 0, 0, 0, 0, 2],
    }
    return pd.DataFrame(d)


def data_davidsonscores():
    d = {
        "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
        "score0": [3, 2, 1, 3, 3, 4, 3, 2, 0],
        "score1": [1, 0, 1, 0, 2, 3, 2, 1, 0],
        "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
    }
    return pd.DataFrame(d)


def data_btorder():
    d = {
        "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
        "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
        "y": [0, 0, 1, 0, 0, 0, 0, 0, 1],
        "z1": [1, 1, 0, 1, 1, 0, 0, 0, 0],
    }
    return pd.DataFrame(d)


def data_davidsonorder():
    d = {
        "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
        "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
        "y": [0, 0, 2, 0, 0, 0, 0, 0, 2],
        "z1": [1, 1, 0, 1, 1, 0, 0, 0, 0],
    }
    return pd.DataFrame(d)


def data_btU():
    d = {
        "player0": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
        ],
        "player1": [
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
        ],
        "y": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
        ],
        "cluster1": [
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
        ],
        "cluster2": [
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
        ],
        "cluster3": [
            "s5",
            "s5",
            "s5",
            "s5",
            "s5",
            "s5",
            "s5",
            "s5",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s3",
            "s3",
            "s3",
            "s3",
            "s3",
            "s3",
            "s3",
            "s3",
            "s2",
            "s2",
            "s2",
            "s2",
            "s2",
            "s2",
            "s1",
            "s1",
            "s1",
            "s1",
            "s1",
        ],
    }
    return pd.DataFrame(d)


def data_davidsonU():
    d = {
        "player0": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
        ],
        "player1": [
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
        ],
        "y": [
            0,
            0,
            2,
            0,
            0,
            0,
            2,
            1,
            1,
            0,
            1,
            1,
            2,
            1,
            2,
            0,
            1,
            1,
            0,
            0,
            2,
            2,
            1,
            1,
            0,
            2,
            1,
            0,
            2,
            1,
            0,
            1,
            1,
            2,
            1,
            1,
        ],
        "cluster1": [
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
        ],
        "cluster2": [
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p1",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p2",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
            "p3",
        ],
        "cluster3": [
            "s5",
            "s5",
            "s5",
            "s5",
            "s5",
            "s5",
            "s5",
            "s5",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s4",
            "s3",
            "s3",
            "s3",
            "s3",
            "s3",
            "s3",
            "s3",
            "s3",
            "s2",
            "s2",
            "s2",
            "s2",
            "s2",
            "s2",
            "s1",
            "s1",
            "s1",
            "s1",
            "s1",
        ],
    }
    return pd.DataFrame(d)


def data_generalized_predictors():
    d = {
        "Player": ["A", "B", "C"],
        "Pred1": [2.3, 4.2, 1.4],
        "Pred2": [-3.2, -2.1, 0.5],
        "Pred3": [0.01, 0.02, 0.04],
        "Pred4": [-1 / 2, -0.3, -0.2],
    }
    return pd.DataFrame(d)


def data_btUorder():
    d = {
        "player0": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
        ],
        "player1": [
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
        ],
        "y": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
        ],
        "cluster": [
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c1",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c2",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c3",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
            "c4",
        ],
        "z1": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    }
    return pd.DataFrame(d)


def bt_subject():
    d = {
        "player0": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
        "player1": ["B", "B", "B", "C", "C", "C", "C", "C", "C"],
        "y": [0, 0, 1, 0, 0, 0, 0, 0, 1],
        "SPred1": [1, 4, 2, 1, 4, 2, 1, 4, 2],
        "SPred2": [2, 0, 3, 2, 0, 3, 2, 0, 3],
        "SPred3": [3, 2, 3, 3, 2, 3, 3, 2, 3],
        "Subject": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
    return pd.DataFrame(d)

