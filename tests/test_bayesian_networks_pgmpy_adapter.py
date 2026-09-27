from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from eyeprocesspy.bayesian_networks.backends import pgmpy_backend as backend
from eyeprocesspy.bayesian_networks.schema import BayesianConstraintSpec


class Graph:
    def __init__(self, edges=()):
        self._edges = list(edges)

    def edges(self):
        return list(self._edges)


class Discovery:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.causal_graph_ = Graph([("a", "b")])

    def fit(self, data):
        self.data = data
        return self


class ExpertKnowledge:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DiscreteModel:
    def __init__(self, edges):
        self.edges = list(edges)
        self._nodes = set(x for e in edges for x in e)
        self.fit_args = None
        self.cpds = []
        self.bayes_args = None

    def add_nodes_from(self, nodes):
        self._nodes.update(nodes)

    def fit(self, data, estimator=None, **kwargs):
        self.fit_args = (data.copy(), estimator, kwargs)
        return self

    def add_cpds(self, *cpds):
        self.cpds.extend(cpds)

    def check_model(self):
        return True

    def nodes(self):
        return list(self._nodes)


class GaussianModel(nx.DiGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.fit_args = None

    def fit(self, data, estimator=None, std_estimator=None):
        self.fit_args = (data.copy(), estimator, std_estimator)
        return self

    def check_model(self):
        return True

    def to_joint_gaussian(self):
        order = list(nx.topological_sort(self))
        mean = np.arange(len(order), dtype=float)
        cov = np.eye(len(order))
        if len(order) >= 2:
            cov[0, 1] = cov[1, 0] = 0.5
        return mean, cov

    def log_likelihood(self, data):
        return -float(len(data))


class BayesianEstimator:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def get_parameters(self, **kwargs):
        self.model.bayes_args = kwargs
        return ["cpd"]


class Factor:
    state_names = {"a": ["low", "high"]}
    values = np.array([0.25, 0.75])


class VE:
    def __init__(self, model):
        self.model = model

    def query(self, variables, evidence, show_progress=False):
        assert variables == ["a"]
        assert show_progress is False
        return Factor()


def fake_import(name):
    if name == "pgmpy.causal_discovery":
        return SimpleNamespace(
            ExpertKnowledge=ExpertKnowledge,
            HillClimbSearch=Discovery,
            PC=Discovery,
            GES=Discovery,
        )
    if name == "pgmpy.models":
        return SimpleNamespace(
            DiscreteBayesianNetwork=DiscreteModel,
            LinearGaussianBayesianNetwork=GaussianModel,
        )
    if name == "pgmpy.estimators":
        return SimpleNamespace(
            MaximumLikelihoodEstimator=object,
            BayesianEstimator=BayesianEstimator,
        )
    if name == "pgmpy.inference":
        return SimpleNamespace(VariableElimination=VE)
    if name == "pgmpy.metrics":
        return SimpleNamespace(log_likelihood_score=lambda model, data: -3.0)
    if name == "networkx":
        return nx
    raise ImportError(name)


def test_import_and_score_helpers(monkeypatch):
    monkeypatch.setattr(
        backend,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(ImportError, match="bayesnet"):
        backend._import("pgmpy")
    assert backend._score_name(None, "discrete") is None
    assert backend._score_name("bic", "discrete") == "bic-d"
    assert backend._score_name("aic", "gaussian") == "aic-g"
    assert backend._score_name("ll", "mixed") == "ll-cg"
    assert backend._score_name("bdeu", "discrete") == "bdeu"


def test_expert_and_structure_algorithms(monkeypatch):
    monkeypatch.setattr(backend, "import_module", fake_import)
    assert backend._expert_knowledge(None) is None
    constraints = BayesianConstraintSpec(
        temporal_tiers=(("a",), ("b",)),
        required_edges=(("a", "b"),),
        forbidden_edges=(("b", "a"),),
        max_parents=2,
    )
    expert = backend._expert_knowledge(constraints)
    assert expert.kwargs["required_edges"] == [("a", "b")]
    data = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    graph, edges, score = backend.learn_structure(
        data,
        family="discrete",
        algorithm="hill_climb",
        score="bic",
        constraints=constraints,
        max_parents=None,
    )
    assert isinstance(graph, Graph)
    assert edges == (("a", "b"),)
    assert score == "bic-d"
    _, _, pc_score = backend.learn_structure(
        data,
        family="discrete",
        algorithm="pc",
        score="bic",
        constraints=None,
        max_parents=1,
    )
    assert pc_score is None
    _, _, ges_score = backend.learn_structure(
        data,
        family="gaussian",
        algorithm="ges",
        score="bic",
        constraints=None,
        max_parents=None,
    )
    assert ges_score == "bic-g"
    with pytest.raises(NotImplementedError):
        backend.learn_structure(
            data,
            family="mixed",
            algorithm="pc",
            score="bic",
            constraints=None,
            max_parents=None,
        )
    with pytest.raises(ValueError):
        backend.learn_structure(
            data,
            family="discrete",
            algorithm="bad",
            score="bic",
            constraints=None,
            max_parents=None,
        )


def test_fit_parameters(monkeypatch):
    monkeypatch.setattr(backend, "import_module", fake_import)
    data_d = pd.DataFrame({"a": ["x", "y"], "b": ["y", "x"]})
    mle = backend.fit_parameters(
        edges=(("a", "b"),),
        nodes=["a", "b"],
        data=data_d,
        family="discrete",
        estimator="mle",
        prior="BDeu",
        equivalent_sample_size=10,
    )
    assert isinstance(mle, DiscreteModel)
    bayes = backend.fit_parameters(
        edges=(("a", "b"),),
        nodes=["a", "b"],
        data=data_d,
        family="discrete",
        estimator="bayesian",
        prior="BDeu",
        equivalent_sample_size=5,
    )
    assert bayes.bayes_args["equivalent_sample_size"] == 5
    assert bayes.bayes_args["prior_type"] == "BDeu"
    assert bayes.cpds == ["cpd"]
    with pytest.raises(ValueError):
        backend.fit_parameters(
            edges=(),
            nodes=["a"],
            data=data_d[["a"]],
            family="discrete",
            estimator="bad",
            prior="BDeu",
            equivalent_sample_size=1,
        )
    with pytest.raises(NotImplementedError):
        backend.fit_parameters(
            edges=(),
            nodes=["a"],
            data=data_d[["a"]],
            family="mixed",
            estimator="mle",
            prior="BDeu",
            equivalent_sample_size=1,
        )

    data_g = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 2.0]})
    g = backend.fit_parameters(
        edges=(("a", "b"),),
        nodes=["a", "b"],
        data=data_g,
        family="gaussian",
        estimator="mle",
        prior="BDeu",
        equivalent_sample_size=1,
    )
    assert isinstance(g, GaussianModel)
    assert g.fit_args[1] == "mle"
    with pytest.raises(ValueError):
        backend.fit_parameters(
            edges=(),
            nodes=["a"],
            data=data_g[["a"]],
            family="gaussian",
            estimator="bayesian",
            prior="BDeu",
            equivalent_sample_size=1,
        )


def test_query_discrete_and_gaussian(monkeypatch):
    monkeypatch.setattr(backend, "import_module", fake_import)
    d = DiscreteModel([("b", "a")])
    posterior, mean, var, method = backend.query(
        d,
        family="discrete",
        target="a",
        evidence={"b": "x"},
    )
    assert posterior == {"low": 0.25, "high": 0.75}
    assert mean is None and var is None
    assert method == "variable_elimination"
    with pytest.raises(ValueError):
        backend.query(d, family="discrete", target="a", evidence={"a": "x"})
    with pytest.raises(ValueError):
        backend.query(d, family="discrete", target="z", evidence={})
    with pytest.raises(ValueError):
        backend.query(d, family="discrete", target="a", evidence={"z": "x"})
    with pytest.raises(NotImplementedError):
        backend.query(d, family="mixed", target="a", evidence={})

    g = GaussianModel([("a", "b")])
    _, mean0, var0, method0 = backend.query(g, family="gaussian", target="a", evidence={})
    assert mean0 == 0.0
    assert var0 == 1.0
    assert method0 == "gaussian_conditioning"
    _, mean1, var1, _ = backend.query(
        g,
        family="gaussian",
        target="a",
        evidence={"b": 2.0},
    )
    assert mean1 == pytest.approx(0.5)
    assert var1 == pytest.approx(0.75)


def test_log_likelihood(monkeypatch):
    monkeypatch.setattr(backend, "import_module", fake_import)
    d = DiscreteModel([("a", "b")])
    data = pd.DataFrame({"a": ["x"], "b": ["y"]})
    assert (
        backend.log_likelihood(
            d,
            family="discrete",
            data=data,
            nodes=["a", "b"],
        )
        == -3.0
    )
    g = GaussianModel([("a", "b")])
    data_g = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 2.0]})
    assert (
        backend.log_likelihood(
            g,
            family="gaussian",
            data=data_g,
            nodes=["a", "b"],
        )
        == -2.0
    )
    with pytest.raises(NotImplementedError):
        backend.log_likelihood(
            d,
            family="mixed",
            data=data,
            nodes=["a", "b"],
        )
