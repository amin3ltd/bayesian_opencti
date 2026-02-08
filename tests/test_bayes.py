from bayes.model import BayesianConfidenceModel
def test_small_network():
    m = BayesianConfidenceModel(max_parents=3)
    m.add_or_update_node("A","Indicator","A",80)
    m.add_or_update_node("B","Malware","B",20)
    m.add_or_update_edge("A","B",70)
    probs = m.infer_all()
    assert probs["A"] > 0.75 and probs["B"] >= 0.20

def test_cyclic_network():
    m = BayesianConfidenceModel(max_parents=3, damping=0.55, epsilon=1e-6, max_iters=1000)
    m.add_or_update_node("A","Indicator","A",50)
    m.add_or_update_node("B","Malware","B",50)
    m.add_or_update_edge("A","B",60)
    m.add_or_update_edge("B","A",60)
    probs = m.infer_all()
    # Symmetric cycle with log-odds: no net evidence, stays at prior
    assert abs(probs["A"] - 0.5) < 0.01
    assert abs(probs["B"] - 0.5) < 0.01

def test_chain_network():
    m = BayesianConfidenceModel(max_parents=3)
    m.add_or_update_node("A","Indicator","A",80)
    m.add_or_update_node("B","Malware","B",20)
    m.add_or_update_node("C","Campaign","C",30)
    m.add_or_update_edge("A","B",70)
    m.add_or_update_edge("B","C",50)
    probs = m.infer_all()
    assert abs(probs["A"] - 0.8) < 0.01  # no parents
    # B: evidence increases confidence
    assert 0.28 < probs["B"] < 0.29
    # C: negative evidence from B reduces confidence
    assert 0.21 < probs["C"] < 0.22
