"""
Unit tests for Bayesian Confidence Model.
Tests inference, edge handling, and validation.
"""

import pytest
from bayes.model import BayesianConfidenceModel, clamp01, NodeInfo


class TestClamp01:
    """Tests for clamp01 utility function."""

    def test_normal_value(self):
        """Test clamping of normal values."""
        assert clamp01(0.5) == pytest.approx(0.5)

    def test_negative_value(self):
        """Test clamping of negative values."""
        result = clamp01(-0.1)
        assert result > 0
        assert result < 0.01

    def test_value_above_one(self):
        """Test clamping of values above 1."""
        result = clamp01(1.5)
        assert result < 1.0
        assert result > 0.99

    def test_exact_zero(self):
        """Test clamping of exact zero."""
        result = clamp01(0.0)
        assert result == pytest.approx(1e-9)

    def test_exact_one(self):
        """Test clamping of exact one."""
        result = clamp01(1.0)
        assert result == pytest.approx(1.0 - 1e-9)


class TestBayesianModel:
    """Tests for BayesianConfidenceModel core functionality."""

    def test_add_single_node(self):
        """Test adding a single node."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "Test Indicator", 80)
        
        assert "A" in m.nodes
        assert m.nodes["A"].type == "Indicator"
        assert m.nodes["A"].name == "Test Indicator"
        assert m.nodes["A"].user_confidence == pytest.approx(0.8)
        assert m.nodes["A"].prior == pytest.approx(0.8)
        assert m.nodes["A"].belief is None

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_node("B", "Malware", "B", 20)
        m.add_or_update_node("C", "Campaign", "C", 30)
        
        assert len(m.nodes) == 3
        assert set(m.nodes.keys()) == {"A", "B", "C"}

    def test_confidence_percentage_conversion(self):
        """Test 0-100 input to 0-1 conversion."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 0)
        assert m.nodes["A"].user_confidence == pytest.approx(1e-9)
        
        m.add_or_update_node("B", "Indicator", "B", 100)
        assert m.nodes["B"].user_confidence == pytest.approx(1.0 - 1e-9)

    def test_update_existing_node(self):
        """Test updating an existing node preserves belief."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.nodes["A"].belief = 0.85  # Set belief manually
        
        m.add_or_update_node("A", "Malware", "Updated", 90)
        
        assert m.nodes["A"].type == "Malware"
        assert m.nodes["A"].name == "Updated"
        assert m.nodes["A"].user_confidence == pytest.approx(0.9)
        assert m.nodes["A"].belief == pytest.approx(0.85)  # Preserved

    def test_add_edge(self):
        """Test adding an edge between nodes."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_node("B", "Malware", "B", 20)
        m.add_or_update_edge("A", "B", 70)
        
        assert m.G.has_edge("A", "B")
        assert m.edge_w[("A", "B")] == pytest.approx(0.7)

    def test_self_loop_rejected(self):
        """Test that self-loops are rejected."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_edge("A", "A", 70)
        
        assert not m.G.has_edge("A", "A")

    def test_edge_to_nonexistent_node_rejected(self):
        """Test edges to non-existent nodes are rejected."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_edge("A", "B", 70)  # B doesn't exist
        
        assert not m.G.has_edge("A", "B")

    def test_parent_cap_enforcement(self):
        """Test that max_parents limit is enforced."""
        m = BayesianConfidenceModel(max_parents=2)
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_node("B", "Indicator", "B", 70)
        m.add_or_update_node("C", "Indicator", "C", 60)
        m.add_or_update_node("D", "Malware", "D", 20)
        
        # Add 3 edges to D
        m.add_or_update_edge("A", "D", 80)
        m.add_or_update_edge("B", "D", 70)
        m.add_or_update_edge("C", "D", 60)
        
        # Only 2 should be kept (strongest)
        assert len(list(m.G.predecessors("D"))) == 2

    def test_infer_single_node(self):
        """Test inference with single node."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 80)
        probs = m.infer_all()
        
        assert probs["A"] == pytest.approx(0.8, rel=1e-3)

    def test_small_network(self):
        """Test basic Noisy-OR inference."""
        m = BayesianConfidenceModel(max_parents=3)
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_node("B", "Malware", "B", 20)
        m.add_or_update_edge("A", "B", 70)
        probs = m.infer_all()
        
        assert probs["A"] > 0.75
        assert probs["B"] >= 0.20

    def test_chain_network(self):
        """Test multi-hop inference."""
        m = BayesianConfidenceModel(max_parents=3)
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_node("B", "Malware", "B", 20)
        m.add_or_update_node("C", "Campaign", "C", 30)
        m.add_or_update_edge("A", "B", 70)
        m.add_or_update_edge("B", "C", 50)
        probs = m.infer_all()
        
        expected_a = 0.8
        expected_b = 1.0 - (1.0 - 0.2) * (1.0 - 0.7 * expected_a)
        expected_c = 1.0 - (1.0 - 0.3) * (1.0 - 0.5 * expected_b)

        assert probs["A"] == pytest.approx(expected_a, rel=1e-3)
        assert probs["B"] == pytest.approx(expected_b, rel=1e-3)
        assert probs["C"] == pytest.approx(expected_c, rel=1e-3)

    def test_cyclic_network(self):
        """Test cyclic network convergence."""
        m = BayesianConfidenceModel(max_parents=3, damping=0.55, epsilon=1e-6, max_iters=1000)
        m.add_or_update_node("A", "Indicator", "A", 50)
        m.add_or_update_node("B", "Malware", "B", 50)
        m.add_or_update_edge("A", "B", 60)
        m.add_or_update_edge("B", "A", 60)
        probs = m.infer_all()
        
        # Symmetric cycle converges to fixed point for Noisy-OR
        expected = 0.5 / 0.7
        assert probs["A"] == pytest.approx(expected, rel=1e-3)
        assert probs["B"] == pytest.approx(expected, rel=1e-3)

    def test_warm_start_preserves_belief(self):
        """Test that exact updates are deterministic across runs."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 50)
        m.add_or_update_node("B", "Malware", "B", 50)
        m.add_or_update_edge("A", "B", 60)
        
        # First inference
        probs1 = m.infer_all()
        
        # Modify belief
        m.nodes["B"].belief = 0.9
        
        # Second inference
        probs2 = m.infer_all()
        
        # Exact updates for DAGs should be stable regardless of warm start
        assert probs2["B"] == pytest.approx(probs1["B"])


class TestGraphExport:
    """Tests for graph export functionality."""

    def test_export_format(self):
        """Test JSON-serializable export format."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "Test", 80)
        m.add_or_update_node("B", "Malware", "Mal", 20)
        m.add_or_update_edge("A", "B", 70)
        m.infer_all()
        
        export = m.export_graph()
        
        assert "nodes" in export
        assert "edges" in export
        assert len(export["nodes"]) == 2
        assert len(export["edges"]) == 1
        
        # Check node structure
        node = export["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "prior" in node
        assert "belief" in node
        assert "user_confidence" in node
        
        # Check edge structure
        edge = export["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "w" in edge


class TestEdgeWeights:
    """Tests for edge weight handling."""

    def test_weight_clamping(self):
        """Test that edge weights are clamped to [0,1]."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_node("B", "Malware", "B", 20)
        
        # Weight above 100
        m.add_or_update_edge("A", "B", 150)
        assert m.edge_w[("A", "B")] <= 1.0
        
        # Negative weight
        m.add_or_update_edge("A", "B", -50)
        assert m.edge_w[("A", "B")] >= 0.0

    def test_default_edge_weight(self):
        """Test that non-configured types use default weight."""
        m = BayesianConfidenceModel()
        m.add_or_update_node("A", "Indicator", "A", 80)
        m.add_or_update_node("B", "Malware", "B", 20)
        m.add_or_update_edge("A", "B", 50)  # Default 50% weight
        
        assert m.edge_w[("A", "B")] == pytest.approx(0.5)
