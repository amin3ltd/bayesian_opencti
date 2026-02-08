"""
Integration tests for SyncManager and OpenCTI client.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from bayes.model import BayesianConfidenceModel
from service.sync_manager import SyncManager
from service.eventbus import EventBus


class MockEventBus:
    """Mock EventBus for testing."""
    
    def __init__(self):
        self.messages = []
    
    def publish(self, msg):
        self.messages.append(msg)


class TestSyncManager:
    """Integration tests for SyncManager."""
    
    def test_build_graph_from_objects(self):
        """Test building graph from STIX objects."""
        bus = MockEventBus()
        cfg = {
            'lbp_damping': 0.55,
            'rel_type_weight': {'indicates': 0.85, 'uses': 0.35},
            'default_rel_weight': 0.5,
        }
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        # Mock STIX objects
        objects = [
            {'id': 'indicator-1', 'type': 'indicator', 'name': 'Malicious IP', 'confidence': 80},
            {'id': 'malware-1', 'type': 'malware', 'name': 'Ransomware', 'confidence': 50},
            {'id': 'campaign-1', 'type': 'campaign', 'name': 'Attack', 'confidence': 60},
        ]
        relationships = [
            {'source_ref': 'indicator-1', 'target_ref': 'malware-1', 'type': 'indicates', 'confidence': 70},
            {'source_ref': 'campaign-1', 'target_ref': 'malware-1', 'type': 'uses', 'confidence': 80},
        ]
        
        manager.build_from_opencti(objects, relationships)
        
        assert len(manager.bayes.nodes) == 3
        assert manager.bayes.G.has_edge('indicator-1', 'malware-1')
        assert manager.bayes.G.has_edge('campaign-1', 'malware-1')
    
    def test_default_priors_by_type(self):
        """Test that default priors are applied by entity type."""
        bus = MockEventBus()
        cfg = {}
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        objects = [
            {'id': 'ind-1', 'type': 'indicator', 'name': 'I1', 'confidence': None},
            {'id': 'mal-1', 'type': 'malware', 'name': 'M1', 'confidence': None},
            {'id': 'rep-1', 'type': 'report', 'name': 'R1', 'confidence': None},
        ]
        
        manager.build_from_opencti(objects, [])
        
        assert manager.bayes.nodes['ind-1'].prior == pytest.approx(0.7)  # Default indicator
        assert manager.bayes.nodes['mal-1'].prior == pytest.approx(0.5)  # Default malware
        assert manager.bayes.nodes['rep-1'].prior == pytest.approx(0.8)  # Default report
    
    def test_report_object_relationship(self):
        """Test report->object relationship handling."""
        bus = MockEventBus()
        cfg = {
            'rel_type_weight': {'object': 0.5},
            'report_object_min': 30,
        }
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        objects = [
            {'id': 'report-1', 'type': 'report', 'name': 'Report', 'confidence': 80},
            {'id': 'malware-1', 'type': 'malware', 'name': 'Malware', 'confidence': None},
        ]
        relationships = [
            {'source_ref': 'report-1', 'target_ref': 'malware-1', 'type': 'object', 'confidence': None},
        ]
        
        manager.build_from_opencti(objects, relationships)
        
        # Edge should use report prior * type_weight, minimum 30
        assert manager.bayes.G.has_edge('report-1', 'malware-1')

    def test_relationship_type_from_stix_relationship(self):
        """Test relationship_type is used when type=relationship."""
        bus = MockEventBus()
        cfg = {
            'rel_type_weight': {'uses': 0.20},
            'default_rel_weight': 0.90,
        }
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)

        objects = [
            {'id': 'campaign-1', 'type': 'campaign', 'name': 'C1', 'confidence': 80},
            {'id': 'malware-1', 'type': 'malware', 'name': 'M1', 'confidence': 50},
        ]
        relationships = [
            {'source_ref': 'campaign-1', 'target_ref': 'malware-1',
             'type': 'relationship', 'relationship_type': 'uses', 'confidence': 80},
        ]

        manager.build_from_opencti(objects, relationships)

        w = manager.bayes.edge_w.get(('campaign-1', 'malware-1'))
        assert w == pytest.approx(0.16)
    
    def test_inference_and_diff(self):
        """Test inference generates diffs."""
        bus = MockEventBus()
        cfg = {'ema_alpha': 0.0, 'confidence_push_delta_min': 0}
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        objects = [
            {'id': 'A', 'type': 'indicator', 'name': 'A', 'confidence': 80},
            {'id': 'B', 'type': 'malware', 'name': 'B', 'confidence': 20},
        ]
        relationships = [
            {'source_ref': 'A', 'target_ref': 'B', 'type': 'indicates', 'confidence': 70},
        ]
        
        manager.build_from_opencti(objects, relationships)
        diffs = manager.run_inference_and_diff()
        
        assert len(diffs) == 2  # Two nodes
        assert all(len(d) == 3 for d in diffs)  # Each diff has (id, old, new)
    
    def test_history_tracking(self):
        """Test confidence history is tracked."""
        bus = MockEventBus()
        cfg = {'ema_alpha': 0.0, 'confidence_push_delta_min': 0}
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        objects = [{'id': 'A', 'type': 'indicator', 'name': 'A', 'confidence': 50}]
        
        manager.build_from_opencti(objects, [])
        manager.run_inference_and_diff()
        
        history = manager.get_history('A')
        assert len(history) >= 1
        assert 'ts' in history[0]
        assert 'old' in history[0]
        assert 'new' in history[0]
    
    def test_graph_export(self):
        """Test graph export format."""
        bus = MockEventBus()
        cfg = {}
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        objects = [
            {'id': 'A', 'type': 'indicator', 'name': 'A', 'confidence': 80},
            {'id': 'B', 'type': 'malware', 'name': 'B', 'confidence': 20},
        ]
        relationships = [
            {'source_ref': 'A', 'target_ref': 'B', 'type': 'indicates', 'confidence': 70},
        ]
        
        manager.build_from_opencti(objects, relationships)
        manager.run_inference_and_diff()
        export = manager.export_graph()
        
        assert 'nodes' in export
        assert 'edges' in export
        assert len(export['nodes']) == 2
        assert len(export['edges']) == 1


class TestTimeDecay:
    """Tests for time decay functionality."""
    
    def test_decay_calculation(self):
        """Test time decay is applied correctly."""
        bus = MockEventBus()
        cfg = {
            'time_decay_half_life': {'Indicator': 60},
            'ema_alpha': 0.0,
            'confidence_push_delta_min': 0,
        }
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        # Create object with recent timestamp
        recent = datetime.now(timezone.utc).isoformat()
        objects = [
            {'id': 'A', 'type': 'indicator', 'name': 'A', 'confidence': 100, 'updated_at': recent}
        ]
        
        manager.build_from_opencti(objects, [])
        manager.run_inference_and_diff()
        
        # Should be close to original (recent)
        assert manager.last_conf.get('A', 0) >= 90
    
    def test_no_decay_without_timestamp(self):
        """Test no decay when timestamp missing."""
        bus = MockEventBus()
        cfg = {
            'time_decay_half_life': {'Indicator': 60},
            'ema_alpha': 0.0,
            'confidence_push_delta_min': 0,
        }
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        objects = [
            {'id': 'A', 'type': 'indicator', 'name': 'A', 'confidence': 100, 'updated_at': None}
        ]
        
        manager.build_from_opencti(objects, [])
        manager.run_inference_and_diff()
        
        # Should be exactly original
        assert manager.last_conf.get('A') == 100

    def test_decay_uses_modified_timestamp(self):
        """Test decay applies when modified timestamp is present."""
        bus = MockEventBus()
        cfg = {
            'time_decay_half_life': {'Indicator': 30},
            'ema_alpha': 0.0,
            'confidence_push_delta_min': 0,
        }
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)

        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        objects = [
            {'id': 'A', 'type': 'indicator', 'name': 'A', 'confidence': 100, 'modified': old_time}
        ]

        manager.build_from_opencti(objects, [])
        manager.run_inference_and_diff()

        assert manager.last_conf.get('A', 0) < 100


class TestTypeWeight:
    """Tests for relationship type weighting."""
    
    def test_custom_type_weights(self):
        """Test custom relationship type weights."""
        bus = MockEventBus()
        cfg = {
            'rel_type_weight': {
                'indicates': 0.95,  # Strong
                'uses': 0.20,       # Weak
            },
            'ema_alpha': 0.0,
            'confidence_push_delta_min': 0,
        }
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        objects = [
            {'id': 'A', 'type': 'indicator', 'name': 'A', 'confidence': 80},
            {'id': 'B', 'type': 'malware', 'name': 'B', 'confidence': 50},
            {'id': 'C', 'type': 'malware', 'name': 'C', 'confidence': 50},
        ]
        relationships = [
            {'source_ref': 'A', 'target_ref': 'B', 'type': 'indicates', 'confidence': 80},
            {'source_ref': 'A', 'target_ref': 'C', 'type': 'uses', 'confidence': 80},
        ]
        
        manager.build_from_opencti(objects, relationships)
        manager.run_inference_and_diff()
        
        # B should have higher belief due to stronger weight
        belief_B = manager.bayes.nodes['B'].belief
        belief_C = manager.bayes.nodes['C'].belief
        assert belief_B > belief_C


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        bus = MockEventBus()
        cfg = {}
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        manager.build_from_opencti([], [])
        
        assert len(manager.bayes.nodes) == 0
        assert len(list(manager.bayes.G.edges())) == 0
    
    def test_relationship_missing_refs(self):
        """Test relationships with missing source/target."""
        bus = MockEventBus()
        cfg = {}
        manager = SyncManager(max_parents=5, bus=bus, cfg=cfg)
        
        objects = [
            {'id': 'A', 'type': 'indicator', 'name': 'A', 'confidence': 80},
        ]
        relationships = [
            {'source_ref': None, 'target_ref': 'A', 'type': 'indicates', 'confidence': 70},
            {'source_ref': 'A', 'target_ref': None, 'type': 'indicates', 'confidence': 70},
        ]
        
        manager.build_from_opencti(objects, relationships)
        
        # No edges should be created
        assert len(list(manager.bayes.G.edges())) == 0
    
    def test_max_parents_enforcement(self):
        """Test max_parents is enforced."""
        bus = MockEventBus()
        cfg = {'ema_alpha': 0.0, 'confidence_push_delta_min': 0}
        manager = SyncManager(max_parents=2, bus=bus, cfg=cfg)
        
        objects = [
            {'id': f'S{i}', 'type': 'indicator', 'name': f'S{i}', 'confidence': 80}
            for i in range(5)
        ] + [
            {'id': 'T', 'type': 'malware', 'name': 'T', 'confidence': 50}
        ]
        relationships = [
            {'source_ref': f'S{i}', 'target_ref': 'T', 'type': 'indicates', 'confidence': 80}
            for i in range(5)
        ]
        
        manager.build_from_opencti(objects, relationships)
        
        # Only 2 parents should be kept
        preds = list(manager.bayes.G.predecessors('T'))
        assert len(preds) == 2
