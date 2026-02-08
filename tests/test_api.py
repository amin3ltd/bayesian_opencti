"""
API endpoint tests for Flask server.
"""

import pytest

pytestmark = pytest.mark.integration
from unittest.mock import Mock, patch
from bayes.model import BayesianConfidenceModel
from service.eventbus import EventBus


class TestAPIEndpoints:
    """Test API endpoints."""
    
    @pytest.fixture
    def app_and_client(self):
        """Create test Flask app and client."""
        from api.server import create_app
        
        # Create minimal sync manager
        bus = Mock()
        bus.subscribe.return_value = Mock()
        manager = Mock()
        manager.bayes = BayesianConfidenceModel()
        manager.bayes.add_or_update_node("A", "Indicator", "Test", 80)
        manager.bayes.add_or_update_node("B", "Malware", "Mal", 20)
        manager.bayes.add_or_update_edge("A", "B", 70)
        manager.last_conf = {}
        manager.export_graph.return_value = manager.bayes.export_graph()
        manager.get_history.return_value = []
        manager.run_inference_and_diff.return_value = [("A", -1, 80), ("B", -1, 20)]
        
        app = create_app(
            sync_manager=manager,
            event_bus=bus,
            recompute_cb=None,
            config_path="config/bayes.yaml"
        )
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            yield app, client, manager
    
    def test_status_endpoint(self, app_and_client):
        """Test /api/v1/status endpoint."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/status')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
    
    def test_network_endpoint(self, app_and_client):
        """Test /api/v1/network endpoint."""
        _, client, manager = app_and_client
        response = client.get('/api/v1/network')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'nodes' in data
        assert 'edges' in data
    
    def test_node_endpoint_valid_id(self, app_and_client):
        """Test /api/v1/node endpoint with valid ID."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/node?id=A')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['id'] == 'A'
        assert 'name' in data
        assert 'type' in data
    
    def test_node_endpoint_missing_id(self, app_and_client):
        """Test /api/v1/node endpoint without ID."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/node')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_node_endpoint_unknown_id(self, app_and_client):
        """Test /api/v1/node endpoint with unknown ID."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/node?id=UNKNOWN')
        
        assert response.status_code == 404
    
    def test_contributions_endpoint(self, app_and_client):
        """Test /api/v1/contributions endpoint."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/contributions?id=B')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'node' in data
        assert 'contributions' in data
    
    def test_contributions_endpoint_topk(self, app_and_client):
        """Test /api/v1/contributions with topk parameter."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/contributions?id=B&topk=5')
        
        assert response.status_code == 200
    
    def test_paths_endpoint(self, app_and_client):
        """Test /api/v1/paths endpoint."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/paths?id=B')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'paths' in data
    
    def test_paths_endpoint_with_params(self, app_and_client):
        """Test /api/v1/paths with k and maxlen parameters."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/paths?id=B&k=10&maxlen=5')
        
        assert response.status_code == 200
    
    def test_history_endpoint(self, app_and_client):
        """Test /api/v1/history endpoint."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/history?id=A')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'history' in data
    
    def test_history_endpoint_missing_id(self, app_and_client):
        """Test /api/v1/history without ID."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/history')
        
        assert response.status_code == 400
    
    def test_config_endpoint(self, app_and_client):
        """Test /api/v1/config endpoint."""
        _, client, _ = app_and_client
        response = client.get('/api/v1/config')
        
        assert response.status_code == 200


class TestEventBus:
    """Test EventBus functionality."""
    
    def test_subscribe_unsubscribe(self):
        """Test subscribe and unsubscribe."""
        bus = EventBus()
        q = bus.subscribe()
        
        assert len(bus._subs) == 1
        assert q in bus._subs
        
        bus.unsubscribe(q)
        assert len(bus._subs) == 0
    
    def test_publish_to_subscribers(self):
        """Test publishing events to subscribers."""
        bus = EventBus()
        q = bus.subscribe()
        
        bus.publish({"type": "test", "data": "value"})
        
        # Should receive the event
        try:
            event = q.get_nowait()
            assert event['type'] == 'test'
            assert event['data'] == 'value'
        except Exception:
            pytest.fail("Event not received")
    
    def test_multiple_subscribers(self):
        """Test multiple subscribers receive events."""
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        
        bus.publish({"type": "broadcast"})
        
        # Both should receive
        try:
            e1 = q1.get_nowait()
            e2 = q2.get_nowait()
            assert e1 == e2
        except Exception:
            pytest.fail("Event not received by all subscribers")
    
    def test_unsubscribe_removes_correct_queue(self):
        """Test unsubscribe only removes specific queue."""
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        
        bus.unsubscribe(q1)
        
        assert q1 not in bus._subs
        assert q2 in bus._subs


class TestRecomputeEndpoint:
    """Test /api/v1/recompute endpoint."""
    
    @pytest.fixture
    def app_with_recompute(self):
        """Create app with recompute callback."""
        from api.server import create_app
        
        bus = Mock()
        bus.subscribe.return_value = Mock()
        manager = Mock()
        manager.bayes = BayesianConfidenceModel()
        manager.bayes.add_or_update_node("A", "Indicator", "Test", 80)
        manager.last_conf = {}
        manager.run_inference_and_diff.return_value = [("A", -1, 80)]
        
        recompute_called = []
        def recompute_cb():
            recompute_called.append(True)
            return [("A", -1, 85)], 1
        
        app = create_app(
            sync_manager=manager,
            event_bus=bus,
            recompute_cb=recompute_cb,
        )
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            yield app, client, recompute_called
    
    def test_recompute_with_callback(self, app_with_recompute):
        """Test recompute with callback."""
        _, client, recompute_called = app_with_recompute
        response = client.post('/api/v1/recompute')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'updated' in data
        assert 'diffs' in data
        assert len(recompute_called) == 1
