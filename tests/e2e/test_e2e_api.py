import json
from pathlib import Path

import pytest

from api.server import create_app
from service.eventbus import EventBus
from service.sync_manager import SyncManager


pytestmark = pytest.mark.e2e


def build_sync_manager_from_sample():
    root = Path(__file__).resolve().parents[2]
    with (root / "sample_data.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    objects = [o for o in data.get("objects", []) if o.get("type") != "relationship"]
    rels = [o for o in data.get("objects", []) if o.get("type") == "relationship"]

    manager = SyncManager(max_parents=5, bus=EventBus(), cfg={})
    manager.build_from_opencti(objects, rels)
    return manager


def test_end_to_end_api_flow():
    manager = build_sync_manager_from_sample()
    app = create_app(sync_manager=manager, event_bus=manager.bus)
    app.config["TESTING"] = True

    with app.test_client() as client:
        status = client.get("/api/v1/status")
        assert status.status_code == 200

        network = client.get("/api/v1/network")
        assert network.status_code == 200
        payload = network.get_json()
        assert payload["nodes"]
        assert payload["edges"]

        node_id = payload["nodes"][0]["id"]
        node = client.get(f"/api/v1/node?id={node_id}")
        assert node.status_code == 200

        contribs = client.get(f"/api/v1/contributions?id={node_id}&topk=5")
        assert contribs.status_code == 200

        paths = client.get(f"/api/v1/paths?id={node_id}&k=3&maxlen=3")
        assert paths.status_code == 200

        recompute = client.post("/api/v1/recompute")
        assert recompute.status_code == 200

        history = client.get(f"/api/v1/history?id={node_id}")
        assert history.status_code == 200
