

import threading, logging, yaml, os
from dotenv import load_dotenv
from config.settings import settings
from service.logging_setup import setup_logging
from service.opencti_client import OpenCTIClientWrapper
from service.sync_manager import SyncManager
from service.eventbus import EventBus
from api.server import create_app

log = logging.getLogger(__name__)

class MainApp:
    def __init__(self):
        load_dotenv()
        setup_logging(settings.log_level)
        log.info('Starting Bayesian Confidence Service')

        self.bus = EventBus()
        cfg_path = os.path.join('config','bayes.yaml')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        self.sync = SyncManager(max_parents=settings.max_parents_per_node, bus=self.bus, cfg=cfg)
        self.cti = OpenCTIClientWrapper(settings.opencti_url, settings.opencti_token)
        if self.cti.test_connection():
            objs = self.cti.fetch_all_objects()
            rels = self.cti.fetch_all_relationships()
            self.sync.build_from_opencti(objs, rels)
            log.info('Loaded data from OpenCTI')
        else:
            log.warning('Cannot reach OpenCTI. Loading sample data.')
            import json
            with open('sample_data.json', 'r') as f:
                data = json.load(f)
            objs = [o for o in data['objects'] if o.get('type') not in ['relationship', 'sighting']]
            rels = [o for o in data['objects'] if o.get('type') == 'relationship']
            self.sync.build_from_opencti(objs, rels)
            log.info('Loaded sample data')

        # initial compute + push
        initial_diffs = self.sync.run_inference_and_diff()
        if initial_diffs:
            log.info("Initial push: %d confidence updates", len(initial_diffs))
            for (node_id, old, new) in initial_diffs:
                ok = self.cti.update_confidence(node_id, new)
                if not ok:
                    log.warning("Failed to push confidence for %s", node_id)

        # recompute path used by API
        def _recompute_cb():
            if self.cti.test_connection():
                objs2 = self.cti.fetch_all_objects()
                rels2 = self.cti.fetch_all_relationships()
                self.sync.build_from_opencti(objs2, rels2)
                diffs = self.sync.run_inference_and_diff()
                pushed = 0
                for (node_id, old, new) in diffs:
                    if self.cti.update_confidence(node_id, new): pushed += 1
                return diffs, pushed
            else:
                # Re-run inference without fetching new data
                diffs = self.sync.run_inference_and_diff()
                return diffs, 0

        self._stop = threading.Event()
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()

        self.app = create_app(self.sync, self.bus, recompute_cb=_recompute_cb)

    def _poll_loop(self):
        log.info('Polling loop started (interval=%ss).', settings.poll_interval_seconds)
        while not self._stop.is_set():
            try:
                delta = self.cti.poll_changes()
                self.sync.update_from_delta(delta.get('objects',[]), delta.get('relationships',[]))
                diffs = self.sync.run_inference_and_diff()
                if diffs:
                    log.info("Pushing %d confidence updates", len(diffs))
                for (node_id, old, new) in diffs:
                    ok = self.cti.update_confidence(node_id, new)
                    if not ok:
                        log.warning("Failed to push confidence for %s", node_id)
            except Exception as e:
                log.exception('Polling cycle error: %s', e)
            finally:
                self._stop.wait(settings.poll_interval_seconds)

    def run(self):
        log.info('Starting Flask server on 0.0.0.0:5000')
        self.app.run(host='0.0.0.0', port=5000, debug=False)
    def stop(self): self._stop.set()
