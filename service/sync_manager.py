from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import logging, time
from datetime import datetime, timezone
from bayes.model import BayesianConfidenceModel
from service.eventbus import EventBus

log = logging.getLogger(__name__)

class SyncManager:
    def __init__(self, max_parents: int, bus: EventBus, cfg: Dict):
        damping = float(cfg.get('lbp_damping', 0.55))
        epsilon = float(cfg.get('lbp_epsilon', 1e-4))
        max_iters = int(cfg.get('lbp_max_iters', 100))
        self.bayes = BayesianConfidenceModel(
            max_parents=max_parents, damping=damping, epsilon=epsilon, max_iters=max_iters
        )
        self.bus = bus
        self.cfg = cfg or {}
        self.last_conf: Dict[str, int] = {}
        self._history: Dict[str, List[Tuple[float, int, int]]] = {}
        self._graph_cache = {'nodes': [], 'edges': []}
        self._rel_type_weight = {
            (k or '').strip().lower(): v
            for k, v in (self.cfg.get('rel_type_weight') or {}).items()
        }
        self._half_life_by_type = {
            (k or '').strip().lower(): v
            for k, v in (self.cfg.get('time_decay_half_life') or {}).items()
        }

    @staticmethod
    def _normalize_type(value: Optional[str]) -> str:
        return (value or '').strip().lower()

    def _type_weight(self, r: str) -> float:
        m = self._rel_type_weight.get(self._normalize_type(r), None)
        if m is None: m = self.cfg.get('default_rel_weight', 0.5)
        return max(0.0, min(1.0, float(m)))

    def _decay(self, val: int, ntype: str, updated_at: Optional[str]) -> int:
        hl = self._half_life_by_type.get(self._normalize_type(ntype), None)
        if not (hl and updated_at): return val
        try:
            t = datetime.fromisoformat(updated_at.replace('Z','+00:00'))
            age_days = max(0.0, (datetime.now(timezone.utc)-t).total_seconds()/86400.0)
            factor = 0.5 ** (age_days/float(hl))
            return int(round(val*factor))
        except Exception:
            return val

    def build_from_opencti(self, objs: List[Dict], rels: List[Dict]):
        # Default priors based on entity type for automatic confidence scoring
        default_priors = {
            'indicator': 70,  # Indicators start with moderate confidence
            'malware': 50,    # Malware needs evidence
            'threat-actor': 60,
            'threat-actor-individual': 60,
            'campaign': 55,
            'intrusion-set': 55,
            'attack-pattern': 65,
            'report': 80,     # Reports are authoritative
            'identity': 75,
            'infrastructure': 45,
            'course-of-action': 85,
        }

        for o in objs:
            ntype = o.get('type', '')
            ntype_key = self._normalize_type(ntype)
            # Use confidence from data if available, else default prior based on type
            raw_confidence = o.get('confidence')
            prior_pct = default_priors.get(ntype_key, 50) if raw_confidence is None else raw_confidence
            updated_at = o.get('updated_at') or o.get('modified') or o.get('created')
            prior_pct = self._decay(prior_pct, ntype_key, updated_at)
            self.bayes.add_or_update_node(o['id'], ntype or 'Unknown', o.get('name', o['id']), prior_pct)

        for r in rels:
            s = r.get('source_ref')
            d = r.get('target_ref')
            t = r.get('relationship_type') or r.get('type')
            t_key = self._normalize_type(t)
            if not (s and d):
                continue

            # floors & fallbacks
            rel_conf_fallback = int(self.cfg.get('rel_conf_fallback', 50))    # applies when confidence is 0/None
            report_object_min = int(self.cfg.get('report_object_min', 30))    # floor for report "object" refs, in [0..100]

            if t_key == 'object':
                # Report -> object: use the report prior but don't let it drop below report_object_min
                src = self.bayes.nodes.get(s)
                src_prior_pct = int(round((src.prior if src else 0.6) * 100))
                base = max(src_prior_pct, report_object_min)
                w = int(round(base * self._type_weight('object')))
            else:
                # Normal relationships: if OpenCTI confidence is 0/None, fall back (e.g., 50)
                raw_confidence = r.get('confidence')
                if raw_confidence is None:
                    base_rel = rel_conf_fallback
                else:
                    base_rel = int(raw_confidence or 0)
                    if base_rel <= 0:
                        base_rel = rel_conf_fallback
                w = int(round(base_rel * self._type_weight(t_key)))

            self.bayes.add_or_update_edge(s, d, w)


        self._graph_cache = {
            'nodes': [{'id': nid, 'label': info.name, 'type': info.type, 'prior': info.prior}
                      for nid, info in self.bayes.nodes.items()],
            'edges': [{'source': src, 'target': dst, 'w': float(self.bayes.edge_w.get((src, dst), 0.0))}
                      for src, dst in self.bayes.G.edges()],
        }

    def update_from_delta(self, objs: List[Dict], rels: List[Dict]):
        self.build_from_opencti(objs, rels)

    def run_inference_and_diff(self) -> List[Tuple[str,int,int]]:
        probs = self.bayes.infer_all()
        diffs: List[Tuple[str,int,int]] = []
        alpha = float(self.cfg.get('ema_alpha', 0.0))
        min_delta = int(self.cfg.get('confidence_push_delta_min', 0))
        ts = time.time()

        for nid, p in probs.items():
            raw = p * 100.0
            prev = self.last_conf.get(nid)
            smoothed = alpha*raw + (1-alpha)*prev if (prev is not None and alpha>0) else raw
            new = int(round(smoothed))
            if prev is None or abs(new - prev) >= min_delta:
                diffs.append((nid, prev if prev is not None else -1, new))
                self.last_conf[nid] = new
                hist = self._history.setdefault(nid, [])
                hist.append((ts, prev if prev is not None else -1, new))
                if len(hist) > 200: del hist[: len(hist) - 200]

        if diffs:
            self.bus.publish({'type':'confidence_updates',
                              'updates':[{'id':i,'old':o,'new':n} for (i,o,n) in diffs]})
        return diffs

    def get_history(self, nid: str) -> List[Dict]:
        hist = self._history.get(nid, [])
        return [{'ts': t, 'old': o, 'new': n} for (t, o, n) in hist]

    def export_graph(self) -> Dict:
        nodes = []
        for nid, info in self.bayes.nodes.items():
            belief = self.last_conf.get(nid, int(round((info.belief if info.belief is not None else info.prior) * 100)))
            nodes.append({'id': nid, 'label': info.name, 'type': info.type,
                          'prior': info.prior * 100, 'belief': belief})
        edges = [{'source': src, 'target': dst, 'w': float(self.bayes.edge_w.get((src, dst), 0.0))}
                 for src, dst in self.bayes.G.edges()]
        return {'nodes': nodes, 'edges': edges}
