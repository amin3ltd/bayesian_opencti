from __future__ import annotations
from typing import Dict, Tuple, Set, Optional
from dataclasses import dataclass
import math
import networkx as nx

# ------------------ utilities ------------------

EPS = 1e-9

def clamp01(x: float) -> float:
    return max(EPS, min(1.0 - EPS, float(x)))

# ------------------ data ------------------

@dataclass
class NodeInfo:
    id: str
    type: str
    name: str
    user_confidence: float  # raw input 0..1
    prior: float            # model prior 0..1 (here kept equal to user_confidence)
    belief: Optional[float] = None  # posterior 0..1

# ------------------ model ------------------

class BayesianConfidenceModel:
    """
    Noisy-OR based confidence propagation. Key points:
      - Uses Noisy-OR update: P(node) = 1 - (1-prior) * ∏_parents (1 - w * P(parent))
      - For acyclic singletons we apply one exact update (parents' beliefs are known).
      - For cyclic SCCs we do damped fixed-point iteration until convergence.
      - Parent cap enforced per node (keep top-k strongest parents).
    """
    def __init__(
        self,
        max_parents: int = 5,
        damping: float = 0.45,
        epsilon: float = 1e-5,
        max_iters: int = 200,
    ):
        self.max_parents = int(max_parents)
        self.damping = float(damping)
        self.epsilon = float(epsilon)
        self.max_iters = int(max_iters)

        self.nodes: Dict[str, NodeInfo] = {}
        self.G = nx.DiGraph()
        self.edge_w: Dict[Tuple[str, str], float] = {}  # (src,dst) -> weight in [0,1]

    # ---------- building ----------

    def add_or_update_node(self, nid: str, ntype: str, name: str, confidence_pct: int):
        """
        Add or update node. confidence_pct is 0..100 (user input).
        We store as floats in 0..1. We keep prior == user_confidence for now.
        """
        uc = clamp01(float(confidence_pct) / 100.0)
        prior = uc
        if nid not in self.nodes:
            self.nodes[nid] = NodeInfo(id=nid, type=ntype, name=name,
                                      user_confidence=uc, prior=prior, belief=None)
            self.G.add_node(nid)
        else:
            n = self.nodes[nid]
            n.type = ntype
            n.name = name
            n.user_confidence = uc
            n.prior = prior
            # do not overwrite belief (warm start preserved)

    def add_or_update_edge(self, src: str, dst: str, weight_pct: int):
        """
        Add/update directed edge src -> dst with weight in 0..100.
        Enforces parent cap (keeps strongest max_parents parents for dst).
        """
        if src == dst or src not in self.nodes or dst not in self.nodes:
            return

        w = clamp01(float(weight_pct) / 100.0)
        # store/update edge
        self.G.add_edge(src, dst)
        self.edge_w[(src, dst)] = float(w)

        # enforce parent cap: keep top-k parents by weight
        preds = list(self.G.predecessors(dst))
        if len(preds) > self.max_parents:
            weighted = [(p, self.edge_w.get((p, dst), 0.0)) for p in preds]
            weighted.sort(key=lambda kv: kv[1], reverse=True)
            keep_set = {p for p, _ in weighted[: self.max_parents]}
            # remove edges not kept
            for p in preds:
                if p not in keep_set:
                    try:
                        self.G.remove_edge(p, dst)
                    except Exception:
                        pass
                    self.edge_w.pop((p, dst), None)

    # ---------- inference ----------

    def _noisyor_value(self, nid: str, belief_dict: Dict[str, float]) -> float:
        """
        Compute Noisy-OR posterior for node 'nid' given beliefs in belief_dict for parents.
        belief_dict should contain values for all potential parents (fallback to prior if missing).
        Formula:
          P(node) = 1 - (1 - prior) * Π_{p in parents} (1 - w_p * P(parent))
        """
        node = self.nodes[nid]
        prod = 1.0 - node.prior
        for p in self.G.predecessors(nid):
            w = float(self.edge_w.get((p, nid), 0.0))
            bp = float(belief_dict.get(p, self.nodes[p].prior if p in self.nodes else 0.0))
            # each parent reduces the "false" probability by factor (1 - w * bp)
            prod *= (1.0 - w * bp)
        val = 1.0 - prod
        return clamp01(val)

    def infer_all(self) -> Dict[str, float]:
        """
        Run inference over entire graph and return dict nid->posterior (0..1).
        Approach:
          - Build condensation DAG of SCCs.
          - Process DAG in topological order:
              * If SCC is a single node without self-loop: one noisy-or update using current beliefs.
              * Else: run damped fixed-point iterations inside the SCC using noisy-or updates,
                      treating external parents with their already-computed beliefs.
        """
        # warm start: use existing belief if present else prior
        beliefs: Dict[str, float] = {
            nid: (info.belief if info.belief is not None else info.prior)
            for nid, info in self.nodes.items()
        }

        # condensation DAG
        sccs = list(nx.strongly_connected_components(self.G))
        comp_index = {n: i for i, comp in enumerate(sccs) for n in comp}
        C = nx.DiGraph()
        for i, comp in enumerate(sccs):
            C.add_node(i, nodes=set(comp))
        for u, v in self.G.edges():
            cu, cv = comp_index[u], comp_index[v]
            if cu != cv:
                C.add_edge(cu, cv)
        order = list(nx.topological_sort(C))

        # process components
        for cid in order:
            comp_nodes: Set[str] = C.nodes[cid]["nodes"]
            # singleton and acyclic (no self-loop): exact noisy-or
            if len(comp_nodes) == 1:
                n = next(iter(comp_nodes))
                if not self.G.has_edge(n, n):
                    # parents outside (or none) use current beliefs
                    beliefs[n] = self._noisyor_value(n, beliefs)
                    continue

            # cyclic or multi-node SCC: damped fixed-point
            comp_list = list(comp_nodes)
            # initialize local beliefs from global beliefs (warm start)
            b_old = {n: beliefs.get(n, self.nodes[n].prior) for n in comp_list}

            for _ in range(self.max_iters):
                b_new: Dict[str, float] = {}
                max_delta = 0.0
                for n in comp_list:
                    # build a belief dict where parents inside SCC use b_old, external parents use beliefs
                    def parent_bel(p):
                        return b_old[p] if p in comp_nodes else beliefs.get(p, self.nodes[p].prior if p in self.nodes else 0.0)

                    # compute noisy-or using parent_bel
                    prod = 1.0 - self.nodes[n].prior
                    for p in self.G.predecessors(n):
                        w = float(self.edge_w.get((p, n), 0.0))
                        bp = parent_bel(p)
                        prod *= (1.0 - w * bp)
                    val = clamp01(1.0 - prod)
                    # damping update
                    updated = (1.0 - self.damping) * b_old[n] + self.damping * val
                    updated = clamp01(updated)
                    b_new[n] = updated
                    max_delta = max(max_delta, abs(updated - b_old[n]))

                b_old = b_new
                if max_delta < self.epsilon:
                    break

            for n in comp_list:
                beliefs[n] = b_old[n]

        # persist posteriors as plain floats
        for nid, v in beliefs.items():
            self.nodes[nid].belief = float(v)

        return {nid: float(v) for nid, v in beliefs.items()}

    # ---------- helpers ----------

    def export_graph(self) -> Dict:
        """
        JSON-serializable representation for API: nodes and edges.
        node fields: id, label, type, prior, belief, user_confidence
        edges: source, target, w
        """
        nodes = []
        for nid, info in self.nodes.items():
            nodes.append({
                "id": nid,
                "label": info.name,
                "type": info.type,
                "prior": float(info.prior),
                "belief": float(info.belief) if info.belief is not None else None,
                "user_confidence": float(info.user_confidence)
            })
        edges = []
        for s, t in self.G.edges():
            edges.append({"source": s, "target": t, "w": float(self.edge_w.get((s, t), 0.0))})
        return {"nodes": nodes, "edges": edges}
