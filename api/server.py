from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from pathlib import Path
from typing import Optional
import json, queue, yaml, copy

def create_app(sync_manager, event_bus, recompute_cb=None, config_path="config/bayes.yaml"):
    # Serve dashboard as before
    BASE_DIR = Path(__file__).resolve().parent.parent
    DASHBOARD_DIR = BASE_DIR / "dashboard"
    app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")
    CORS(app)

    # helpers
    def _belief(nid: str) -> float:
        n = sync_manager.bayes.nodes.get(nid)
        if not n:
            return 0.0
        # prefer pushed (last_conf), else model's belief
        if nid in sync_manager.last_conf:
            return float(sync_manager.last_conf[nid]) / 100.0
        return n.belief if n.belief is not None else n.prior


    def _prior(nid: str) -> float:
        n = sync_manager.bayes.nodes.get(nid)
        return n.prior if n else 0.0

    def _parents(nid: str):
        if nid not in sync_manager.bayes.nodes:
            return []
        parents = []
        for p in sync_manager.bayes.G.predecessors(nid):
            w = sync_manager.bayes.edge_w.get((p, nid), 0.0)
            parents.append((p, w))
        return parents

    def _noisyor_with_parents(nid: str, exclude_parent: Optional[str] = None) -> float:
        p0 = _prior(nid)
        prod = (1.0 - p0)
        for pid, w in _parents(nid):
            if exclude_parent and pid == exclude_parent:
                continue
            bj = _belief(pid)
            prod *= (1.0 - float(w) * float(bj))
        return 1.0 - max(0.0, min(1.0, prod))

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/v1/status")
    def status():
        return jsonify({"status": "ok"}), 200

    @app.get("/api/v1/network")
    def network():
        try:
            return jsonify(sync_manager.export_graph()), 200
        except Exception as e:
            return jsonify({"error": f"Failed to export graph: {e}"}), 500

    @app.post("/api/v1/recompute")
    def recompute():
        try:
            if callable(recompute_cb):
                diffs, pushed = recompute_cb()
                return jsonify({
                    "updated": int(pushed or 0),
                    "diffs": [{"id": i, "old": o, "new": n} for (i, o, n) in (diffs or [])]
                }), 200
            diffs = sync_manager.run_inference_and_diff()
            return jsonify({
                "updated": 0,
                "diffs": [{"id": i, "old": o, "new": n} for (i, o, n) in diffs]
            }), 200
        except Exception as e:
            return jsonify({"error": f"Recompute failed: {e}"}), 500

    @app.get("/api/v1/node")
    def node_info():
        try:
            nid = request.args.get("id")
            if not nid: return jsonify({"error": "id required"}), 400
            n = sync_manager.bayes.nodes.get(nid)
            if not n: return jsonify({"error": "unknown id"}), 404
            return jsonify({
                "id": nid, "name": n.name, "type": n.type,
                "prior": _prior(nid), "posterior": _belief(nid)
            }), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get node info: {e}"}), 500

    @app.get("/api/v1/contributions")
    def contributions():
        try:
            nid = request.args.get("id")
            topk = int(request.args.get("topk", 10))
            if not nid: return jsonify({"error": "id required"}), 400
            if nid not in sync_manager.bayes.nodes: return jsonify({"error": "unknown id"}), 404
            base = _noisyor_with_parents(nid)
            out = []
            for pid, w in _parents(nid):
                bj = _belief(pid)
                score = float(w) * float(bj)
                wo = _noisyor_with_parents(nid, exclude_parent=pid)
                lift = max(0.0, base - wo)
                out.append({"parent": pid, "w": float(w), "belief": float(bj),
                            "score": float(score), "marginal_lift": float(lift)})
            out.sort(key=lambda d: d["score"], reverse=True)
            return jsonify({"node": nid, "posterior": base, "contributions": out[:topk]}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get contributions: {e}"}), 500

    @app.get("/api/v1/paths")
    def paths():
        try:
            nid = request.args.get("id")
            k = int(request.args.get("k", 5))
            maxlen = int(request.args.get("maxlen", 3))
            if not nid: return jsonify({"error": "id required"}), 400
            if nid not in sync_manager.bayes.nodes: return jsonify({"error": "unknown id"}), 404
            best = []
            def dfs(curr, path, prod_w, start_belief, depth):
                if depth > maxlen: return
                parents = list(sync_manager.bayes.G.predecessors(curr))
                if not parents or depth == maxlen:
                    if len(path) > 1:
                        best.append({"path": list(reversed(path)), "score": float(prod_w * start_belief)})
                    return
                for p in parents:
                    if p in path: continue
                    w = sync_manager.bayes.edge_w.get((p, curr), 0.0)
                    bj = _belief(p)
                    dfs(p, path + [p], prod_w * w,
                        start_belief if len(path) > 1 else bj, depth + 1)
            dfs(nid, [nid], 1.0, 0.0, 0)
            best.sort(key=lambda d: d["score"], reverse=True)
            return jsonify({"paths": best[:k]}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get paths: {e}"}), 500

    @app.get("/api/v1/history")
    def history():
        nid = request.args.get("id")
        if not nid: return jsonify({"error": "id required"}), 400
        if hasattr(sync_manager, "get_history"):
            return jsonify({"history": sync_manager.get_history(nid)}), 200
        return jsonify({"history": []}), 200

    @app.get("/api/v1/config")
    def get_config():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return jsonify(cfg), 200
        except Exception as e:
            return jsonify({"error": f"failed to read {config_path}: {e}"}), 500

    @app.get("/api/v1/stream")
    def stream():
        q = event_bus.subscribe()
        def gen():
            try:
                while True:
                    try:
                        event = q.get(timeout=15)
                        yield "data: " + json.dumps(event) + "\n\n"
                    except queue.Empty:
                        yield ": ping\n\n"
            finally:
                event_bus.unsubscribe(q)
        return Response(gen(), mimetype="text/event-stream")

    return app
