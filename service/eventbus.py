from typing import Any, Dict, List
import queue, threading
class EventBus:
    def __init__(self): self._subs: List[queue.Queue] = []; self._lock = threading.Lock()
    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=1000); 
        with self._lock: self._subs.append(q); 
        return q
    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subs: self._subs.remove(q)
    def publish(self, event: Dict[str, Any]):
        with self._lock: subs = list(self._subs)
        for q in subs:
            try: q.put_nowait(event)
            except queue.Full:
                try: q.get_nowait()
                except Exception: pass
                try: q.put_nowait(event)
                except Exception: pass
