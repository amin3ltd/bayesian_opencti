import logging, os
def setup_logging(level: str = 'INFO'):
    os.makedirs('logs', exist_ok=True)
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('logs/service.log', encoding='utf-8')])
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    # Quiet noisy third-party loggers
# for noisy in ("api", "admin", "graphql", "pycti", "requests", "urllib3"):
#     try:
#         logging.getLogger(noisy).setLevel(logging.WARNING)
#     except Exception:
#         pass

