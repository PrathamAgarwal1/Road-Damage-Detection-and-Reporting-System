"""
Gunicorn settings, picked up automatically when gunicorn starts in the project folder
(so the Render start command can stay a plain `gunicorn app:app`).
"""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
# One process keeps a single copy of the models in memory; threads serve concurrent requests,
# and the gthread worker keeps sending heartbeats while a background thread loads the models.
workers = int(os.getenv('WEB_CONCURRENCY', '1'))
worker_class = 'gthread'
threads = int(os.getenv('GUNICORN_THREADS', '4'))
# Loading PyTorch + the models and analysing large photos can take well over the 30 s default.
timeout = int(os.getenv('GUNICORN_TIMEOUT', '180'))
graceful_timeout = 30
# Load the app in the worker, not the master: models load once, in the process that serves requests.
preload_app = False
accesslog = '-'
