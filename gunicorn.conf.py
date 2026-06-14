import multiprocessing
import os

# Binding
bind = "0.0.0.0:" + os.environ.get("PORT", "5000")

# Worker Options
# Calculate optimal workers: (2 * CPUs) + 1 is standard formula
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gthread"
threads = 4
timeout = 120 # Higher timeout for potential slow ML inference

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
