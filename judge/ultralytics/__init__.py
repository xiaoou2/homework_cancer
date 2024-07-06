# Ultralytics YOLO 🚀, GPL-3.0 license

__version__ = '8.0.72'

from .hub import start
from .yolo.engine.model import YOLO
from .yolo.utils.checks import check_yolo as checks

__all__ = '__version__', 'YOLO', 'checks', 'start'  # allow simpler import
