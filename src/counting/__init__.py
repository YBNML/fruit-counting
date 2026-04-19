"""counting — fruit counting pipeline."""

from counting.config.loader import load_config
from counting.io.results import CountingResult
from counting.pipeline import Pipeline, build_pipeline

__version__ = "0.1.0"

__all__ = ["__version__", "Pipeline", "build_pipeline", "load_config", "CountingResult"]
