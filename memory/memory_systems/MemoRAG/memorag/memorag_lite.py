import logging

logger = logging.getLogger(__name__)


class MemoRAGLite:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("MemoRAGLite is disabled in this build (transformers dependency removed).")
