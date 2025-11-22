import logging
import sys

def setup_logging():
    """Configure le logger centralisé pour le projet."""
    logger = logging.getLogger("rag_ollama")
    logger.setLevel(logging.INFO)

    # Evite d'ajouter plusieurs handlers si la fonction est appelée plusieurs fois
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logging()
