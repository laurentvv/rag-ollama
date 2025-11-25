import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
import ollama
import subprocess
import yaml
from .config import RAGConfig
from .utils.logging import logger
from .utils.exceptions import RAGException, OllamaError, ModelUnavailableError, ProcessingError
from .processors.base import DocumentProcessor
from .processors.pdf import PDFProcessor
from .processors.image import ImageProcessor
from .processors.docling import DoclingProcessor

# ... (les fonctions de vérification restent les mêmes) ...


def check_ollama_available():
    try:
        ollama.list()
    except Exception:
        raise OllamaError("Impossible de communiquer avec Ollama.")

def check_model_available(model_name):
    try:
        models_info = ollama.list()
        available = [m['model'] for m in models_info.get('models', [])]
        if not any(model_name in m or model_name.split(':')[0] in m.split(':')[0] for m in available):
            raise ModelUnavailableError(f"Modèle '{model_name}' non trouvé.")
    except Exception as e:
        raise OllamaError(f"Impossible de vérifier les modèles Ollama : {e}")

def setup_directories(config: RAGConfig):
    try:
        config.source_dir.mkdir(parents=True, exist_ok=True)
        if config.processed_dir.exists():
            shutil.rmtree(config.processed_dir)
        config.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dossiers prêts. Source: {config.source_dir}, Sortie: {config.processed_dir}")
    except Exception as e:
        raise RAGException(f"Erreur lors de la création des dossiers: {e}")

PROCESSORS: list[DocumentProcessor] = [
    PDFProcessor(),
    ImageProcessor(),
    DoclingProcessor(),
]

def add_filename_context(file_path: Path, original_filename: str):
    if not file_path or not file_path.exists(): return
    try:
        content = file_path.read_text(encoding='utf-8')
        header = f"# Source: {original_filename}\n\n"
        if not content.startswith("# Source:"):
            file_path.write_text(header + content, encoding='utf-8')
    except Exception as e:
        logger.warning(f"Impossible d'ajouter le contexte au fichier {file_path.name}: {e}")

def process_document(file_path: Path, config: RAGConfig, processors: list[DocumentProcessor]):
    """Trouve le bon processeur et traite un seul document."""
    for processor in processors:
        if processor.can_process(file_path):
            try:
                logger.info(f"Traitement de {file_path.name} avec {processor.__class__.__name__}...")
                output_path = processor.process(file_path, config)
                add_filename_context(output_path, file_path.name)
                return
            except ProcessingError as e:
                logger.error(f"Erreur lors du traitement de {file_path.name}: {e}")
                return # Arrêter pour ce fichier
    logger.warning(f"Aucun processeur trouvé pour le type de fichier: {file_path.name}")


def main():
    try:
        # Configuration et parsing des arguments
        yaml_conf = {} # Simplifié
        parser = argparse.ArgumentParser(description="Prétraitement de documents pour RAG.")
        parser.add_argument("--input", "-i", type=Path, required=True)
        parser.add_argument("--output", "-o", type=Path, required=True)
        parser.add_argument("--vision-model", default=yaml_conf.get("vision_model", "llama3.2-vision"))
        parser.add_argument("--pdf-model", default=yaml_conf.get("pdf_model", "llama3.2-vision"))
        args = parser.parse_args()

        if not args.input.exists():
            raise RAGException(f"Dossier source '{args.input}' introuvable.")

        config = RAGConfig(
            source_dir=args.input.resolve(),
            processed_dir=args.output.resolve(),
            db_path=Path(), # Non utilisé ici
            vision_model=args.vision_model,
            pdf_model=args.pdf_model
        )

        # Vérifications initiales
        check_ollama_available()
        check_model_available(config.vision_model)
        check_model_available(config.pdf_model)

        setup_directories(config)

        # Traitement des fichiers
        files_to_process = [p for p in config.source_dir.iterdir() if p.is_file()]

        if not files_to_process:
            logger.warning(f"Aucun fichier trouvé dans le dossier source {config.source_dir}.")
            return

        logger.info(f"{len(files_to_process)} fichiers à traiter.")

        for file_path in tqdm(files_to_process, desc="Traitement des documents"):
            process_document(file_path, config, PROCESSORS)

        logger.info("✅ Prétraitement de tous les documents terminé.")

    except (RAGException, OllamaError, ModelUnavailableError) as e:
        logger.error(f"Erreur de configuration ou système: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
