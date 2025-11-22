import argparse
import shutil
import sys
from pathlib import Path
from .config import RAGConfig
from .prepare_documents import process_document, PROCESSORS
from .rag import load_or_initialize_vector_db, update_vector_db_incrementally
from .utils.logging import logger
from .utils.exceptions import RAGException

def main():
    try:
        parser = argparse.ArgumentParser(description="Ajouter un document au système RAG.")
        parser.add_argument("file_path", type=Path, help="Chemin vers le fichier à ajouter.")
        parser.add_argument("--source-dir", type=Path, required=True, help="Dossier source où copier le fichier")
        parser.add_argument("--processed-dir", type=Path, default="./processed_md", help="Dossier de sortie pour les fichiers Markdown")
        parser.add_argument("--db", type=Path, default="./chroma_db", help="Chemin de la base de données Chroma")
        
        args = parser.parse_args()
        
        file_to_add = args.file_path
        if not file_to_add.exists():
            raise RAGException(f"Le fichier {file_to_add} n'existe pas.")

        config = RAGConfig(
            source_dir=args.source_dir.resolve(),
            processed_dir=args.processed_dir.resolve(),
            db_path=args.db.resolve()
        )

        logger.info(f"Ajout du document : {file_to_add.name}")
        
        # 1. Copie vers le dossier source
        config.source_dir.mkdir(parents=True, exist_ok=True)
        target_path = config.source_dir / file_to_add.name
        
        try:
            shutil.copy2(file_to_add, target_path)
            logger.info(f"✅ Fichier copié vers {target_path}")
        except Exception as e:
            raise RAGException(f"Erreur lors de la copie : {e}")

        # 2. Exécution du traitement
        logger.info("--- Lancement du prétraitement ---")
        process_document(target_path, config, PROCESSORS)

        # 3. Indexation
        logger.info("--- Mise à jour de l'index ---")
        vector_db = load_or_initialize_vector_db(config)
        update_vector_db_incrementally(vector_db, config)

        logger.info("✅ Document ajouté et indexé avec succès !")

    except RAGException as e:
        logger.error(f"Erreur lors de l'ajout du document: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
