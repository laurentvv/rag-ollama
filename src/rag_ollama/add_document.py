import argparse
import shutil
import sys
from pathlib import Path
import subprocess

# Configuration
# Note: This should match prepare_documents.py

def main():
    parser = argparse.ArgumentParser(description="Ajouter un document au système RAG.")
    parser.add_argument("file_path", type=Path, help="Chemin vers le fichier à ajouter.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Dossier source où copier le fichier")
    parser.add_argument("--processed-dir", type=Path, required=True, help="Dossier de sortie pour les fichiers Markdown")
    parser.add_argument("--db", type=Path, required=True, help="Chemin de la base de données Chroma")
    
    args = parser.parse_args()
    
    file_path = args.file_path
    target_dir = args.source_dir
    
    if not file_path.exists():
        print(f"Erreur: Le fichier {file_path} n'existe pas.")
        sys.exit(1)
        
    print(f"Ajout du document : {file_path.name}")
    
    # 1. Copie vers le dossier source
    if not target_dir.exists():
        print(f"Création du dossier cible : {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        
    target_path = target_dir / file_path.name
    
    try:
        shutil.copy2(file_path, target_path)
        print(f"✅ Fichier copié vers {target_path}")
    except Exception as e:
        print(f"Erreur lors de la copie : {e}")
        sys.exit(1)
        
    # 2. Exécution de prepare_documents.py
    print("\n--- Lancement du prétraitement ---")
    try:
        # On appelle rag-prepare avec les arguments
        cmd_prepare = [
            "uv", "run", "rag-prepare",
            "--input", str(target_dir),
            "--output", str(args.processed_dir)
        ]
        subprocess.run(cmd_prepare, check=True)
    except subprocess.CalledProcessError:
        print("Erreur lors du prétraitement.")
        sys.exit(1)
        
    # 3. Indexation (rag_ollama_new.py -> rag.py)
    print("\n--- Mise à jour de l'index ---")
    try:
        # On appelle rag-chat avec les arguments
        cmd_chat = [
            "uv", "run", "rag-chat",
            "--index-only",
            "--input", str(args.processed_dir),
            "--db", str(args.db)
        ]
        subprocess.run(cmd_chat, check=True)
    except subprocess.CalledProcessError:
        print("Erreur lors de l'indexation.")
        sys.exit(1)
        
    print("\n✅ Document ajouté et indexé avec succès !")

if __name__ == "__main__":
    main()
