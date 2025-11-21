import subprocess
import shutil
import sys
import os
from pathlib import Path
import zipfile
import base64
from openai import OpenAI
import mimetypes
import ollama
import yaml

import argparse

# --- Configuration des Chemins ---
# Utilise pathlib pour une gestion multiplateforme (Windows, Linux, macOS)
# BASE_DIR = Répertoire où se trouve ce script
# BASE_DIR = Répertoire racine du projet (2 niveaux au-dessus de ce fichier)
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
# Global variables to be updated by args
SOURCE_DIR = None
PROCESSED_DIR = None
PDF_PROVIDER = "ollama"
PDF_MODEL = "llama3.2-vision"
VISION_MODEL = "llama3.2-vision"

def check_tesseract_installed():
    """Vérifie si Tesseract-OCR est installé et accessible."""
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True, text=True)
        print("✅ Tesseract-OCR est correctement installé.")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n--- ERREUR PRÉREQUIS ---", file=sys.stderr)
        print("Tesseract-OCR n'est pas trouvé. C'est un prérequis pour l'option --ocr de Docling.", file=sys.stderr)
        print("Veuillez l'installer sur votre système et vous assurer qu'il est dans le PATH.", file=sys.stderr)
        print("\nInstructions d'installation :", file=sys.stderr)
        print("  - Windows : Téléchargez depuis le dépôt Tesseract at UB Mannheim.", file=sys.stderr)
        print("  - Linux (Debian/Ubuntu) : sudo apt-get install tesseract-ocr tesseract-ocr-fra", file=sys.stderr)
        print("  - macOS (Homebrew) : brew install tesseract", file=sys.stderr)
        print("\nAprès l'installation, vérifiez que la commande 'tesseract --version' fonctionne dans votre terminal.", file=sys.stderr)
        return False

def load_config():
    """Charge la configuration depuis config.yaml."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        config_path = BASE_DIR / "config.yaml"
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️  Erreur chargement config.yaml: {e}", file=sys.stderr)
    return {}

def check_ollama_available():
    """Vérifie si le serveur Ollama est accessible."""
    try:
        ollama.list()
        return True
    except Exception:
        print("\n--- ERREUR OLLAMA ---", file=sys.stderr)
        print("❌ Impossible de communiquer avec Ollama.", file=sys.stderr)
        print("Assurez-vous qu'Ollama est lancé (commande 'ollama serve' ou via l'application).", file=sys.stderr)
        return False

def check_model_available(model_name):
    """Vérifie si un modèle est présent dans Ollama."""
    try:
        models_info = ollama.list()
        # models_info['models'] est une liste d'objets
        available_models = [m['model'] for m in models_info.get('models', [])]
        
        # Gestion des tags (ex: 'llama3' vs 'llama3:latest')
        if model_name in available_models:
            return True
        if f"{model_name}:latest" in available_models:
            return True
        
        # Si le modèle contient un tag, on vérifie exact match
        if ':' in model_name:
             if model_name in available_models:
                 return True
        else:
             # Si pas de tag, on regarde si une version existe
             for m in available_models:
                 if m.split(':')[0] == model_name:
                     return True

        print(f"\n⚠️  ATTENTION : Le modèle '{model_name}' n'est pas trouvé dans Ollama.", file=sys.stderr)
        print(f"   -> Installation recommandée : ollama pull {model_name}", file=sys.stderr)
        return False
    except Exception:
        return False

def setup_directories():
    """Prépare les dossiers source et de destination."""
    try:
        # 1. Crée le dossier source s'il n'existe pas
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Dossier source vérifié : {SOURCE_DIR}")
        print("-> Assurez-vous d'y placer vos fichiers (PDF, DOCX, etc.)")

        # 2. Nettoie et (re)crée le dossier de destination
        if PROCESSED_DIR.exists():
            print(f"Nettoyage de l'ancien dossier : {PROCESSED_DIR}")
            shutil.rmtree(PROCESSED_DIR)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Dossier de destination créé : {PROCESSED_DIR}")

    except PermissionError:
        print(f"Erreur: Permission refusée. Impossible de nettoyer/créer {PROCESSED_DIR}.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la préparation des dossiers : {e}", file=sys.stderr)
        sys.exit(1)

def has_images_docx(file_path):
    """
    Détecte si un fichier DOCX/PPTX contient des images.
    """
    try:
        ext = file_path.suffix.lower()
        if ext in [".docx", ".pptx"]:
            with zipfile.ZipFile(file_path, 'r') as z:
                # Les images sont stockées dans word/media ou ppt/media
                for name in z.namelist():
                    if "media/" in name and not name.endswith("/"):
                        return True
            return False
        return False
    except Exception:
        return False # En cas d'erreur, on assume pas d'images pour ne pas bloquer, ou True par sécurité ? Disons False pour Docling fast.

def process_pdf_with_ai(file_path):
    """Traite un PDF avec pdf-ocr-ai via Ollama."""
    output_path = PROCESSED_DIR / (file_path.stem + ".md")
    print(f"\nTraitement AI (Ollama) : {file_path.name}")
    
    # Commande directe pdf-ocr-ai (installé comme dépendance)
    command = [
        "pdf-ocr-ai",
        str(file_path),
        str(output_path),
        "--provider", PDF_PROVIDER,
        "--model", PDF_MODEL
    ]
    
    try:
        # On affiche la sortie en temps réel pour que l'utilisateur voie la progression
        env = os.environ.copy()
        env["OPENAI_TIMEOUT"] = "600" # Augmente le timeout à 10 minutes pour les modèles lents
        subprocess.run(command, check=True, encoding='utf-8', errors='replace', env=env)
        print(f"✅ PDF traité : {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du traitement de {file_path.name}", file=sys.stderr)
    except FileNotFoundError:
        print("❌ Commande 'pdf-ocr-ai' introuvable. Vérifiez l'installation.", file=sys.stderr)

def process_image_with_ai(file_path):
    """Traite une image avec un modèle VLM via Ollama."""
    output_path = PROCESSED_DIR / (file_path.stem + ".md")
    print(f"\nTraitement Image AI : {file_path.name}")
    
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "image/jpeg" # Fallback

        # Encodage image en base64
        # Note: Ollama python lib accepts path or base64. Path is easier if local.
        # But let's stick to the user's request pattern or just pass the path if supported.
        # The python lib supports 'images': ['path/to/img'] or base64.
        
        # Utilisation de la librairie native ollama
        # https://github.com/ollama/ollama-python
        
        # Let's use the Client approach as requested by user
        client = ollama.Client(host='http://localhost:11434', timeout=600) # 10 min request timeout
        
        response = client.chat(
            model=VISION_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe this image in detail, capturing all visible text and visual elements for documentation purposes.',
                    'images': [str(file_path)]
                }
            ],
            options={
                'num_ctx': 4096, # Context window
            },
            keep_alive='15m'
        )
        
        content = response['message']['content']
        
        output_path.write_text(content, encoding='utf-8')
        print(f"✅ Image traitée : {output_path}")
            
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {file_path.name}: {e}", file=sys.stderr)
        print("Assurez-vous qu'Ollama est lancé et que le modèle vision est disponible.", file=sys.stderr)

def run_docling_batch(files, use_ocr):
    """Exécute Docling sur un lot de fichiers."""
    if not files:
        return

    mode_name = "COMPLET (OCR + VLM)" if use_ocr else "RAPIDE (Texte seul)"
    print(f"\nTraitement Docling {mode_name} : {len(files)} fichiers")
    
    # Determine docling path relative to current python executable
    # This handles the case where the script is run via venv python but docling is not in PATH
    python_dir = Path(sys.executable).parent
    docling_path = python_dir / "docling.exe"
    
    # Fallback to "docling" if the specific executable doesn't exist (e.g. linux/mac or global install)
    docling_cmd = str(docling_path) if docling_path.exists() else "docling"

    files_args = [str(f) for f in files]
    command = [
        docling_cmd,
        *files_args,
        "--to", "md",
        "--output", str(PROCESSED_DIR)
    ]

    if use_ocr:
        command.extend(["--ocr", "--enrich-picture-description"])
    else:
        command.extend(["--no-ocr", "--no-enrich-picture-description"])

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(f"✅ Lot Docling {mode_name} terminé.")
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERREUR sur le lot {mode_name} ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)

def add_filename_context(file_path, original_filename):
    """Ajoute le nom du fichier original en en-tête du fichier Markdown."""
    try:
        if not file_path.exists():
            return

        content = file_path.read_text(encoding='utf-8', errors='replace')
        header = f"# Source: {original_filename}\n\n"
        
        # Évite de rajouter l'en-tête s'il existe déjà (cas de re-run)
        if not content.startswith("# Source:"):
            file_path.write_text(header + content, encoding='utf-8')
            print(f"   -> Contexte ajouté : {original_filename}")
    except Exception as e:
        print(f"   -> Erreur ajout contexte : {e}", file=sys.stderr)

def process_raw_text_file(file_path):
    """Copie et traite les fichiers texte bruts (.txt, .md)."""
    output_path = PROCESSED_DIR / (file_path.stem + ".md")
    print(f"\nTraitement Fichier Brut : {file_path.name}")
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='replace')
        header = f"# Source: {file_path.name}\n\n"
        
        # Si c'est déjà un MD et qu'il a déjà l'en-tête, on ne le remet pas
        if content.startswith("# Source:"):
            final_content = content
        else:
            final_content = header + content
            
        output_path.write_text(final_content, encoding='utf-8')
        print(f"✅ Fichier brut copié : {output_path}")
    except Exception as e:
        print(f"❌ Erreur copie fichier brut {file_path.name}: {e}", file=sys.stderr)

def run_docling_conversion():
    """Exécute la commande de conversion Docling."""

    # Vérifie si le dossier source contient des fichiers
    files_to_process = []
    # Extensions supportées par Docling
    docling_extensions = ["*.pdf", "*.docx", "*.pptx", "*.html", "*.asciidoc"]
    # Extensions brutes
    raw_extensions = ["*.txt", "*.md"]
    # Extensions images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]
    
    for ext in docling_extensions + raw_extensions + image_extensions:
        files_to_process.extend(list(SOURCE_DIR.glob(ext)))

    if not files_to_process:
        print(f"\nAvertissement : Aucun fichier supporté trouvé dans '{SOURCE_DIR}'.")
        return

    print(f"\nFichiers trouvés : {len(files_to_process)}")
    
    pdf_files = []
    docling_fast_files = []
    docling_full_files = []
    raw_files = []
    image_files = []

    print("Analyse et routage des fichiers...")
    for f in files_to_process:
        ext = f.suffix.lower()
        if ext == ".pdf":
            print(f" - {f.name} -> PDF OCR AI (Ollama)")
            pdf_files.append(f)
        elif ext in [".docx", ".pptx"]:
            if has_images_docx(f):
                print(f" - {f.name} -> Docling COMPLET (Images détectées)")
                docling_full_files.append(f)
            else:
                print(f" - {f.name} -> Docling RAPIDE (Texte pur)")
                docling_fast_files.append(f)
        elif ext in [".txt", ".md"]:
            print(f" - {f.name} -> Copie Brute (Texte)")
            raw_files.append(f)
        elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            print(f" - {f.name} -> Image AI (Ollama)")
            image_files.append(f)
        else:
            # Autres fichiers (html, asciidoc) -> Docling rapide
            print(f" - {f.name} -> Docling RAPIDE")
            docling_fast_files.append(f)

    print("\nLancement des conversions...")
    
    # 1. Traitement des PDF avec pdf-ocr-ai
    for pdf in pdf_files:
        process_pdf_with_ai(pdf)
        # Ajout du contexte
        output_md = PROCESSED_DIR / (pdf.stem + ".md")
        add_filename_context(output_md, pdf.name)

    # 2. Traitement des Images avec AI
    for img in image_files:
        process_image_with_ai(img)
        output_md = PROCESSED_DIR / (img.stem + ".md")
        add_filename_context(output_md, img.name)

    # 3. Traitement Docling
    if docling_fast_files:
        run_docling_batch(docling_fast_files, use_ocr=False)
        for f in docling_fast_files:
             output_md = PROCESSED_DIR / (f.stem + ".md")
             add_filename_context(output_md, f.name)
    
    if docling_full_files:
        run_docling_batch(docling_full_files, use_ocr=True)
        for f in docling_full_files:
             output_md = PROCESSED_DIR / (f.stem + ".md")
             add_filename_context(output_md, f.name)

    # 4. Traitement Fichiers Bruts
    for raw in raw_files:
        process_raw_text_file(raw)

    print("\n----------------------------------------------------")
    print("✅ Toutes les conversions sont terminées.")
    print(f"Les fichiers Markdown sont disponibles dans : {PROCESSED_DIR}")

def main():
    global SOURCE_DIR, PROCESSED_DIR, PDF_PROVIDER, PDF_MODEL, VISION_MODEL
    
    # Chargement de la config
    config = load_config()
    prepare_config = config.get("prepare", {})
    
    default_vision = prepare_config.get("vision_model", "llama3.2-vision")
    default_pdf_provider = prepare_config.get("pdf_provider", "ollama")
    default_pdf_model = prepare_config.get("pdf_model", "llama3.2-vision")

    parser = argparse.ArgumentParser(description="Script de prétraitement de documents pour RAG (OCR + VLM)")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Dossier contenant les documents sources (PDF, DOCX, Images...)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Dossier de sortie pour les fichiers Markdown")
    parser.add_argument("--vision-model", type=str, default=default_vision, help=f"Modèle Vision pour les images (default: {default_vision})")
    parser.add_argument("--pdf-provider", type=str, default=default_pdf_provider, help=f"Provider pour pdf-ocr-ai (default: {default_pdf_provider})")
    parser.add_argument("--pdf-model", type=str, default=default_pdf_model, help=f"Modèle pour pdf-ocr-ai (default: {default_pdf_model})")
    
    args = parser.parse_args()
    
    SOURCE_DIR = args.input.resolve()
    PROCESSED_DIR = args.output.resolve()
    VISION_MODEL = args.vision_model
    PDF_PROVIDER = args.pdf_provider
    PDF_MODEL = args.pdf_model
    
    print("--- Script de Prétraitement Docling ---")
    print(f"Source : {SOURCE_DIR}")
    print(f"Sortie : {PROCESSED_DIR}")
    print(f"PDF Provider: {PDF_PROVIDER} | PDF Model: {PDF_MODEL}")
    print(f"Image Vision Model: {VISION_MODEL}")
    
    if not check_tesseract_installed():
        sys.exit(1)

    if not check_ollama_available():
        sys.exit(1)
    
    # Vérification des modèles (warning seulement)
    check_model_available(VISION_MODEL)
    if PDF_PROVIDER == "ollama":
        check_model_available(PDF_MODEL)
        
    setup_directories()
    run_docling_conversion()

if __name__ == "__main__":
    main()
