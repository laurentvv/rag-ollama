import subprocess
import shutil
import sys
from pathlib import Path

# --- Configuration des Chemins ---
# Utilise pathlib pour une gestion multiplateforme (Windows, Linux, macOS)
# BASE_DIR = Répertoire où se trouve ce script
BASE_DIR = Path(__file__).parent.resolve()
SOURCE_DIR = BASE_DIR / "sources"
PROCESSED_DIR = BASE_DIR / "processed_md"

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

def run_docling_conversion():
    """Exécute la commande de conversion Docling."""

    # Vérifie si le dossier source contient des fichiers
    if not any(SOURCE_DIR.iterdir()):
        print(f"\nAvertissement : Le dossier source '{SOURCE_DIR}' est vide.")
        print("Processus terminé sans conversion. Ajoutez des fichiers et relancez.")
        return

    print("\nLancement de la conversion Docling...")
    print("Cela peut prendre du temps en fonction du nombre de fichiers et de l'activation de l'OCR...")

    command = [
        "docling",
        "--output-format", "markdown",
        "--output-dir", str(PROCESSED_DIR),
        "--ocr",
        "--image-captions",
        str(SOURCE_DIR)
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')

        print("\n----------------------------------------------------")
        print("✅ Conversion Docling terminée avec succès.")
        print(f"Les fichiers Markdown sont disponibles dans : {PROCESSED_DIR}")

    except FileNotFoundError:
        print("\n--- ERREUR ---", file=sys.stderr)
        print("La commande 'docling' n'a pas été trouvée.", file=sys.stderr)
        print("Veuillez vous assurer que Docling est bien installé et accessible dans le PATH de votre système.", file=sys.stderr)
        print("Installation : pip install docling", file=sys.stderr)
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print("\n--- ERREUR ---", file=sys.stderr)
        print("Une erreur est survenue pendant l'exécution de Docling.", file=sys.stderr)
        print(f"Commande exécutée : {' '.join(command)}", file=sys.stderr)
        print(f"Code de retour : {e.returncode}", file=sys.stderr)
        print("\nSortie d'erreur (Stderr) de Docling :", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("\nSortie standard (Stdout) de Docling :", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        sys.exit(1)

def main():
    print("--- Script de Prétraitement Docling ---")
    if not check_tesseract_installed():
        sys.exit(1)
    setup_directories()
    run_docling_conversion()

if __name__ == "__main__":
    main()
