Voici une analyse détaillée et des améliorations concrètes pour votre projet RAG-Ollama. J'ai identifié plusieurs axes d'amélioration structurale, de robustesse et de maintenabilité.

## 1. Consolidation de la Configuration (Priorité Haute)

**Problème** : Le `config.yaml` n'est jamais chargé. Les `yaml_conf = {}` sont des placeholders inopérants.

**Solution** : Créez un chargeur de config centralisé :

```python
# src/rag_ollama/config.py
from dataclasses import dataclass, fields
from pathlib import Path
import yaml
from typing import Any, Dict, Optional

@dataclass
class RAGConfig:
    source_dir: Path = Path("./sources")
    processed_dir: Path = Path("./processed_md")
    db_path: Path = Path("./chroma_db")
    vision_model: str = "llama3.2-vision"
    pdf_model: str = "llama3.2-vision"
    llm_model: str = "gemma3:12b"
    embedding_model: str = "embeddinggemma:latest"

def load_config(config_path: Path = Path("config.yaml")) -> RAGConfig:
    """Charge la configuration depuis YAML avec fallback sur les valeurs par défaut."""
    defaults = {f.name: f.default for f in fields(RAGConfig) if f.default is not fields._MISSING_TYPE}
    
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        
        # Fusionner avec les valeurs par défaut
        merged = {}
        for key, default_value in defaults.items():
            section_key = key.split('_')[0]  # e.g., "vision_model" -> "vision"
            section = yaml_data.get(section_key, {})
            merged[key] = section.get(key, default_value)
        
        return RAGConfig(**{k: v for k, v in merged.items() if k in defaults})
    
    return RAGConfig()

def save_config(config: RAGConfig, config_path: Path = Path("config.yaml")):
    """Sauvegarde la configuration dans YAML."""
    yaml_data = {
        "prepare": {
            "vision_model": config.vision_model,
            "pdf_model": config.pdf_model,
        },
        "chat": {
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
```

**Mise à jour de `prepare_documents.py`** :

```python
# src/rag_ollama/prepare_documents.py
# ...
def main():
    try:
        config = load_config()  # Charge le YAML
        
        parser = argparse.ArgumentParser(description="Prétraitement de documents pour RAG.")
        parser.add_argument("--input", "-i", type=Path, required=True)
        parser.add_argument("--output", "-o", type=Path, required=True)
        parser.add_argument("--vision-model", default=config.vision_model)
        parser.add_argument("--pdf-model", default=config.pdf_model)
        args = parser.parse_args()

        # Override config with CLI args
        config.source_dir = args.input.resolve()
        config.processed_dir = args.output.resolve()
        config.vision_model = args.vision_model
        config.pdf_model = args.pdf_model
        
        # Suite du code...
```

## 2. Refactorisation des Vérifications Ollama (Évite la duplication)

**Problème** : Les vérifications Ollama sont répétées dans plusieurs fichiers.

**Solution** : Créez un gestionnaire de modèles dédié :

```python
# src/rag_ollama/utils/ollama_manager.py
import ollama
import time
from pathlib import Path
from .exceptions import OllamaError, ModelUnavailableError
from .logging import logger

class OllamaManager:
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = ollama.Client(host=host, timeout=30)
    
    def check_connection(self):
        try:
            self.client.list()
        except Exception as e:
            raise OllamaError(f"Impossible de communiquer avec Ollama sur {self.client.host}: {e}")
    
    def check_model(self, model_name: str, auto_pull: bool = True):
        """Vérifie si un modèle existe, le télécharge si nécessaire."""
        try:
            models = self.client.list()
            available = [m['model'] for m in models.get('models', [])]
            
            # Vérification plus robuste (tag ou nom de base)
            model_base = model_name.split(':')[0]
            if not any(model_name == m or model_base in m for m in available):
                if auto_pull:
                    logger.info(f"Modèle {model_name} non trouvé. Tentative de téléchargement...")
                    self.client.pull(model_name)
                    logger.info(f"✅ Modèle {model_name} téléchargé.")
                else:
                    raise ModelUnavailableError(f"Modèle '{model_name}' non trouvé.")
        except Exception as e:
            if "pull" in str(e).lower():
                raise ModelUnavailableError(f"Modèle '{model_name}' non trouvé et échec du téléchargement: {e}")
            raise OllamaError(f"Erreur lors de la vérification du modèle: {e}")
    
    def unload_model(self, model_name: str):
        """Décharge un modèle de la mémoire."""
        try:
            self.client.chat(model=model_name, messages=[], keep_alive=0)
            logger.debug(f"Modèle {model_name} déchargé.")
            time.sleep(1)  # Pause pour libération mémoire
        except Exception as e:
            logger.warning(f"Erreur lors du déchargement de {model_name}: {e}")
```

**Utilisation dans `prepare_documents.py`** :

```python
from .utils.ollama_manager import OllamaManager

def main():
    # ...
    manager = OllamaManager()
    manager.check_connection()
    manager.check_model(config.vision_model)
    manager.check_model(config.pdf_model)
    # ...
```

## 3. Amélioration des Processeurs (Robustesse & Sécurité)

**Problème** : Exécution de commandes shell sans validation des chemins.

**Solution** : Validez les exécutables et les chemins :

```python
# src/rag_ollama/processors/pdf.py
from pathlib import Path
import shutil
from .base import DocumentProcessor
from ..config import RAGConfig
from ..utils.exceptions import ProcessingError

class PDFProcessor(DocumentProcessor):
    def can_process(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"
    
    def _find_executable(self) -> Path:
        """Trouve l'exécutable pdf-ocr-ai de manière sûre."""
        # 1. Chercher dans le PATH système
        executable = shutil.which("pdf-ocr-ai")
        if executable:
            return Path(executable)
        
        # 2. Chercher dans le site-packages
        import sysconfig
        scripts_dir = Path(sysconfig.get_path("scripts"))
        candidate = scripts_dir / "pdf-ocr-ai"
        if sys.platform == "win32":
            candidate = candidate.with_suffix(".exe")
        
        if candidate.exists():
            return candidate
        
        raise ProcessingError(
            "pdf-ocr-ai non trouvé. Installez-le avec: pip install pdf-ocr-ai"
        )

    def process(self, file_path: Path, config: RAGConfig) -> Path:
        output_path = config.processed_dir / (file_path.stem + ".md")
        pdf_ocr_cmd = self._find_executable()
        
        # Validation des chemins
        if not file_path.exists():
            raise ProcessingError(f"Fichier PDF introuvable: {file_path}")
        
        command = [
            str(pdf_ocr_cmd), str(file_path), str(output_path),
            "--provider", "ollama", "--model", config.pdf_model
        ]
        
        import subprocess
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=3600  # 1h max
            )
            if result.stderr:
                logger.warning(f"pdf-ocr-ai warnings: {result.stderr}")
            return output_path
        except subprocess.TimeoutExpired:
            raise ProcessingError(f"Timeout lors du traitement de {file_path.name}")
        except subprocess.CalledProcessError as e:
            raise ProcessingError(f"pdf-ocr-ai a échoué: {e.stderr}")
```

## 4. Optimisation du RAG avec Cache et Streaming

**Problème** : Chargement répété des documents pour BM25 et pas de streaming progressif.

**Solution** : Cache et streaming amélioré :

```python
# src/rag_ollama/rag.py
from functools import lru_cache
import chromadb

@lru_cache(maxsize=1)
def get_bm25_retriever(vector_db: Chroma):
    """Cache le BM25 retriever (lourd à construire)."""
    logger.info("Construction du BM25Retriever (mis en cache)...")
    all_docs = vector_db.get(include=["documents"])['documents']
    bm25 = BM25Retriever.from_texts(all_docs)
    bm25.k = 3
    return bm25

def setup_rag_chain(vector_db: Chroma, config: RAGConfig):
    logger.info(f"Configuration du modèle Ollama: {config.llm_model}...")
    llm = Ollama(model=config.llm_model)
    
    # Prompt optimisé pour le français
    prompt = ChatPromptTemplate.from_template("""Vous êtes un assistant expert. Répondez à la question en vous basant UNIQUEMENT sur le contexte fourni. Si vous ne trouvez pas la réponse, dites clairement que vous ne savez pas.

Contexte: 
{context}

Question: {input}

Réponse (en français):""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = get_bm25_retriever(vector_db)
    
    return document_chain, EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )

# Amélioration du streaming
while True:
    question = input("\nVotre question: ")
    if question.lower() in ['exit', 'quit', 'q']: break
    
    try:
        relevant_docs = retriever.invoke(question)
        print("\n🤖 Réponse: ", end="", flush=True)
        
        # Streaming avec gestion d'erreurs
        full_response = ""
        for chunk in doc_chain.stream({"input": question, "context": relevant_docs}):
            if chunk:
                print(chunk, end="", flush=True)
                full_response += chunk
        
        # Log la réponse complète
        logger.debug(f"Réponse générée: {full_response[:200]}...")
        print()
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        print(f"\n❌ Erreur: {e}")
```

## 5. Benchmarking Robuste avec Gestion d'Erreurs

**Problème** : Chemins en dur, pas de gestion de la mémoire GPU.

**Solution** :

```python
# benchmark_models.py (amélioré)
import torch  # Pour vérifier la mémoire GPU
import humanize  # Pour affichage lisible

def check_gpu_memory() -> float:
    """Retourne la mémoire GPU disponible en Go."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0

def run_benchmark(test_dir: Path):
    if not test_dir.exists():
        raise FileNotFoundError(f"Dossier de test introuvable: {test_dir}")
    
    # Vérifier mémoire disponible
    gpu_mem = check_gpu_memory()
    if gpu_mem > 0:
        logger.info(f"Mémoire GPU détectée: {gpu_mem:.1f} Go")
    
    # ... suite du code ...
    
    for model in MODELS_TO_TEST:
        # Estimer la taille du modèle
        model_size_gb = estimate_model_size(model)  # À implémenter
        
        if gpu_mem > 0 and model_size_gb > gpu_mem * 0.9:
            logger.warning(f"Modèle {model} trop grand pour la mémoire GPU ({model_size_gb:.1f} Go > {gpu_mem:.1f} Go)")
            continue
        
        # ... processus de benchmark avec timeout ...
```

## 6. Tests Complétés et Fiabilisés

**Problème** : `test_rag_automated.py` dépend d'un environnement local non reproductible.

**Solution** : Utilisez des fixtures temporaires :

```python
# tests/test_rag_automated.py
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="module")
def test_environment(tmp_path_factory):
    """Crée un environnement de test complet avec des documents factices."""
    base_dir = tmp_path_factory.mktemp("rag_test")
    
    # Créer des documents de test
    processed_dir = base_dir / "processed_md"
    processed_dir.mkdir()
    
    (processed_dir / "test_doc.md").write_text("""
    # Test Document
    
    Ceci est un document de test avec des informations spécifiques.
    Le montant minimum du salaire imposé est 1600 euros.
    """)
    
    db_dir = base_dir / "chroma_db"
    
    yield {
        "processed_dir": processed_dir,
        "db_dir": db_dir,
    }
    
    # Nettoyage automatique par pytest

def test_end_to_end_rag(test_environment):
    """Test E2E complet."""
    config = RAGConfig(
        processed_dir=test_environment["processed_dir"],
        db_path=test_environment["db_dir"],
    )
    
    # Indexation
    vector_db = load_or_initialize_vector_db(config)
    update_vector_db_incrementally(vector_db, config)
    
    # Requête
    doc_chain, retriever = setup_rag_chain(vector_db, config)
    question = "Quel est le montant minimum du salaire imposé ?"
    
    relevant_docs = retriever.invoke(question)
    response = doc_chain.invoke({"input": question, "context": relevant_docs})
    
    assert "1600" in response
```

## 7. Ajouts au `pyproject.toml`

**Problèmes** :
- `langchain_classic` n'existe pas (c'est `langchain`)
- `uv` ne gère pas les dépendances dev correctement avec `tool.uv`

**Solution** :

```toml
[project]
name = "rag-ollama"
version = "0.1.0"
description = "RAG system with Ollama, local embeddings, and AI-powered OCR"
readme = "README.md"
requires-python = ">=3.10"  # 3.13 est trop restrictif
dependencies = [
    "docling>=2.8,<3.0",
    "langchain>=0.2,<0.3",
    "langchain-community>=0.2,<0.3",
    "langchain-ollama>=0.2,<0.3",
    "langchain-chroma>=0.1,<0.2",
    "pydantic>=2.0,<3.0",
    "chromadb>=0.5,<0.6",
    "rank_bm25>=0.2,<0.3",
    "markdown>=3.5,<4.0",
    "requests>=2.31,<3.0",
    "ollama>=0.3,<0.4",
    "pdf-ocr-ai @ git+https://github.com/laurentvv/pdf-ocr-ai",
    "pyyaml>=6.0,<7.0",
    "tqdm>=4.66,<5.0",
    "humanize>=4.0,<5.0",  # Pour le benchmark
    "torch>=2.0",  # Pour détection GPU
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "black>=24.0",
    "ruff>=0.5",  # Remplacement moderne de flake8
]

[project.scripts]
rag-prepare = "rag_ollama.prepare_documents:main"
rag-chat = "rag_ollama.rag:main"
rag-add = "rag_ollama.add_document:main"
rag-benchmark = "rag_ollama.benchmark:main"  # Renommé pour plus de clarté

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## 8. Documentation README.md Améliorée

```markdown
## 🔧 Installation Développeur

```bash
# Cloner le repo
git clone https://github.com/laurentvv/rag-ollama
cd rag-ollama

# Installer avec uv (recommandé)
uv sync --extra dev

# Configuration initiale
cp config.yaml.example config.yaml
# Éditer config.yaml selon vos besoins

# Télécharger les modèles par défaut
make pull-models  # Ajoutez un Makefile avec les commandes ollama pull
```

## 📊 Benchmark Personnalisé

```bash
# Benchmark rapide avec vos propres documents
uv run rag-benchmark --dir "./mes_documents" --models qwen3-vl:8b llama3.2-vision

# Avec rapport HTML
uv run rag-benchmark --dir "./mes_documents" --output-format html --save-report
```

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| `OllamaError: Connection refused` | Vérifiez qu'Ollama tourne: `ollama list` |
| `Model unavailable` | Téléchargez le modèle: `ollama pull <nom>` |
| Timeout sur PDF | Augmentez `OPENAI_TIMEOUT` ou réduisez la taille du PDF |
| Erreurs d'encodage | Les fichiers sont sauvegardés en UTF-8 avec remplacement des caractères invalides |
```

## 9. Makefile pour Simplifier les Tâches Courantes

```makefile
# Makefile
.PHONY: install dev-install test benchmark pull-models clean

install:
	uv sync

dev-install:
	uv sync --extra dev

test:
	uv run pytest tests/ -v --cov=src/rag_ollama

benchmark:
	uv run python benchmark_models.py --dir "./test_docs"

pull-models:
	ollama pull gemma3:12b
	ollama pull embeddinggemma:latest
	ollama pull llama3.2-vision

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf chroma_db/ processed_md/ benchmark_output/
```

## 10. Corrections de Bugs Critiques

### Bug 1 : `langchain_classic` inexistant
Dans `rag.py`, remplacez :
```python
# ❌ Mauvais imports
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever

# ✅ Bons imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
```

### Bug 2 : `yaml_conf` non utilisé
Dans tous les fichiers, utilisez la fonction `load_config()` mentionnée au point 1.

### Bug 3 : `benchmark_models.py` dépend de `prepare_documents`
Modifiez l'import pour qu'il fonctionne depuis n'importe où :

```python
# benchmark_models.py
import sys
from pathlib import Path

# Ajoute le src au path de manière robuste
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))

import rag_ollama.prepare_documents as prep_docs
```

---

## Résumé des Actions Prioritaires

1. ✅ **Implémenter le chargement de `config.yaml`** (copier le code du point 1)
2. ✅ **Corriger les imports `langchain_classic` ** immédiatement
3. ✅ ** Créer `OllamaManager` ** pour centraliser les vérifications
4. ✅ ** Renforcer la sécurité des processeurs ** (validation des chemins)
5. ✅ ** Ajouter `uv.lock` ** au `.gitignore` pour les projets uv
6. ✅ ** Comparer les dépendances ** : `uv pip list` vs `pyproject.toml`

Ce refactoring rendra votre projet plus robuste, maintenable et prêt pour une utilisation en production. Les tests deviendront reproductibles sur n'importe quelle machine avec `pytest`.