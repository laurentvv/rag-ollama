# Améliorations pour le projet RAG-Ollama

Après analyse du code, voici plusieurs améliorations que je propose pour rendre le projet plus robuste, efficace et convivial :

## 1. Amélioration de la gestion de la configuration

Le fichier `config.yaml` existe mais n'est pas pleinement utilisé dans le code. Je propose de centraliser la configuration :

```python
# src/rag_ollama/config.py
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

@dataclass
class RAGConfig:
    source_dir: Path
    processed_dir: Path
    db_path: Path
    vision_model: str = "llama3.2-vision"
    pdf_model: str = "llama3.2-vision"
    llm_model: str = "gemma3:12b"
    embedding_model: str = "embeddinggemma:latest"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 3
    retriever_weights: list = field(default_factory=lambda: [0.5, 0.5])
    timeout: int = 600
    
    @classmethod
    def from_yaml(cls, config_path: Path, **kwargs) -> 'RAGConfig':
        """Charge la configuration depuis un fichier YAML et permet de surcharger avec des kwargs."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extraire les configurations pertinentes
        prepare_config = config_data.get('prepare', {})
        chat_config = config_data.get('chat', {})
        
        # Fusionner avec les valeurs par défaut et les kwargs
        default_values = {
            'vision_model': prepare_config.get('vision_model', "llama3.2-vision"),
            'pdf_model': prepare_config.get('pdf_model', "llama3.2-vision"),
            'llm_model': chat_config.get('llm_model', "gemma3:12b"),
            'embedding_model': chat_config.get('embedding_model', "embeddinggemma:latest"),
        }
        
        # Mettre à jour avec les kwargs fournis
        default_values.update(kwargs)
        
        return cls(**default_values)
```

## 2. Amélioration du traitement des documents

Ajout d'un système de cache pour éviter de retraiter les documents non modifiés :

```python
# src/rag_ollama/prepare_documents.py
import json
from pathlib import Path
from .utils.hash import get_file_hash

class ProcessingCache:
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def is_processed(self, file_path: Path) -> bool:
        file_str = str(file_path)
        if file_str not in self.cache:
            return False
        
        current_hash = get_file_hash(file_path)
        return self.cache[file_str].get('hash') == current_hash
    
    def mark_processed(self, file_path: Path, output_path: Path):
        file_str = str(file_path)
        self.cache[file_str] = {
            'hash': get_file_hash(file_path),
            'output_path': str(output_path),
            'timestamp': time.time()
        }
        self._save_cache()
    
    def get_output_path(self, file_path: Path) -> Optional[Path]:
        file_str = str(file_path)
        if file_str in self.cache:
            return Path(self.cache[file_str].get('output_path'))
        return None

def process_document(file_path: Path, config: RAGConfig, processors: list[DocumentProcessor], cache: ProcessingCache):
    """Traite un document en utilisant le cache pour éviter les retraitements."""
    # Vérifier si le fichier a déjà été traité et n'a pas été modifié
    if cache.is_processed(file_path):
        cached_output = cache.get_output_path(file_path)
        if cached_output and cached_output.exists():
            logger.info(f"Fichier {file_path.name} déjà traité et non modifié, utilisation du cache.")
            return cached_output
    
    # Trouver le bon processeur et traiter le document
    for processor in processors:
        if processor.can_process(file_path):
            try:
                logger.info(f"Traitement de {file_path.name} avec {processor.__class__.__name__}...")
                output_path = processor.process(file_path, config)
                add_filename_context(output_path, file_path.name)
                
                # Marquer comme traité dans le cache
                cache.mark_processed(file_path, output_path)
                
                return output_path
            except ProcessingError as e:
                logger.error(f"Erreur lors du traitement de {file_path.name}: {e}")
                return None
    
    logger.warning(f"Aucun processeur trouvé pour le type de fichier: {file_path.name}")
    return None
```

## 3. Amélioration de la recherche hybride

Optimisation de la recherche hybride pour de meilleurs résultats :

```python
# src/rag_ollama/rag.py
def setup_rag_chain(vector_db: Chroma, config: RAGConfig):
    logger.info(f"Configuration du modèle Ollama: {config.llm_model}...")
    llm = Ollama(model=config.llm_model)
    
    # Template de prompt amélioré
    template = """
    Vous êtes un assistant IA spécialisé dans l'analyse de documents. 
    Répondez à la question en vous basant UNIQUEMENT sur le contexte fourni.
    Si le contexte ne contient pas d'informations pertinentes, indiquez-le clairement.
    
    Contexte:
    {context}
    
    Question: {input}
    
    Réponse:
    """
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Configuration améliorée des récupérateurs
    vector_retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": config.retriever_k,
            "score_threshold": 0.5  # Seuil de pertinence
        }
    )
    
    logger.info("Récupération de tous les documents pour BM25...")
    all_docs = vector_db.get(include=["documents", "metadatas"])['documents']
    
    logger.info("Initialisation de BM25Retriever...")
    bm25_retriever = BM25Retriever.from_texts(all_docs)
    bm25_retriever.k = config.retriever_k
    
    logger.info("Création de l'EnsembleRetriever avec poids optimisés...")
    return document_chain, EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=config.retriever_weights
    )
```

## 4. Amélioration de la gestion des erreurs

Ajout d'une meilleure gestion des erreurs avec des messages plus informatifs :

```python
# src/rag_ollama/utils/exceptions.py
class RAGException(Exception):
    """Exception base pour le projet"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{super().__str__()} (Détails: {details_str})"
        return super().__str__()

class ModelUnavailableError(RAGException):
    """Le modèle demandé n'est pas disponible dans Ollama."""
    def __init__(self, model_name: str, available_models: List[str] = None):
        super().__init__(f"Modèle '{model_name}' non trouvé.", {
            "model_name": model_name,
            "available_models": available_models or []
        })
        self.suggestions = [
            "Vérifiez que le modèle est bien installé avec 'ollama pull'",
            "Vérifiez l'orthographe du nom du modèle",
            "Utilisez 'ollama list' pour voir les modèles disponibles"
        ]

class ProcessingError(RAGException):
    """Erreur lors du traitement d'un document."""
    def __init__(self, message: str, file_path: str = None, processor: str = None):
        super().__init__(message, {
            "file_path": file_path,
            "processor": processor
        })

class OllamaError(RAGException):
    """Erreur de communication avec Ollama."""
    def __init__(self, message: str, host: str = None, port: int = None):
        super().__init__(message, {
            "host": host,
            "port": port
        })
        self.suggestions = [
            "Vérifiez qu'Ollama est bien en cours d'exécution",
            "Vérifiez que le port 11434 est accessible",
            "Essayez de redémarrer Ollama"
        ]
```

## 5. Amélioration de l'interface utilisateur

Ajout d'une interface plus conviviale avec une barre de progression et des messages plus informatifs :

```python
# src/rag_ollama/rag.py
def main():
    try:
        # ... (code existant) ...
        
        if args.index_only:
            logger.info("Indexation incrémentale terminée.")
            return

        doc_chain, retriever = setup_rag_chain(vector_db, config)
        
        # Interface utilisateur améliorée
        print("\n" + "="*50)
        print("🤖 RAG-Ollama - Prêt à répondre à vos questions")
        print("="*50)
        print("Tapez 'exit' pour quitter, 'help' pour l'aide")
        print("-"*50)

        while True:
            question = input("\n❓ Votre question: ").strip()
            
            if question.lower() == 'exit':
                print("\n👋 Au revoir!")
                break
            elif question.lower() == 'help':
                print_help()
                continue
            elif not question:
                continue
            
            try:
                print("\n🔍 Recherche d'informations pertinentes...")
                relevant_docs = retriever.invoke(question)
                
                print("\n🤖 Réponse: ", end="", flush=True)
                for chunk in doc_chain.stream({"input": question, "context": relevant_docs}):
                    print(chunk, end="", flush=True)
                print("\n")
                
                # Afficher les sources
                print("\n📚 Sources utilisées:")
                for i, doc in enumerate(relevant_docs, 1):
                    source_path = Path(doc.metadata.get('source', 'Inconnue')).name
                    print(f"  {i}. {source_path}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la génération de la réponse: {e}")
                print(f"\n❌ Erreur: {e}")

    except RAGException as e:
        logger.error(f"Erreur RAG: {e}")
        if hasattr(e, 'suggestions'):
            print("\nSuggestions:")
            for suggestion in e.suggestions:
                print(f"  - {suggestion}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        sys.exit(1)

def print_help():
    """Affiche l'aide pour les commandes disponibles."""
    help_text = """
    Commandes disponibles:
    - help: Affiche cette aide
    - exit: Quitte le programme
    
    Astuces:
    - Soyez précis dans vos questions
    - Mentionnez des documents spécifiques si nécessaire
    - Pour des informations factuelles, demandez des sources
    """
    print(help_text)
```

## 6. Amélioration des tests

Ajout de tests plus complets et de tests d'intégration :

```python
# tests/test_rag_integration.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag_ollama.rag import main as rag_main
from src.rag_ollama.prepare_documents import main as prepare_main
from src.rag_ollama.config import RAGConfig

@pytest.fixture
def temp_dirs():
    """Crée des répertoires temporaires pour les tests."""
    temp_dir = tempfile.mkdtemp()
    source_dir = Path(temp_dir) / "source"
    processed_dir = Path(temp_dir) / "processed"
    db_dir = Path(temp_dir) / "db"
    
    source_dir.mkdir()
    processed_dir.mkdir()
    db_dir.mkdir()
    
    yield source_dir, processed_dir, db_dir
    
    # Nettoyage
    shutil.rmtree(temp_dir)

@pytest.mark.integration
def test_end_to_end_workflow(temp_dirs):
    """Test du workflow complet: préparation -> indexation -> interrogation."""
    source_dir, processed_dir, db_dir = temp_dirs
    
    # Créer un document de test
    test_file = source_dir / "test.txt"
    test_file.write_text("Ceci est un document de test sur l'intelligence artificielle.")
    
    # Simuler la préparation des documents
    with patch('sys.argv', ['rag-prepare', '--input', str(source_dir), '--output', str(processed_dir)]):
        with patch('src.rag_ollama.prepare_documents.check_ollama_available'):
            with patch('src.rag_ollama.prepare_documents.check_model_available'):
                prepare_main()
    
    # Vérifier que le document a été traité
    processed_files = list(processed_dir.glob("*.md"))
    assert len(processed_files) == 1
    
    # Simuler l'indexation et l'interrogation
    with patch('sys.argv', ['rag-chat', '--input', str(processed_dir), '--db', str(db_dir), '--index-only']):
        with patch('src.rag_ollama.rag.OllamaEmbeddings'):
            with patch('src.rag_ollama.rag.Chroma'):
                rag_main()
    
    # Vérifier que la base de données a été créée
    assert db_dir.exists()
```

## 7. Amélioration de la documentation

Ajout d'une documentation plus complète dans le README :

```markdown
# 🧠 RAG-Ollama

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Integrated-orange)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-green)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-purple)](https://github.com/laurentvv/rag-ollama/blob/main/LICENSE)

**Système RAG local avancé avec OCR IA et recherche hybride.**
*Alimenté par Ollama, ChromaDB et les modèles de langage vision.*

## Table des matières

1. [Fonctionnalités](#-fonctionnalités)
2. [Prérequis](#-prérequis)
3. [Installation](#-installation)
4. [Guide de démarrage rapide](#-guide-de-démarrage-rapide)
5. [Configuration avancée](#-configuration-avancée)
6. [Utilisation des modèles](#-utilisation-des-modèles)
7. [Dépannage](#-dépannage)
8. [Contribuer](#-contribuer)
9. [Licence](#-licence)

## 🚀 Guide de démarrage rapide

### Étape 1: Installation des dépendances

```bash
# Installer uv (gestionnaire de paquets Python rapide)
pip install uv

# Cloner le dépôt
git clone https://github.com/laurentvv/rag-ollama.git
cd rag-ollama

# Installer les dépendances
uv sync
```

### Étape 2: Configuration d'Ollama

```bash
# Installer Ollama (suivez les instructions sur ollama.com)
# Pull des modèles par défaut
ollama pull gemma3:12b
ollama pull embeddinggemma:latest
ollama pull llama3.2-vision
```

### Étape 3: Traitement des documents

```bash
# Traiter tous les documents dans un dossier
uv run rag-prepare --input "/chemin/vers/vos/documents" --output "./processed_md"
```

### Étape 4: Démarrer une conversation

```bash
# Lancer l'interface de conversation
uv run rag-chat --input "./processed_md" --db "./chroma_db"
```

## 🔧 Configuration avancée

### Personnalisation avec config.yaml

Vous pouvez créer un fichier `config.yaml` pour personnaliser les paramètres par défaut :

```yaml
# RAG-Ollama Configuration
prepare:
  vision_model: "llama3.2-vision"
  pdf_provider: "ollama"
  pdf_model: "llama3.2-vision"

chat:
  llm_model: "gemma3:12b"
  embedding_model: "embeddinggemma:latest"
  chunk_size: 1000
  chunk_overlap: 200
  retriever_k: 3
  retriever_weights: [0.5, 0.5]
```

### Utilisation de la configuration

```bash
# Utiliser un fichier de configuration personnalisé
uv run rag-chat --config "./mon_config.yaml" --input "./processed_md" --db "./chroma_db"
```

## 📚 Exemples d'utilisation

### Question-réponse sur des documents

```
❓ Votre question: Quelles sont les principales conclusions du rapport trimestriel?

🤖 Réponse: Selon le rapport trimestriel, les principales conclusions sont...
   1. Augmentation des revenus de 15%
   2. Expansion sur de nouveaux marchés
   3. Lancement de trois nouveaux produits

📚 Sources utilisées:
  1. rapport_T3_2023.pdf
  2. presentation_investisseurs.pptx
```

### Analyse de documents visuels

```
❓ Votre question: Que montre le diagramme dans la page 5 du rapport?

🤖 Réponse: Le diagramme à la page 5 montre l'évolution des ventes par trimestre...
   Il illustre une tendance à la hausse avec une accélération notable au T3...

📚 Sources utilisées:
  1. rapport_annuel_2023.pdf
```

## 🛠️ Dépannage

### Problèmes courants

1. **Modèle non trouvé**
   ```
   Erreur: Modèle 'nom_du_modele' non trouvé.
   ```
   Solution: Installez le modèle avec `ollama pull nom_du_modele`

2. **Erreur de connexion à Ollama**
   ```
   Erreur: Impossible de communiquer avec Ollama.
   ```
   Solution: Vérifiez qu'Ollama est en cours d'exécution avec `ollama list`

3. **Traitement lent des documents**
   Solution: Essayez avec un modèle plus petit ou augmentez le timeout avec `--timeout`

### Obtenir de l'aide

Pour obtenir de l'aide sur une commande spécifique:
```bash
uv run rag-chat --help
uv run rag-prepare --help
uv run rag-add --help
```
```

## Conclusion

Ces améliorations visent à rendre le projet RAG-Ollama plus robuste, convivial et efficace. Les points clés sont :

1. Centralisation de la configuration avec un meilleur support du fichier YAML
2. Optimisation du traitement des documents avec un système de cache
3. Amélioration de la recherche hybride pour de meilleurs résultats
4. Gestion des erreurs plus informative avec des suggestions
5. Interface utilisateur plus conviviale avec des messages clairs
6. Tests plus complets pour assurer la fiabilité
7. Documentation améliorée pour faciliter l'utilisation

Ces modifications devraient considérablement améliorer l'expérience utilisateur tout en rendant le code plus maintenable et évolutif.