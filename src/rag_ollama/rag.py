import sys
import ollama
import yaml
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from pathlib import Path
import argparse
from tqdm import tqdm
from .config import RAGConfig
from .utils.logging import logger
from .utils.exceptions import RAGException, OllamaError, ModelUnavailableError
from .utils.hash import get_file_hash

def load_or_initialize_vector_db(config: RAGConfig):
    logger.info(f"Initialisation du modèle d'embeddings: {config.embedding_model}...")
    embeddings = OllamaEmbeddings(model=config.embedding_model)
    db_path = str(config.db_path)
    
    logger.info(f"Chargement/Création de la base de données Chroma depuis {db_path}...")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    logger.info("Base de données Chroma prête.")
    return vector_db

def update_vector_db_incrementally(vector_db: Chroma, config: RAGConfig):
    logger.info(f"Vérification des documents dans {config.processed_dir} pour l'indexation...")
    
    processed_files = list(config.processed_dir.glob("*.md"))
    if not processed_files:
        logger.warning("Aucun document .md trouvé. L'indexation est sautée.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file_path in tqdm(processed_files, desc="Indexation incrémentale"):
        file_hash = get_file_hash(file_path)
        
        # Vérifier si ce hash existe déjà pour ce fichier
        existing_docs = vector_db.get(where={"source": str(file_path), "hash": file_hash})
        if existing_docs and existing_docs['ids']:
            continue # Fichier déjà indexé et non modifié

        # Si le fichier a été modifié (hash différent), supprimer l'ancienne version
        old_docs = vector_db.get(where={"source": str(file_path)})
        if old_docs and old_docs['ids']:
            vector_db.delete(ids=old_docs['ids'])
            logger.info(f"Document '{file_path.name}' modifié détecté, ancienne version supprimée.")

        # Charger, traiter et ajouter le nouveau document/version
        logger.info(f"Indexation du nouveau/modifié document: {file_path.name}")
        loader = UnstructuredMarkdownLoader(str(file_path))
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["id"] = f"{str(file_path)}:{i}"
            chunk.metadata["source"] = str(file_path)
            chunk.metadata["hash"] = file_hash

        vector_db.add_documents(chunks, ids=[c.metadata["id"] for c in chunks])

def setup_rag_chain(vector_db: Chroma, config: RAGConfig):
    logger.info(f"Configuration du modèle Ollama: {config.llm_model}...")
    llm = Ollama(model=config.llm_model)
    prompt = ChatPromptTemplate.from_template("Répondez à la question en vous basant sur le contexte.\n\nContexte: {context}\n\nQuestion: {input}")
    document_chain = create_stuff_documents_chain(llm, prompt)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    logger.info("Récupération de tous les documents pour BM25...")
    all_docs = vector_db.get(include=["documents"])['documents']
    
    logger.info("Initialisation de BM25Retriever...")
    bm25_retriever = BM25Retriever.from_texts(all_docs)
    bm25_retriever.k = 3
    
    logger.info("Création de l'EnsembleRetriever...")
    return document_chain, EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

# ... (fonctions de vérification et de configuration restent les mêmes) ...

def main():
    try:
        # ... (parsing des arguments reste le même) ...
        # NOTE: Pour la simplicité, je vais le copier ici.
        yaml_conf = {} # load_config_from_yaml().get("chat", {})
        parser = argparse.ArgumentParser(description="RAG System with Ollama")
        parser.add_argument("--index-only", action="store_true")
        parser.add_argument("--input", "-i", type=Path, required=True)
        parser.add_argument("--db", type=Path, required=True)
        parser.add_argument("--model", default=yaml_conf.get("llm_model", "gemma3:12b"))
        parser.add_argument("--embedding-model", default=yaml_conf.get("embedding_model", "embeddinggemma:latest"))
        args = parser.parse_args()

        if not args.input.exists(): raise RAGException(f"Dossier d'entrée '{args.input}' introuvable.")

        config = RAGConfig(source_dir=Path(), processed_dir=args.input.resolve(), db_path=args.db.resolve(), llm_model=args.model, embedding_model=args.embedding_model)

        # check_ollama_available()
        # check_model_available(config.llm_model)
        # check_model_available(config.embedding_model)

        vector_db = load_or_initialize_vector_db(config)
        update_vector_db_incrementally(vector_db, config)

        if args.index_only:
            logger.info("Indexation incrémentale terminée.")
            return

        doc_chain, retriever = setup_rag_chain(vector_db, config)
        print("\n--- Prêt à interagir ---")

        while True:
            question = input("\nVotre question: ")
            if question.lower() == 'exit': break
            
            try:
                relevant_docs = retriever.invoke(question)
                print("\nRéponse: ", end="", flush=True)
                for chunk in doc_chain.stream({"input": question, "context": relevant_docs}):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                logger.error(f"Erreur lors de la génération de la réponse: {e}")

    except RAGException as e:
        logger.error(f"Erreur RAG: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
