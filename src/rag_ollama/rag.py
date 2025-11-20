import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# --- Configuration ---
# --- Configuration ---
from pathlib import Path

# --- Configuration ---
# On suppose que le script est lancé depuis la racine du projet
PROCESSED_DIR = None
CHROMA_DB_PATH = None
OLLAMA_MODEL = "gemma3:12b"           # Modèle Ollama à utiliser (par exemple, "llama2", "mistral", "gemma")
EMBEDDING_MODEL_NAME = "embeddinggemma:latest" # Modèle d'embeddings

# --- 1. Ingestion et Traitement des Documents ---
# --- 1. Ingestion et Traitement des Documents ---
def ingest_documents(processed_dir=None):
    target_dir = processed_dir or PROCESSED_DIR
    print(f"Chargement des documents Markdown depuis {target_dir}...")

    if not target_dir or not os.path.exists(target_dir) or not os.listdir(target_dir):
        print(f"Le dossier '{target_dir}' est vide ou n'existe pas. Veuillez exécuter 'prepare_documents.py' d'abord.")
        return None

    # Utilise DirectoryLoader pour charger tous les fichiers .md
    # et UnstructuredMarkdownLoader pour les parser efficacement.
    loader = DirectoryLoader(
        str(target_dir),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True # Peut accélérer le chargement sur de nombreux fichiers
    )
    documents = loader.load()
    print(f"Chargé {len(documents)} documents.")

    # Séparateur de texte intelligent pour le Markdown
    # Conserve la structure du document générée par Docling
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Priorise les doubles sauts de ligne pour les paragraphes
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Divisé en {len(chunks)} chunks.")
    return chunks

def calculate_chunk_ids(chunks):
    """
    Génère des IDs uniques et stables pour chaque chunk (Source + Index).
    Cela permet d'éviter les doublons dans ChromaDB lors des mises à jour.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        # On utilise le nom du fichier comme identifiant de base
        # (Note: Docling ou DirectoryLoader met le chemin complet, on garde ça pour l'unicité)
        current_page_id = source

        # Si c'est la même page/fichier que le précédent, on incrémente l'index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # ID format: "path/to/file.md:0", "path/to/file.md:1", etc.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Ajout de l'ID aux métadonnées (optionnel mais utile) et retour
        chunk.metadata["id"] = chunk_id
    
    return chunks


# --- 2. Création ou Chargement de la Base de Données Vectorielle ---
def setup_vector_db(chunks, chroma_db_path=None, embedding_model_name=None):
    db_path = str(chroma_db_path or CHROMA_DB_PATH)
    emb_model = embedding_model_name or EMBEDDING_MODEL_NAME
    
    print(f"Initialisation du modèle d'embeddings: {emb_model}...")
    embeddings = OllamaEmbeddings(model=emb_model)

    # Calcul des IDs stables pour éviter les doublons
    chunks_with_ids = calculate_chunk_ids(chunks)
    ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    
    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"Chargement de la base de données Chroma existante depuis {db_path}...")
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Mise à jour incrémentale
        print("Mise à jour de la base de données avec les nouveaux documents...")
        # Chroma mettra à jour les documents existants si les IDs correspondent
        vector_db.add_documents(documents=chunks_with_ids, ids=ids)
        print(f"Ajouté/Mis à jour {len(chunks)} chunks dans la base.")
        
    else:
        print(f"Création d'une nouvelle base de données Chroma dans {db_path}...")
        vector_db = Chroma.from_documents(
            documents=chunks_with_ids,
            embedding=embeddings,
            ids=ids,
            persist_directory=db_path
        )
        print("Base de données Chroma créée.")

    return vector_db

# --- 3. Configuration du RAG Chain ---
def setup_rag_chain(vector_db, chunks, ollama_model=None):
    model_name = ollama_model or OLLAMA_MODEL
    print(f"Configuration du modèle Ollama: {model_name}...")
    llm = Ollama(model=model_name)

    # Définition du prompt pour le LLM
    prompt = ChatPromptTemplate.from_template("""Répondez à la question suivante en vous basant uniquement sur le contexte fourni.
    Si la réponse ne peut pas être trouvée dans le contexte, dites que vous ne savez pas.

    Contexte: {context}

    Question: {input}
    """)

    # Crée la chaîne pour combiner les documents avec le prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Crée le retriever à partir de la base de données vectorielle
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Crée le retriever BM25
    print("Initialisation de BM25Retriever...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    # Crée l'EnsembleRetriever
    print("Création de l'EnsembleRetriever (Hybrid Search)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5] # Poids égal pour le moment
    )

    return document_chain, ensemble_retriever

import argparse

# --- Fonction principale ---
def main():
    global PROCESSED_DIR, CHROMA_DB_PATH, OLLAMA_MODEL, EMBEDDING_MODEL_NAME
    
    parser = argparse.ArgumentParser(description="RAG System with Ollama")
    parser.add_argument("--index-only", action="store_true", help="Run indexing only and exit")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Directory containing processed Markdown files")
    parser.add_argument("--db", type=Path, required=True, help="Path to ChromaDB directory")
    parser.add_argument("--model", type=str, default="gemma3:12b", help="Ollama LLM model name (default: gemma3:12b)")
    parser.add_argument("--embedding-model", type=str, default="embeddinggemma:latest", help="Ollama embedding model name (default: embeddinggemma:latest)")
    
    args = parser.parse_args()
    
    PROCESSED_DIR = args.input
    CHROMA_DB_PATH = args.db
    OLLAMA_MODEL = args.model
    EMBEDDING_MODEL_NAME = args.embedding_model
    
    print(f"Source Documents: {PROCESSED_DIR}")
    print(f"Vector Database: {CHROMA_DB_PATH}")
    print(f"LLM Model: {OLLAMA_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")

    print(f"LLM Model: {OLLAMA_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")

    chunks = ingest_documents(processed_dir=PROCESSED_DIR)
    if chunks is None:
        return # Arrête si aucun document n'a été chargé

    vector_db = setup_vector_db(chunks, chroma_db_path=CHROMA_DB_PATH, embedding_model_name=EMBEDDING_MODEL_NAME)
    
    if args.index_only:
        print("Indexation terminée. Mode --index-only activé, arrêt du script.")
        return

    document_chain, retriever = setup_rag_chain(vector_db, chunks, ollama_model=OLLAMA_MODEL)

    print("\n--- Prêt à interagir avec le RAG ---")
    print(f"Modèle Ollama: {OLLAMA_MODEL}")
    print(f"Base de données Chroma: {CHROMA_DB_PATH}")
    print("Tapez 'exit' pour quitter.")

    while True:
        question = input("\nVotre question: ")
        if question.lower() == 'exit':
            break

        print("Recherche de réponse...")
        try:
            # Utilisation de l'EnsembleRetriever qui fait déjà la combinaison
            # retrieval_chain = create_retrieval_chain(retriever, document_chain) # Si on utilisait la nouvelle API
            
            # On récupère les documents pertinents manuellement pour voir ce qui se passe (optionnel)
            # docs = retriever.invoke(question)
            
            # Exécution de la chaîne
            # Note: create_stuff_documents_chain attend 'context' et 'input'
            
            # Récupération des documents via l'ensemble retriever
            relevant_docs = retriever.invoke(question)
            
            # Génération de la réponse
            response = document_chain.invoke({"input": question, "context": relevant_docs})
            print("\nRéponse:")
            print(response)

            # Pour voir les sources récupérées
            # print("\nSources utilisées:")
            # for doc in unique_results:
            #      print(f"- {doc.metadata.get('source', 'Inconnu')}")


        except Exception as e:
            print(f"Une erreur est survenue: {e}")
            print("Assurez-vous qu'Ollama est en cours d'exécution et que le modèle est téléchargé.")

if __name__ == "__main__":
    main()
