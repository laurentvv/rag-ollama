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
PROCESSED_DIR = "./processed_md"  # Dossier contenant les fichiers Markdown traités par Docling
CHROMA_DB_PATH = "./chroma_db"    # Chemin où la base de données Chroma sera stockée
OLLAMA_MODEL = "gemma3:12b"           # Modèle Ollama à utiliser (par exemple, "llama2", "mistral", "gemma")
EMBEDDING_MODEL_NAME = "embeddinggemma:latest" # Modèle d'embeddings

# --- 1. Ingestion et Traitement des Documents ---
def ingest_documents():
    print(f"Chargement des documents Markdown depuis {PROCESSED_DIR}...")

    if not os.path.exists(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
        print(f"Le dossier '{PROCESSED_DIR}' est vide ou n'existe pas. Veuillez exécuter 'prepare_documents.py' d'abord.")
        return None

    # Utilise DirectoryLoader pour charger tous les fichiers .md
    # et UnstructuredMarkdownLoader pour les parser efficacement.
    loader = DirectoryLoader(
        PROCESSED_DIR,
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




# --- 2. Création ou Chargement de la Base de Données Vectorielle ---
def setup_vector_db(chunks):
    print("Initialisation du modèle d'embeddings...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print(f"Chargement de la base de données Chroma existante depuis {CHROMA_DB_PATH}...")
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        
        # Check if we should re-index (simple heuristic: if user asks or if we want to force it)
        # For now, we'll just print a message. In a real app, we might check file timestamps.
        print("Note: Pour forcer la réindexation, supprimez le dossier 'chroma_db' ou utilisez une option de commande (à implémenter).")
        
    else:
        print(f"Création d'une nouvelle base de données Chroma dans {CHROMA_DB_PATH}...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        # vector_db.persist() # Chroma 0.4+ persists automatically usually, but good to keep if using older version
        print("Base de données Chroma créée.")

    return vector_db

# --- 3. Configuration du RAG Chain ---
def setup_rag_chain(vector_db, chunks):
    print(f"Configuration du modèle Ollama: {OLLAMA_MODEL}...")
    llm = Ollama(model=OLLAMA_MODEL)

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

# --- Fonction principale ---
def main():
    chunks = ingest_documents()
    if chunks is None:
        return # Arrête si aucun document n'a été chargé

    vector_db = setup_vector_db(chunks)
    document_chain, retriever = setup_rag_chain(vector_db, chunks)

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
