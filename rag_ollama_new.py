import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- Configuration ---
PROCESSED_DIR = "./processed_md"  # Dossier contenant les fichiers Markdown traités par Docling
CHROMA_DB_PATH = "./chroma_db"    # Chemin où la base de données Chroma sera stockée
OLLAMA_MODEL = "llama2"           # Modèle Ollama à utiliser (par exemple, "llama2", "mistral", "gemma")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Modèle d'embeddings

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

def full_text_search(query, chunks, top_k=2):
    """
    Effectue une recherche plein texte simple sur les chunks.
    """
    results = []
    for chunk in chunks:
        score = 0
        # Compte les occurrences de chaque mot de la query dans le contenu du chunk
        for word in query.lower().split():
            score += chunk.page_content.lower().count(word)

        if score > 0:
            results.append({"chunk": chunk, "score": score})

    # Trie les résultats par score décroissant et retourne le top_k
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    # Retourne uniquement les chunks
    return [res["chunk"] for res in sorted_results[:top_k]]


# --- 2. Création ou Chargement de la Base de Données Vectorielle ---
def setup_vector_db(chunks):
    print("Initialisation du modèle d'embeddings...")
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    # Pour utiliser CUDA si disponible
    # embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cuda'})

    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print(f"Chargement de la base de données Chroma existante depuis {CHROMA_DB_PATH}...")
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        # Optionnel: Si vous voulez ajouter de nouveaux chunks, vous pouvez faire:
        # vector_db.add_documents(chunks)
    else:
        print(f"Création d'une nouvelle base de données Chroma dans {CHROMA_DB_PATH}...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        vector_db.persist() # Sauvegarde la DB sur le disque
        print("Base de données Chroma créée et persistée.")

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
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    return document_chain, retriever

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
            # 1. Vector search
            vector_results = retriever.get_relevant_documents(question)

            # 2. Full-text search
            full_text_results = full_text_search(question, chunks, top_k=2)

            # 3. Combine and deduplicate results
            combined_results = vector_results + full_text_results
            unique_results = []
            seen_content = set()
            for doc in combined_results:
                if doc.page_content not in seen_content:
                    unique_results.append(doc)
                    seen_content.add(doc.page_content)

            # 4. Invoke RAG chain with combined context
            response = document_chain.invoke({"input": question, "context": unique_results})
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
