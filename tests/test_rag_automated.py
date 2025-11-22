import os
import sys
from pathlib import Path
# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_ollama.rag import setup_rag_chain, load_or_initialize_vector_db
from src.rag_ollama.config import RAGConfig

# Test Cases - Kept from original file
TEST_CASES = [
    {"id": "PDF_OCR_1", "question": "Quels sont les groupes ou noms d'utilisateurs listés ?", "expected_keywords": ["MASOCITE", "Utilisateurs du domaine"]},
    {"id": "DOCX_PSSI_1", "question": "Quelle est la date de la PSSI ?", "expected_keywords": ["28/11/2013"]},
    {"id": "DOCX_SQL_1", "question": "Quel est le montant minimum du salaire imposé ?", "expected_keywords": ["1600"]},
]

def run_tests():
    print("--- Démarrage des tests d'intégration du RAG ---")
    
    # Setup paths and config for testing
    TEST_PROCESSED_DIR = Path("./processed_md")
    TEST_CHROMA_DB = Path("./chroma_db")
    
    if not TEST_PROCESSED_DIR.exists() or not any(TEST_PROCESSED_DIR.iterdir()):
        print("\nERREUR: Le dossier 'processed_md' est vide. Exécutez d'abord 'rag-prepare'.")
        sys.exit(1)

    config = RAGConfig(
        source_dir=Path(), # Not needed for this test
        processed_dir=TEST_PROCESSED_DIR,
        db_path=TEST_CHROMA_DB,
        llm_model="gemma3:12b",
        embedding_model="embeddinggemma:latest"
    )

    # Use the new refactored functions
    vector_db = load_or_initialize_vector_db(config)
    # Note: We assume the DB is already indexed for this test.
    # A full e2e test would run prepare, then index, then chat.
    
    document_chain, retriever = setup_rag_chain(vector_db, config)
    
    print(f"\n--- Exécution de {len(TEST_CASES)} tests ---\n")
    passed_count = 0
    
    for test in TEST_CASES:
        print(f"Test ID: {test['id']} - Question: {test['question']}")
        
        try:
            relevant_docs = retriever.invoke(test['question'])
            response = document_chain.invoke({"input": test['question'], "context": relevant_docs})
            
            missing_keywords = [kw for kw in test['expected_keywords'] if kw.lower() not in response.lower()]
            
            if not missing_keywords:
                print("  -> Résultat: [SUCCÈS]")
                passed_count += 1
            else:
                print(f"  -> Résultat: [ÉCHEC] - Mots-clés manquants: {missing_keywords}")
                
        except Exception as e:
            print(f"  -> Résultat: [ERREUR] - Exception: {e}")
            
    print("-" * 50)
    print(f"Total: {len(TEST_CASES)}, Succès: {passed_count}, Échecs: {len(TEST_CASES) - passed_count}")

    if passed_count != len(TEST_CASES):
        sys.exit(1) # Fail the CI/CD pipeline if tests fail

if __name__ == "__main__":
    run_tests()
