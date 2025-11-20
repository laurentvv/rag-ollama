import os
import sys
from rag_ollama.rag import ingest_documents, setup_vector_db, setup_rag_chain

# Test Cases
TEST_CASES = [
    {
        "id": "PDF_OCR_1",
        "question": "Quels sont les groupes ou noms d'utilisateurs listés dans la fenêtre de propriétés système pour le bureau à distance ?",
        "expected_keywords": ["MASOCITE", "Utilisateurs du domaine"],
        "description": "Test OCR/VLM capability on PDF screenshots"
    },
    {
        "id": "DOCX_PSSI_1",
        "question": "Quelle est la date de la Politique de Sécurité des Systèmes d’Information ?",
        "expected_keywords": ["28/11/2013"],
        "description": "Test text extraction from PSSI DOCX"
    },
    {
        "id": "DOCX_PSSI_2",
        "question": "Quel est le nom de l'entité concernée par la PSSI ?",
        "expected_keywords": ["KALIDEA"],
        "description": "Test text extraction from PSSI DOCX header"
    },
    {
        "id": "DOCX_SQL_1",
        "question": "Quel est le montant minimum du salaire imposé par le trigger upd_sal ?",
        "expected_keywords": ["1600"],
        "description": "Test code/text extraction from SQL DOCX"
    },
    {
        "id": "DOCX_SQL_2",
        "question": "Sur quelle table le trigger upd_sal est-il défini ?",
        "expected_keywords": ["employes"],
        "description": "Test code/text extraction from SQL DOCX"
    }
]

def run_tests():
    print("--- Démarrage des tests automatisés du RAG ---")
    
    # 1. Setup RAG Pipeline
    print("Initialisation du pipeline RAG...")
    
    # Define paths for testing (using defaults or specific test paths)
    from pathlib import Path
    TEST_PROCESSED_DIR = Path("./processed_md")
    TEST_CHROMA_DB = Path("./chroma_db")
    
    chunks = ingest_documents(processed_dir=TEST_PROCESSED_DIR)
    if chunks is None:
        print("Erreur: Aucun document chargé.")
        return

    vector_db = setup_vector_db(chunks, chroma_db_path=TEST_CHROMA_DB)
    document_chain, retriever = setup_rag_chain(vector_db, chunks)
    
    print(f"\n--- Exécution de {len(TEST_CASES)} tests ---\n")
    
    passed_count = 0
    
    for test in TEST_CASES:
        print(f"Test ID: {test['id']}")
        print(f"Description: {test['description']}")
        print(f"Question: {test['question']}")
        
        try:
            # Retrieve documents (Hybrid Search)
            relevant_docs = retriever.invoke(test['question'])
            
            # Generate Answer
            response = document_chain.invoke({"input": test['question'], "context": relevant_docs})
            
            print(f"Réponse RAG: {response}")
            
            # Verification
            missing_keywords = []
            for keyword in test['expected_keywords']:
                if keyword.lower() not in response.lower():
                    missing_keywords.append(keyword)
            
            if not missing_keywords:
                print("Résultat: [SUCCÈS]")
                passed_count += 1
            else:
                print(f"Résultat: [ÉCHEC] - Mots-clés manquants: {missing_keywords}")
                
        except Exception as e:
            print(f"Résultat: [ERREUR] - Exception: {e}")
            
        print("-" * 50)

    print(f"\n--- Résumé des tests ---")
    print(f"Total: {len(TEST_CASES)}")
    print(f"Succès: {passed_count}")
    print(f"Échecs: {len(TEST_CASES) - passed_count}")
    
    if passed_count == len(TEST_CASES):
        print("\n>>> TOUS LES TESTS ONT RÉUSSI <<<")
    else:
        print("\n>>> CERTAINS TESTS ONT ÉCHOUÉ <<<")

if __name__ == "__main__":
    # Suppress LangChain deprecation warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    run_tests()
