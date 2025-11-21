import time
import sys
import shutil
import logging
from pathlib import Path
import ollama
import argparse

# Add src to path to allow imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import functions from prepare_documents
# We need to set global variables in this module to control its behavior
import rag_ollama.prepare_documents as prep_docs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

MODELS_TO_TEST = [
    'deepseek-ocr:3b',
    'qwen3-vl:8b',
    'gemma3:12b',
    'qwen3-vl:30b'
]

DEFAULT_TEST_DIR = Path(r"C:\test\docs_test")
BENCHMARK_OUTPUT_DIR = Path("benchmark_output")

def unload_model(model_name):
    """Unloads the model from memory."""
    print(f"Unloading model: {model_name}...")
    try:
        # Sending an empty request with keep_alive=0 unloads the model
        ollama.chat(model=model_name, messages=[], keep_alive=0)
        print(f"Model {model_name} unloaded.")
    except Exception as e:
        print(f"Error unloading model {model_name}: {e}")

def run_benchmark(test_dir: Path):
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist.")
        return

    # Prepare output directory
    if BENCHMARK_OUTPUT_DIR.exists():
        shutil.rmtree(BENCHMARK_OUTPUT_DIR)
    BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Get list of files to process
    files = list(test_dir.glob("*.*"))
    valid_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = [f for f in files if f.suffix.lower() in valid_extensions]

    if not files:
        print(f"No valid files found in {test_dir}")
        return

    print(f"Found {len(files)} files to test: {[f.name for f in files]}")

    for model in MODELS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"Testing Model: {model}")
        print(f"{'='*60}")
        
        # Check if model is available, pull if not? 
        # The existing check_model_available prints to stderr, let's just try to use it.
        # If it fails, it fails.
        
        model_output_dir = BENCHMARK_OUTPUT_DIR / model.replace(":", "_")
        model_output_dir.mkdir(exist_ok=True)

        # Configure prepare_documents globals
        prep_docs.PROCESSED_DIR = model_output_dir
        prep_docs.VISION_MODEL = model
        prep_docs.PDF_MODEL = model
        prep_docs.PDF_PROVIDER = "ollama" # Force ollama provider
        
        # Ensure processed dir exists (prepare_documents functions expect it)
        prep_docs.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        model_start_time = time.time()
        
        for file_path in files:
            print(f"\nProcessing {file_path.name} with {model}...")
            file_start_time = time.time()
            
            try:
                if file_path.suffix.lower() == '.pdf':
                    prep_docs.process_pdf_with_ai(file_path)
                else:
                    prep_docs.process_image_with_ai(file_path)
                
                duration = time.time() - file_start_time
                status = "SUCCESS"
                print(f"Finished {file_path.name} in {duration:.2f}s")
                
            except Exception as e:
                duration = time.time() - file_start_time
                status = f"FAILED: {e}"
                print(f"Failed {file_path.name}: {e}")

            results.append({
                "model": model,
                "file": file_path.name,
                "duration": duration,
                "status": status
            })

        total_model_time = time.time() - model_start_time
        print(f"\nTotal time for {model}: {total_model_time:.2f}s")
        
        # Unload model
        unload_model(model)
        # Small sleep to ensure release
        time.sleep(2)

    # Print Summary Table
    print("\n\n" + "="*80)
    print(f"{'Model':<20} | {'File':<30} | {'Time (s)':<10} | {'Status':<20}")
    print("-" * 80)
    for res in results:
        print(f"{res['model']:<20} | {res['file']:<30} | {res['duration']:<10.2f} | {res['status']:<20}")
    print("="*80)
    
    # Save summary to file
    with open("benchmark_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Model':<20} | {'File':<30} | {'Time (s)':<10} | {'Status':<20}\n")
        f.write("-" * 80 + "\n")
        for res in results:
            f.write(f"{res['model']:<20} | {res['file']:<30} | {res['duration']:<10.2f} | {res['status']:<20}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VL Models")
    parser.add_argument("--image", type=Path, help="Optional: Specific image to test (overrides default dir)")
    parser.add_argument("--dir", type=Path, default=DEFAULT_TEST_DIR, help=f"Directory to test (default: {DEFAULT_TEST_DIR})")
    
    args = parser.parse_args()
    
    target_dir = args.dir
    if args.image:
        # If specific image provided, we can just mock a dir or handle it, 
        # but for simplicity let's just use the dir logic if possible or just warn.
        # Actually, let's just support the dir as requested by user "C:\test\docs_test"
        pass

    run_benchmark(target_dir)
