# TODO

## Finalise test bench modèle
- [ ] benchmark_models.py à lancer et valider

## Documentation Improvements
# 📝 Documentation & Polish

## Documentation Improvements
- [ ] **Explain `uvx` parameters**: Detail what each argument (`--input`, `--output`, `--db`, etc.) does in the `README.md`.
- [ ] **Vision Model Details**: Add a dedicated section explaining the default vision model https://ollama.com/library/qwen3-vlhttps://ollama.com/library/qwen3-vl : qwen3-vl:8b or qwen3-vl:30b and how `pdf-ocr-ai` is used.
- [ ] **Workflow Clarity**: Ensure the "Workflow" section clearly explains *why* each step is needed.

## Future Improvements
- [ ] **Error Handling**: Add better error messages for missing models or wrong paths.
- [x] Implement `config.yaml` for default settings (model, paths)
- [x] Add error handling for missing Ollama/Models
- [x] Full Process Validation (End-to-End Test)rform a complete end-to-end test of the process (Prepare -> Chat) to validate the integration.