# ✅ TODO - Projet PDF OCR AI

## 1. Traitement des images courantes
- [ ] Support des formats **JPEG**, **TIFF**, **PNG** et autres formats courants
- [ ] Vérifier la compatibilité avec la logique actuelle (conversion PDF → image)
- [ ] Tester l’intégration avec ma [solution existante](https://github.com/laurentvv/pdf-ocr-ai) du projet
- [ ] Ajouter un **schéma d’architecture** (pipeline OCR → indexation → recherche)
- [ ] Expliquer les **pré-requis** et la configuration
- [ ] Fournir des **exemples d’utilisation** (commandes, API)

---

## 2. Gestion de l’ajout d’un nouveau document à indexer
- [ ] Définir le **workflow d’ajout** (upload → OCR → vectorisation → indexation)
- [ ] Gérer les **erreurs** (format non supporté, OCR échoué)
- [ ] Ajouter une **API ou CLI** pour automatiser l’ajout

---

## 3. Gestion de la base vectorielle
- [ ] Implémenter la **création et mise à jour** des embeddings
- [ ] Définir la **stratégie de persistance** (fichiers, DB)
- [ ] Ajouter des **mécanismes de recherche** (similarité, filtres)

---

## 4. Migration pip vers uv
- [ ] Migration des dépendances dans un fichier pyproject.toml
- [ ] Migration des scripts pour pouvoir utiliser uvx

---

## 5. Implémentation des tests
- [ ] Tests unitaires pour :
  - OCR
  - Conversion PDF → image
  - Indexation

---

## Idées futures
- [ ] Optimisation des performances (batch OCR, parallélisation)
