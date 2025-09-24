# TranscripteurSynthèse IA - Santé / Qualité (Version Mistral)

## Description

**TranscripteurSynthèse** est un script Python tout-en-un qui permet :

- 🎤 La transcription automatique de fichiers audio, vidéo ou de liens YouTube (via Whisper)
- 📝 La génération d'une synthèse IA avancée : choix du public cible, contexte, niveau de langue, style, extraction de bullet points, citations, glossaire, recommandations, encadré juridique, etc.
- 📄 L'export direct en **Word (.docx)**
- 🖥️ Une utilisation aussi bien en terminal (menu interactif) qu'en **API** (FastAPI)
- 🤖 Utilisation de **Mistral local** via Transformers (modèles : Mistral-7B-Instruct, etc.), avec optimisations mémoire et GPU

---

## Fonctionnalités principales

- Détection automatique du type de fichier (audio, vidéo, texte, YouTube)
- Transcription multilingue avec **Whisper**
- Menus interactifs pour personnaliser la synthèse (public, contexte, style…)
- Génération d'un prompt IA ultra-adapté à vos besoins
- Synthèse complète : résumé, structuration, analyse, recommandations, glossaire, citations, analyse critique, encadré juridique
- Export Word prêt à l'usage professionnel
- Double usage : terminal **et** API (mode batch ou intégré à un autre outil)
- **Nouveau** : IA locale avec Mistral via Transformers, sans dépendance externe

---

## Nouveautés Version 2.1 (Mistral)

### 🆕 Intégration Mistral Local
- **Remplacement d'Ollama** par Mistral directement via Transformers
- **Cache intelligent** : le modèle reste en mémoire entre les appels
- **Optimisations mémoire** : quantification 8-bit automatique si disponible
- **Support GPU/CPU** : détection automatique et utilisation optimale

### 🚀 Amélirations performances
- **Choix de modèles** : Mistral-7B-Instruct-v0.1/v0.2 selon vos besoins
- **Gestion mémoire** : optimisations pour fonctionner sur machines 8-16GB RAM
- **Templates de chat** : utilisation du format officiel Mistral
- **Contrôle de génération** : repetition penalty et paramètres fins

---

## Installation

### 1. **Cloner le dépôt**
   ```bash
   git clone <lien-du-repo>
   cd <nom-du-repo>
   ```

### 2. **Créer et activer un environnement virtuel** (recommandé)
   ```bash
   python3 -m venv mistral_env
   source mistral_env/bin/activate  # Linux/Mac
   # ou
   mistral_env\Scripts\activate     # Windows
   ```

### 3. **Installer les dépendances**
   ```bash
   pip install torch whisper yt-dlp requests python-docx fastapi uvicorn transformers accelerate
   ```

### 4. **Installation GPU (optionnel, recommandé)**
   ```bash
   # Pour NVIDIA GPU avec CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Pour optimisations mémoire (optionnel)
   pip install bitsandbytes
   ```

### 5. **Premier lancement (téléchargement du modèle)**
   ```bash
   python transcriptsynthese.py
   # Le modèle Mistral sera téléchargé automatiquement au premier usage (~13GB)
   ```

---

## Configuration requise

### Minimum
- **Python 3.8+**
- **8 GB RAM** (pour Mistral-7B)
- **15 GB espace disque** (modèles + cache)
- **CPU moderne** (Intel i5/AMD Ryzen 5 ou supérieur)

### Recommandé
- **16 GB RAM** ou plus
- **GPU NVIDIA** avec 6GB+ VRAM (RTX 3060, RTX 4060, etc.)
- **SSD** pour stockage des modèles

---

## Utilisation

### Mode Terminal (Interactif)

1. **Lancer le script**
   ```bash
   python transcriptsynthese.py
   ```

2. **Suivre les menus :**
   - Indiquer le fichier (audio, vidéo, texte) ou URL YouTube à traiter
   - Choisir le niveau de transcription Whisper (tiny, base, small…)
   - Choisir les paramètres de synthèse IA (public, contexte, style, structuration, etc.)
   - Sélectionner le modèle Mistral (7B-Instruct-v0.1 par défaut)
   - Lancer la génération de la synthèse IA
   - Récupérer le fichier Word (.docx) généré

### Mode API

1. **Lancer l'API**
   ```bash
   python transcriptsynthese.py --api
   ```

2. **Utiliser l'endpoint**
   ```bash
   curl -X POST "http://localhost:8000/synthese/" \
        -F "file=@mon_texte.txt" \
        -F "public=Professionnels de santé" \
        -F "modele=mistralai/Mistral-7B-Instruct-v0.1"
   ```

3. **Documentation interactive**
   - Accéder à `http://localhost:8000/docs` pour l'interface Swagger

---

## Modèles Mistral disponibles

### Par défaut inclus

| Modèle | Taille | RAM requise | Usage recommandé |
|--------|--------|-------------|------------------|
| `mistralai/Mistral-7B-Instruct-v0.1` | 7B | 8-12 GB | Production, équilibre qualité/performance |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | 8-12 GB | Version améliorée, instructions plus fines |
| `mistralai/Mistral-7B-v0.1` | 7B | 8-12 GB | Version de base, plus rapide |

### Optimisations automatiques
- **Quantification 8-bit** : réduction RAM à ~6GB si bitsandbytes installé
- **Device mapping** : répartition automatique GPU/CPU selon disponibilité
- **Cache persistant** : le modèle reste chargé entre les synthèses

---

## Exemples d'usage

### Transcription + Synthèse complète
```bash
# Fichier audio vers synthèse Word
python transcriptsynthese.py
> Chemin: meeting_qualite.mp3
> Modèle Whisper: 3 (small)
> Public: Professionnels de santé
> Contexte: Rapport interne/qualité
# → meeting_qualite_transcription.txt + synthese_finale.docx
```

### YouTube vers rapport
```bash
python transcriptsynthese.py
> Chemin: https://youtube.com/watch?v=xyz123
> Public: Décideurs/gestionnaires
> Style: Prêt à publier
# → Téléchargement auto + transcription + synthèse structurée
```

### API pour intégration
```python
import requests

with open("texte.txt", "rb") as f:
    response = requests.post(
        "http://localhost:8000/synthese/",
        files={"file": f},
        data={
            "public": "Grand public",
            "contexte": "Communication à destination des usagers",
            "modele": "mistralai/Mistral-7B-Instruct-v0.2"
        }
    )
# → Fichier Word en réponse
```

---

## Dépannage

### Erreur "CUDA out of memory"
```bash
# Utiliser la quantification 8-bit
pip install bitsandbytes
# Ou forcer CPU
export CUDA_VISIBLE_DEVICES=""
```

### Modèle trop lent
```bash
# Utiliser un modèle plus petit ou optimiser
# Dans le script, modifier load_mistral_model() :
model_kwargs["load_in_4bit"] = True  # Quantification aggressive
```

### Erreur de téléchargement du modèle
```bash
# Vérifier la connexion et l'espace disque
df -h  # Linux/Mac
dir   # Windows
# Cache Hugging Face : ~/.cache/huggingface/
```

### API ne démarre pas
```bash
# Vérifier les dépendances FastAPI
pip install fastapi uvicorn --upgrade
```

---

## Architecture technique

```
TranscripteurSynthèse/
├── transcriptsynthese.py    # Script principal avec Mistral
├── README.md               # Cette documentation
├── exigence.txt           # Dépendances Python
├── changes.log            # Historique des versions
└── cache/                 # Cache des modèles (auto-créé)
    ├── whisper/           # Modèles Whisper
    └── huggingface/       # Modèles Mistral
```

### Flux de traitement
1. **Input** → Détection format (audio/vidéo/texte/YouTube)
2. **Transcription** → Whisper (si nécessaire)
3. **Paramétrage** → Menus interactifs
4. **Prompt** → Construction adaptée au contexte
5. **IA** → Mistral local via Transformers
6. **Export** → Document Word structuré

---

## Performance et benchmarks

### Temps de traitement typiques (Mistral-7B, RTX 4060)
- **Transcription 10min audio** : ~2-3 minutes (Whisper small)
- **Synthèse 5000 mots** : ~30-60 secondes
- **Export Word** : instantané

### Consommation mémoire
- **CPU seul** : 12-16 GB RAM
- **GPU + quantification** : 6-8 GB RAM + 6 GB VRAM
- **Cache disque** : ~13 GB (modèle) + transcriptions

---

## Dépendances

### Essentielles
```
torch>=2.0.0
whisper>=20231117
yt-dlp>=2023.12.30
requests>=2.31.0
python-docx>=1.1.0
transformers>=4.36.0
accelerate>=0.25.0
```

### API (optionnel)
```
fastapi>=0.104.0
uvicorn>=0.24.0
```

### Optimisations (optionnel)
```
bitsandbytes>=0.41.0    # Quantification 8-bit
torch-audio             # Support audio étendu
```

---

## Sécurité et confidentialité

### ✅ Avantages Mistral local
- **Données privées** : aucun envoi vers le cloud
- **Offline** : fonctionne sans internet (après téléchargement)
- **Contrôle total** : paramètres, modèles, cache local
- **RGPD compliant** : traitement local des données sensibles

### 🔒 Bonnes pratiques
- **Environnement virtuel** : isolation des dépendances
- **Cache sécurisé** : vérifier les permissions du dossier ~/.cache/
- **Nettoyage** : supprimer les fichiers temporaires après usage

---

## Migration depuis Ollama

Si vous utilisez la version précédente avec Ollama :

1. **Sauvegarder** vos paramètres et scripts existants
2. **Installer** les nouvelles dépendances : `pip install transformers accelerate`
3. **Remplacer** le fichier `transcriptsynthese.py`
4. **Premier lancement** : le modèle Mistral se télécharge automatiquement
5. **Désinstaller Ollama** si souhaité (optionnel)

### Différences principales
- **Plus d'installation Ollama** requise
- **Interface identique** : menus et API inchangés
- **Performance améliorée** : optimisations Transformers natives
- **Cache unifié** : avec autres modèles Hugging Face

---

## Contribution et support

### Auteur
Projet développé par **Tanoux_Pixel**

### Licence
**CC-BY-NC** (Creative Commons Attribution - Pas d'utilisation commerciale)
Voir : https://fr.wikipedia.org/wiki/Licence_Creative_Commons

### Support
- **Issues** : Signaler les bugs via GitHub Issues
- **Améliorations** : Pull requests bienvenues
- **Documentation** : Contributions à la doc appréciées

### Roadmap
- [ ] Support des modèles Mistral plus grands (22B+)
- [ ] Interface web (Streamlit/Gradio)
- [ ] Export formats multiples (PDF, HTML, Markdown)
- [ ] Batch processing automatisé
- [ ] Intégration bases de données

---

## Exemples concrets secteur santé

### Cas d'usage typiques

#### 1. Transcription réunion CVS
```
Input: reunion_cvs_2024.mp3
Public: Institutionnels (ARS, HAS…)
Contexte: Dossier de certification (HAS…)
→ Synthèse avec références réglementaires automatiques
```

#### 2. Formation personnel soignant
```
Input: formation_hygiene.mp4
Public: Professionnels de santé
Style: Pédagogique (explications, définitions…)
→ Support avec glossaire et recommandations pratiques
```

#### 3. Communication patients
```
Input: https://youtube.com/watch?v=info-diabete
Public: Patients/résidents
Style: Très grand public (aucun jargon)
→ Fiche d'information accessible et structurée
```

---

*Version 2.1 - Mistral Local | Dernière mise à jour : 2025*
