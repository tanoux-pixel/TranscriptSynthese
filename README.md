# TranscripteurSynthèse IA - Santé / Qualité

## Description

**TranscripteurSynthèse** est un script Python tout-en-un qui permet :

- 🎤 La transcription automatique de fichiers audio, vidéo ou de liens YouTube (via Whisper)
- 📝 La génération d'une synthèse IA avancée : choix du public cible, contexte, niveau de langue, style, extraction de bullet points, citations, glossaire, recommandations, encadré juridique, etc.
- 📄 L’export direct en **Word (.docx)**
- 🖥️ Une utilisation aussi bien en terminal (menu interactif) qu’en **API** (FastAPI)
- 🤖 Utilisation de l’IA locale **Ollama** par défaut (modèles : mistral, phi3, llama3…), facilement adaptable à OpenAI/Gemini si besoin

---

## Fonctionnalités principales

- Détection automatique du type de fichier (audio, vidéo, texte, YouTube)
- Transcription multilingue avec **Whisper**
- Menus interactifs pour personnaliser la synthèse (public, contexte, style…)
- Génération d’un prompt IA ultra-adapté à vos besoins
- Synthèse complète : résumé, structuration, analyse, recommandations, glossaire, citations, analyse critique, encadré juridique
- Export Word prêt à l’usage professionnel
- Double usage : terminal **et** API (mode batch ou intégré à un autre outil)

---

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone <lien-du-repo>
   cd <nom-du-repo>

2. **Créer et activer un environnement virtuel** (optionnel mais recommandé)
    ```bash
    python3 -m venv mon_env
   source mon_env/bin/activate

3. **Installer les dépendances**
   ```bash
   pip install torch whisper yt-dlp requests python-docx fastapi uvicorn

4. **Installer Ollama et les modèles IA souhaités**
   https://ollama.com/download
   ```bash
   ollama pull mistral
   
---

## Utilisation

1. **Lancer le script**
   ```bash
   python transcriptsynthese.py

2. **Suivre les menus :**
   
   Indiquer le fichier (audio, vidéo, texte) ou URL YouTube à traiter

   Choisir le niveau de transcription Whisper (tiny, base, small…)

   Choisir les paramètres de synthèse IA (public, contexte, style, structuration, etc.)

   Lancer la génération de la synthèse IA

   Récupérer le fichier Word (.docx) généré
   
---

## Utilisation (Mode API)

1. **Lancer le script**
   ```bash
   python transcriptsynthese.py --api

2. **Accéder au endpoint POST /synthese/**

   Envoie un fichier .txt et les paramètres (public, contexte…)

   Récupère un fichier Word généré
   
## Dépendances

   torch

   whisper

   yt-dlp

   requests

   python-docx

   fastapi, uvicorn (pour le mode API)

   ollama (pour l’IA locale)

   (optionnel : openai, google-generativeai)

## Auteur

Projet développé par Tanoux_Pixel.
Licence : CC-BY-NC (voir https://fr.wikipedia.org/wiki/Licence_Creative_Commons).
