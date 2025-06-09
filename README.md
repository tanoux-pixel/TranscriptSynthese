# TranscripteurSynthèse

## Description

**TranscripteurSynthèse** est un script Python tout-en-un permettant de :
- Transcrire automatiquement des fichiers audio/vidéo ou des contenus YouTube en texte (grâce à Whisper),
- Générer des synthèses IA de ces transcriptions : résumé court, synthèse structurée, analyse détaillée ou argumentée,
- Choisir entre plusieurs fournisseurs d’IA : OpenAI (GPT), Google Gemini, ou IA locale via Ollama (ex : Mistral).

Ce projet vise à offrir un outil simple, personnalisable et local-friendly pour tout besoin de transcription et de synthèse automatique.

---

## Fonctionnalités principales

- 📥 **Téléchargement YouTube** (audio extrait automatiquement)
- 🎤 **Transcription multilingue** (plusieurs niveaux de précision avec Whisper)
- 📝 **Synthèse IA** (résumé, synthèse standard, analyse détaillée ou critique)
- 🤖 **Choix du moteur d’IA** : OpenAI (GPT), Google Gemini, ou IA locale (Ollama/Mistral)
- 🔒 **Gestion sécurisée des clés API** (via variable d’environnement ou prompt)
- 🛠️ **Logs détaillés** (console ou fichier)
- ⚡ **Utilisation CPU ou GPU** (auto-détection pour Whisper)
- 🗃️ **Résultats sauvegardés automatiquement** (transcription.txt + synthese.txt)

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
pip install -r exigence.txt

4. **Installer Whisper et/ou Ollama** si nécessaire ([voir docs de chaque projet])
   
---

## Utilisation

1. **Lancer le script**
   ```bash
   python transcriptsynthese.py

2. **Suivre les instructions en console**
   Choix du mode : “Transcrire + Synthétiser” ou “Synthétiser une transcription existante”
   Saisie du fichier audio/vidéo, texte, ou URL YouTube
   Choix du niveau de précision Whisper (Tiny à Large)
   Choix du moteur IA (OpenAI, Gemini, Ollama)
   Configuration fine (température, longueur de synthèse, modèle Ollama…)

3. **Résultats**
   Les fichiers .txt (transcription et synthèse) sont générés à côté de votre fichier source
