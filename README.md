TranscripteurSynthèse

Description

TranscripteurSynthèse est un script Python tout-en-un permettant de :

    Transcrire automatiquement des fichiers audio/vidéo ou des contenus YouTube en texte (grâce à Whisper),

    Générer des synthèses IA de ces transcriptions : résumé court, synthèse structurée, analyse détaillée ou argumentée,

    Choisir entre plusieurs fournisseurs d’IA : OpenAI (GPT), Google Gemini, ou IA locale via Ollama (ex : Mistral).

Ce projet vise à offrir un outil simple, personnalisable et local-friendly pour tout besoin de transcription et de synthèse automatique.
Fonctionnalités principales

    📥 Téléchargement YouTube (audio extrait automatiquement)

    🎤 Transcription multilingue (plusieurs niveaux de précision avec Whisper)

    📝 Synthèse IA (résumé, synthèse standard, analyse détaillée ou critique)

    🤖 Choix du moteur d’IA : OpenAI (GPT), Google Gemini, ou IA locale (Ollama/Mistral)

    🔒 Gestion sécurisée des clés API (via variable d’environnement ou prompt)

    🛠️ Logs détaillés (console ou fichier)

    ⚡ Utilisation CPU ou GPU (auto-détection pour Whisper)

    🗃️ Résultats sauvegardés automatiquement (transcription.txt + synthese.txt)

Installation

    Cloner le dépôt

git clone <lien-du-repo>
cd <nom-du-repo>

Créer et activer un environnement virtuel (optionnel mais recommandé)

python3 -m venv mon_env
source mon_env/bin/activate

Installer les dépendances

    pip install -r exigence.txt

    Installer Whisper et/ou Ollama si nécessaire ([voir docs de chaque projet])

Utilisation

    Lancer le script

    python transcriptsynthese.py

    Suivre les instructions en console

        Choix du mode : “Transcrire + Synthétiser” ou “Synthétiser une transcription existante”

        Saisie du fichier audio/vidéo, texte, ou URL YouTube

        Choix du niveau de précision Whisper (Tiny à Large)

        Choix du moteur IA (OpenAI, Gemini, Ollama)

        Configuration fine (température, longueur de synthèse, modèle Ollama…)

    Résultats

        Les fichiers .txt (transcription et synthèse) sont générés à côté de votre fichier source

Configuration et sécurité

    Clés API :
    À renseigner via un prompt sécurisé au lancement, ou via des variables d’environnement pour automatiser (OPENAI_API_KEY, GEMINI_API_KEY…)

    Fichier .env :
    Vous pouvez stocker vos clés API dans un fichier .env non publié sur GitHub.

    Logs :
    Les logs sont affichés en console et peuvent être sauvegardés en fichier (décommenter dans le script si besoin).

Dépendances principales

    whisper

    openai

    google-generativeai

    yt-dlp

    ollama (optionnel, pour IA locale)

    python-dotenv (optionnel)

    torch

    psutil

    requests

    (et autres dans exigence.txt)
