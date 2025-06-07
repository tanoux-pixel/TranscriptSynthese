# Transcripteur Audio/Vidéo avec Synthèse GPT

Ce projet fournit une interface graphique simplifiant la transcription d'un fichier audio ou vidéo à l'aide du modèle *Whisper* et la génération d'un résumé via l'API GPT d'OpenAI.

## Dépendances

Les bibliothèques suivantes doivent être installées :

- `whisper`
- `openai`
- `torch`
- `tkinter` (inclus avec Python sur la plupart des systèmes)

## Installation

Installez les dépendances à l'aide de `pip` :

```bash
pip install -r requirements.txt
```

## Utilisation

Exécutez directement le script pour lancer l'interface graphique :

```bash
python transcriptsynthese.py
```

Une fenêtre permet de sélectionner un fichier audio ou vidéo, puis de générer la transcription et la synthèse à l'aide d'une clé API OpenAI.

