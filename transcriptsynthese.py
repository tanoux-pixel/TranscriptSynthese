import os
import sys
import torch
import requests
import json
import time
from pathlib import Path
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

try:
    import argparse
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import FileResponse
    import uvicorn
except ImportError:
    pass  # Ces modules ne sont nécessaires que pour l'utilisation API

# ----- CONFIG -----
PUBLICS_CIBLES = [
    "Grand public (adolescents/adultes)",
    "Familles",
    "Professionnels de santé",
    "Institutionnels (ARS, HAS…)",
    "Décideurs/gestionnaires",
    "Patients/résidents",
    "Autre (saisie manuelle)"
]

CONTEXTES = [
    "Rapport interne/qualité",
    "Support à une réunion",
    "Communication à destination des usagers",
    "Restitution à la direction",
    "Dossier de certification (HAS…)",
    "Présentation à une commission (CVS, CME…)",
    "Article ou publication externe",
    "Autre (saisie manuelle)"
]

NIVEAUX_LANGUE = [
    "Très grand public (aucun jargon)",
    "Pédagogique (explications, définitions…)",
    "Professionnel courant (style rapport de service)",
    "Technique (lexique professionnel/savant)",
    "Prêt à publier (style éditorial, sans fautes, structuré)",
    "Journalistique (style article de presse, narratif)",
    "Scientifique (style article scientifique)",
    "Administratif (style institutionnel)",
    "Autre (saisie manuelle)"
]

# ----- PROMPT BUILDER -----

def build_prompt(texte, params):
    """
    Génère le prompt pour l'IA selon les paramètres choisis.
    """
    public = params["public"]
    contexte = params["contexte"]
    style = params["niveau_langue"]
    ton = params["ton"]
    objectifs = params["objectifs"]
    synthese_types = params["synthese_types"]  # liste de bools
    structuration = params["structuration"]  # liste de bools

    prompt = (
        f"Tu es un assistant IA expert en synthèse et vulgarisation en santé.\n"
        f"Ta mission : produire une synthèse fidèle, claire, structurée et utile du texte suivant, "
        f"en français, à destination du public suivant : **{public}**.\n"
        f"Contexte d'usage : **{contexte}**.\n"
        f"Niveau de langue : **{style}**.\n"
        f"Ton : **{ton}**.\n"
        f"Objectifs : {objectifs}.\n\n"
        f"Consignes :\n"
    )

    # Types de synthèse (résumé court, structuré, analytique, critique…)
    instructions = []
    if synthese_types[0]:
        instructions.append("- Commence par un résumé ultra-court (3 phrases max, va à l'essentiel).")
    if synthese_types[1]:
        instructions.append("- Ajoute un résumé structuré avec titres/sous-titres.")
    if synthese_types[2]:
        instructions.append("- Rédige une synthèse analytique : structure avec un plan, arguments, points forts/faibles.")
    if synthese_types[3]:
        instructions.append("- Intègre une analyse critique (mais reste neutre et factuel).")
    if synthese_types[4]:
        instructions.append("- Combine les différents styles ci-dessus si pertinent.")
    if synthese_types[5]:
        instructions.append("- Génère plusieurs versions de synthèse différentes si pertinent.")

    # Structuration spécifique
    if structuration[0]:
        instructions.append("- Structure la synthèse selon les chapitres/parties du texte d'origine si présents.")
    if structuration[1]:
        instructions.append("- Regroupe les idées par thématique.")
    if structuration[2]:
        instructions.append("- Mets en avant les citations ou éléments factuels précis entre guillemets.")

    # Extraction d’éléments spécifiques
    instructions += [
        "- En fin de synthèse, liste les points clés (bullet points).",
        "- Fournis une table des matières détaillée et numérotée.",
        "- Rédige un glossaire expliquant chaque notion importante rencontrée.",
        "- Sélectionne les citations marquantes entre guillemets.",
        "- Propose des recommandations pratico-pratiques issues de la transcription, ou des recommandations reconnues dans le monde de la santé en France.",
    ]
    # Bloc "Synthèse augmentée"
    instructions += [
        "- En fin de synthèse, ajoute un bloc d'analyse critique IA, neutre, sans prise de position.",
        "- Recherche toutes les références juridiques, lois, articles cités, et génère un résumé automatique de chaque source.",
        "- Dresse une liste complète des éléments juridiques détectés.",
        "- Ajoute un encadré 'Encadré juridique' distinct avec ces informations.",
    ]

    prompt += "\n".join(instructions)
    prompt += (
        "\n\nVoici le texte à synthétiser :\n"
        f"{texte[:12000]}"  # limite pour éviter l'overflow prompt
    )
    return prompt

# ----- MENU UTILISATEUR -----

def choix_menu(label, options):
    print(f"\n{label}")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    choix = input("Votre choix (numéro, ou saisie libre pour 'Autre') : ").strip()
    if choix.isdigit() and 1 <= int(choix) <= len(options):
        val = options[int(choix) - 1]
        if val.lower().startswith("autre"):
            return input("Veuillez préciser : ")
        else:
            return val
    else:
        return choix  # saisie libre

def choix_multi(label, options):
    print(f"\n{label} (séparez les numéros par une virgule pour multi-sélection)")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    choix = input("Votre choix : ").strip().replace(" ", "")
    idxs = [int(x) - 1 for x in choix.split(",") if x.isdigit() and 1 <= int(x) <= len(options)]
    if not idxs:
        return [False] * len(options)
    return [i in idxs for i in range(len(options))]

# ----- IA : APPEL OLLAMA (EXEMPLE, À ADAPTER POUR OPENAI/GEMINI SI BESOIN) -----

def generate_synthese_ollama(prompt, modele="mistral", temperature=0.5, max_tokens=2048):
    data = {
        "model": modele,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    r = requests.post("http://localhost:11434/api/generate", data=json.dumps(data))
    if r.status_code != 200:
        raise Exception(f"Ollama API renvoie le code {r.status_code}: {r.text}")
    synthese = r.json().get("response", "").strip()
    return synthese

# ----- EXPORT WORD -----

def export_word(texte_synthese, nom_sortie="synthese_finale.docx"):
    doc = Document()
    doc.add_heading("Synthèse IA de la transcription", 0)
    for bloc in texte_synthese.split("\n\n"):
        p = doc.add_paragraph(bloc.strip())
        p.style.font.size = Pt(11)
    doc.save(nom_sortie)
    print(f"✅ Fichier Word généré : {nom_sortie}")

# ----- MAIN (CLI) -----

def main():
    print("=== Synthèse IA de transcription (Santé/Qualité) ===")

    # 1. Lecture du texte à synthétiser
    chemin = input("\nChemin du fichier à synthétiser (txt) : ").strip()
    if not os.path.exists(chemin):
        print("Fichier introuvable.")
        sys.exit(1)
    with open(chemin, "r", encoding="utf-8") as f:
        texte = f.read()

    # 2. Paramétrage par menus
    params = {}

    # Public cible
    params["public"] = choix_menu("Choisissez le public cible :", PUBLICS_CIBLES)
    # Contexte d'usage
    params["contexte"] = choix_menu("Choisissez le contexte d'usage :", CONTEXTES)
    # Niveau de langue
    params["niveau_langue"] = choix_menu("Choisissez le niveau de langue souhaité :", NIVEAUX_LANGUE)
    # Ton
    params["ton"] = choix_menu("Choisissez le ton à adopter :", ["Autoritaire", "Engageant", "Neutre", "Pédagogique", "Autre"])
    # Objectifs
    params["objectifs"] = "Informer de façon neutre, faire un rappel de la législation si possible, lister des actions concrètes si possible (issues des cas cités dans la transcription)."
    # Types de synthèse
    params["synthese_types"] = choix_multi(
        "Types de synthèse à générer :\n1. Résumé court\n2. Résumé structuré\n3. Synthèse analytique\n4. Analyse critique\n5. Combinaison de styles\n6. Plusieurs synthèses différentes",
        ["Résumé court", "Résumé structuré", "Synthèse analytique", "Analyse critique", "Combinaison de styles", "Plusieurs synthèses différentes"])
    # Structuration
    params["structuration"] = choix_multi(
        "Structuration souhaitée :\n1. Selon les chapitres/parties du texte original\n2. Par thématique\n3. Mettre en avant les citations/faits précis",
        ["Selon les chapitres/parties", "Par thématique", "Citations/faits précis"])

    # Modèle IA et réglages
    modele = input("\nNom du modèle IA à utiliser (ex : mistral, phi3, llama3) : ").strip() or "mistral"
    try:
        temperature = float(input("Température (0=factuel, 1=créatif) [0.5] : ") or "0.5")
    except:
        temperature = 0.5
    try:
        max_tokens = int(input("Nombre max de tokens pour la synthèse [2048] : ") or "2048")
    except:
        max_tokens = 2048

    # 3. Génération du prompt
    prompt = build_prompt(texte, params)
    print("\n=== PROMPT GÉNÉRÉ POUR L'IA ===\n")
    print(prompt[:1000], "...\n")

    # 4. Synthèse IA
    print("\nEnvoi du prompt à l'IA locale (Ollama)...")
    synthese = generate_synthese_ollama(prompt, modele=modele, temperature=temperature, max_tokens=max_tokens)
    print("Synthèse générée :\n", synthese[:1000], "...\n")

    # 5. Export Word
    export_word(synthese)

# ----- MODE API -----

def run_api():
    app = FastAPI()

    @app.post("/synthese/")
    async def synthese_api(
        file: UploadFile = File(...),
        public: str = "Grand public",
        contexte: str = "Rapport interne/qualité",
        niveau_langue: str = "Très grand public",
        ton: str = "Neutre",
        synthese_types: str = "1,2,3",
        structuration: str = "1,2",
        modele: str = "mistral",
        temperature: float = 0.5,
        max_tokens: int = 2048
    ):
        texte = (await file.read()).decode("utf-8")
        params = {
            "public": public,
            "contexte": contexte,
            "niveau_langue": niveau_langue,
            "ton": ton,
            "objectifs": "Informer de façon neutre, faire un rappel de la législation si possible, lister des actions concrètes si possible (issues des cas cités dans la transcription).",
            "synthese_types": [str(i+1) in synthese_types.split(",") for i in range(6)],
            "structuration": [str(i+1) in structuration.split(",") for i in range(3)]
        }
        prompt = build_prompt(texte, params)
        synthese = generate_synthese_ollama(prompt, modele=modele, temperature=temperature, max_tokens=max_tokens)
        nom_sortie = "synthese_api.docx"
        export_word(synthese, nom_sortie)
        return FileResponse(nom_sortie, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename=nom_sortie)

    print("Lancement de l'API (FastAPI, /synthese/)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ----- CLI OR API -----

if __name__ == "__main__":
    if "--api" in sys.argv:
        run_api()
    else:
        main()

