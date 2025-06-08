import whisper
import os
import sys
import torch
import traceback
import requests
import json
import re
import time
from pathlib import Path
import subprocess
import platform
import psutil

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("\n[ERREUR] Le module 'yt-dlp' n'est pas installé.\nVeuillez l'installer avec : pip install yt-dlp")
    sys.exit(1)

try:
    from openai import OpenAI
    from openai import AuthenticationError
except ImportError:
    print("\n[ERREUR] Le module 'openai' n'est pas installé.\nVeuillez l'installer avec : pip install openai")
    sys.exit(1)

try:
    import google.generativeai as genai
    from google.api_core.exceptions import InvalidArgument, GoogleAPIError
except ImportError:
    print("\n[ERREUR] Le module 'google-generativeai' n'est pas installé.\nVeuillez l'installer avec : pip install google-generativeai")
    sys.exit(1)

class TranscripteurTerminal:
    whisper_models = {
        "1": ("Rapide (tiny)", "tiny", 1),
        "2": ("Standard (base)", "base", 1.5),
        "3": ("Précis (small)", "small", 2.5),
        "4": ("Ultra-précis (medium)", "medium", 5),
        "5": ("Ultra-précis+ (large)", "large", 10),
    }

    synthese_modes = {
        "1": {
            "nom": "Résumé ultra-court",
            "desc": "3 phrases maximum, va à l'essentiel.",
            "def_tokens": 256
        },
        "2": {
            "nom": "Résumé standard",
            "desc": "1 à 2 paragraphes, structure, titres.",
            "def_tokens": 1024
        },
        "3": {
            "nom": "Synthèse détaillée",
            "desc": "Plan, analyse, points forts/faibles.",
            "def_tokens": 2048
        },
        "4": {
            "nom": "Analyse argumentée",
            "desc": "Points de discussion, arguments, critiques.",
            "def_tokens": 4096
        }
    }

    def __init__(self):
        self.api_key_openai = None
        self.api_key_gemini = None
        self.selected_api_provider = None
        self.operation_mode = "transcribe_synthesize"
        self.modele_synth = None
        self.ollama_models = []
        self.niveau_whisper = None
        self.ram_totale = self.detect_ram()
        self.ollama_temperature = 0.6
        self.ollama_max_tokens = None
        self.synthese_mode = None

    def log(self, message):
        print(message)

    def get_user_input(self, prompt, default=None):
        if default:
            return input(f"{prompt} (Défaut: {default}) : ").strip() or default
        return input(f"{prompt} : ").strip()

    def detect_ram(self):
        ram = psutil.virtual_memory().total / (1024**3)
        return ram

    def select_operation_mode(self):
        print("\n--- Sélection du Mode d'Opération ---")
        print("1. Transcrire et synthétiser (audio/vidéo -> transcription + synthèse IA)")
        print("2. Synthétiser une transcription existante (fichier .txt -> synthèse IA)")
        while True:
            choice = self.get_user_input("Choisissez un mode (1 ou 2)", default="1")
            if choice == "1":
                self.operation_mode = "transcribe_synthesize"
                self.log("Mode sélectionné : Transcrire et synthétiser.")
                break
            elif choice == "2":
                self.operation_mode = "synthesize_only"
                self.log("Mode sélectionné : Synthétiser une transcription existante.")
                break
            else:
                print("[ATTENTION] Choix invalide. Veuillez saisir '1' ou '2'.")

    def choose_file(self):
        while True:
            file_prompt = (
                "Entrez le chemin complet du fichier audio/vidéo à traiter\n"
                "Ou bien, entrez une URL YouTube pour la télécharger automatiquement"
                if self.operation_mode == "transcribe_synthesize"
                else "Entrez le chemin complet du fichier de transcription (.txt) à synthétiser"
            )
            fichier = self.get_user_input(file_prompt)
            if not fichier:
                print("[ERREUR] Chemin/URL vide.")
                continue

            # Vérifie si c'est une URL YouTube
            if self.operation_mode == "transcribe_synthesize" and self.is_youtube_url(fichier):
                try:
                    downloaded_file = self.download_youtube_audio(fichier)
                    if downloaded_file:
                        self.log(f"📥 Audio extrait de YouTube : {downloaded_file}")
                        return downloaded_file
                    else:
                        print("[ERREUR] Échec du téléchargement YouTube.")
                        continue
                except Exception as e:
                    print(f"[ERREUR] Impossible de récupérer la vidéo YouTube : {e}")
                    continue

            # Sinon, fichier local normal
            if not os.path.exists(fichier):
                print("[ERREUR] Fichier introuvable.")
                continue

            valid_extensions = (
                [".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus", ".webm", ".3gp", ".ogv", ".mpg", ".mpeg"]
                if self.operation_mode == "transcribe_synthesize"
                else [".txt"]
            )
            if Path(fichier).suffix.lower() not in valid_extensions:
                confirm = self.get_user_input("Extension inattendue. Continuer ? (oui/non)", default="non")
                if confirm.lower() != "oui":
                    continue
            self.log(f"📁 Fichier sélectionné : {os.path.basename(fichier)}")
            return fichier

    def is_youtube_url(self, url):
        youtube_regex = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/'
        return re.match(youtube_regex, url) is not None

    def download_youtube_audio(self, url):
        self.log(f"\n🔗 [ÉTAPE 1] Téléchargement de la vidéo YouTube : {url}")
        print("L'audio va être extrait au format FLAC (qualité sans perte, idéal pour la transcription).")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'yt_audio_%(id)s.%(ext)s',
            'noplaylist': True,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'flac',
                'preferredquality': '0',
            }]
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_file = f"yt_audio_{info['id']}.flac"
            if os.path.exists(audio_file):
                return audio_file
            filename = ydl.prepare_filename(info)
            return filename if os.path.exists(filename) else None

    def select_whisper_model(self):
        print("\n--- [ÉTAPE 2] Choix du niveau de précision de la transcription Whisper ---")
        print(f"RAM système détectée : {self.ram_totale:.1f} Go")
        for key, (label, model, ram_req) in self.whisper_models.items():
            print(f"{key}. {label} (modèle : {model}, RAM conseillée : {ram_req} Go{'+' if model in ['medium','large'] else ''})")
        while True:
            choix = self.get_user_input("Sélectionnez la précision désirée (1 à 5)", default="2")
            if choix in self.whisper_models:
                label, model, ram_req = self.whisper_models[choix]
                if self.ram_totale < ram_req:
                    print(f"\n[⚠️ AVERTISSEMENT] Modèle '{label}' sélectionné, mais la RAM système détectée ({self.ram_totale:.1f} Go) est inférieure à la recommandation ({ram_req} Go).")
                    confirm = self.get_user_input("Continuer malgré tout ? (oui/non)", default="non")
                    if confirm.lower() != "oui":
                        continue
                self.niveau_whisper = model
                self.log(f"Vous avez choisi '{label}'.\nPlus le modèle est précis, plus la transcription sera fidèle mais lente.")
                return
            else:
                print("[ATTENTION] Choix invalide.")

    def select_api_provider(self):
        print("\n--- [ÉTAPE 3] Choix de la méthode de synthèse IA ---")
        print("1. OpenAI (ChatGPT, GPT-4o, etc.)")
        print("2. Google Gemini")
        print("3. IA locale (Ollama, Mistral, etc.)")
        while True:
            choice = self.get_user_input("Choisissez une option (1, 2 ou 3)", default="3")
            if choice == "1":
                self.selected_api_provider = "openai"
                self.log("Fournisseur sélectionné : OpenAI.")
                break
            elif choice == "2":
                self.selected_api_provider = "gemini"
                self.log("Fournisseur sélectionné : Google Gemini.")
                break
            elif choice == "3":
                self.selected_api_provider = "ollama"
                self.log("Mode sélectionné : IA locale (Ollama).")
                self.detect_ollama_models()
                if not self.ollama_models:
                    print("[ERREUR] Aucun modèle Ollama détecté. Lancez au moins 'ollama pull mistral' ou un autre modèle.")
                    sys.exit(1)
                self.select_ollama_model()
                break
            else:
                print("[ATTENTION] Choix invalide.")

    def detect_ollama_models(self):
        try:
            result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print("[ERREUR] Impossible d'exécuter 'ollama list'. Ollama est-il installé et lancé ?")
                self.ollama_models = []
                return
            models = []
            for line in result.stdout.splitlines():
                if not line or line.startswith("MODEL"):
                    continue
                model_name = line.split()[0]
                models.append(model_name)
            self.ollama_models = models
        except Exception as e:
            print(f"[ERREUR] Échec lors de la détection des modèles Ollama : {e}")
            self.ollama_models = []

    def select_ollama_model(self):
        print("\n--- [ÉTAPE 4] Modèles Ollama détectés ---")
        for i, model in enumerate(self.ollama_models):
            print(f"{i+1}. {model}")
        while True:
            choix = self.get_user_input("Sélectionnez le modèle Ollama (numéro)", default="1")
            try:
                idx = int(choix) - 1
                if 0 <= idx < len(self.ollama_models):
                    self.modele_synth = self.ollama_models[idx]
                    self.log(f"🤖 Modèle Ollama sélectionné : {self.modele_synth}")
                    return
            except ValueError:
                pass
            print("[ATTENTION] Numéro de modèle invalide.")

    def select_synthese_mode(self):
        print("\n--- [ÉTAPE 5] Choix du type de synthèse (Ollama) ---")
        for key, val in self.synthese_modes.items():
            print(f"{key}. {val['nom']}: {val['desc']}")
        while True:
            choix = self.get_user_input("Sélectionnez un type de synthèse (1 à 4)", default="2")
            if choix in self.synthese_modes:
                self.synthese_mode = choix
                self.log(f"Type de synthèse choisi : {self.synthese_modes[choix]['nom']}")
                return
            else:
                print("[ATTENTION] Choix invalide.")

    def reglages_ollama(self):
        # Température
        print("\n--- Réglage avancé : Température ---")
        print("La température contrôle la créativité de l’IA :")
        print("- 0 : réponse factuelle, fiable\n- 0.5 : standard (équilibré)\n- 1 : réponse plus créative, possibilité d’inventer")
        temp = self.get_user_input("Température souhaitée (0 à 1)", default=str(self.ollama_temperature))
        try:
            temp = float(temp)
            if temp < 0 or temp > 1:
                print("[ATTENTION] Valeur hors bornes, valeur par défaut utilisée.")
                temp = 0.6
        except:
            temp = 0.6
        self.ollama_temperature = temp

        # max_tokens
        print("\n--- Réglage avancé : max_tokens (longueur de la synthèse) ---")
        print("max_tokens définit la longueur maximale de la réponse générée par l’IA.")
        print("Exemples :\n- 256 : réponse courte (3 phrases)\n- 1024 : synthèse standard (1-2 paragraphes)\n- 2048 : synthèse détaillée\n- 4096 : analyse très longue/exhaustive")
        suggestion = self.synthese_modes[self.synthese_mode]['def_tokens']
        max_toks = self.get_user_input(f"Entrez le nombre de tokens souhaité (défaut recommandé : {suggestion})", default=str(suggestion))
        try:
            max_toks = int(max_toks)
            if max_toks < 128:
                print("[ATTENTION] Valeur très basse, la synthèse risque d’être incomplète.")
            elif max_toks > 8192:
                print("[ATTENTION] Valeur très haute, temps de traitement plus long et risque d’erreur !")
        except:
            max_toks = suggestion
        self.ollama_max_tokens = max_toks

    def run(self):
        self.log("🎬 --- DÉMARRAGE DU TRANSCRIPTEUR AUDIO/VIDÉO + SYNTHÈSE IA ---")
        device_info = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        self.log(f"Matériel détecté : {device_info}")
        try:
            self.select_operation_mode()
            self.fichier_selectionne = self.choose_file()
            if not self.fichier_selectionne:
                self.log("Aucun fichier sélectionné. Arrêt de l'application.")
                return
            if self.operation_mode == "transcribe_synthesize":
                self.select_whisper_model()
            if self.operation_mode in ["transcribe_synthesize", "synthesize_only"]:
                self.select_api_provider()
                if self.selected_api_provider == "openai":
                    if not self.get_openai_api_key():
                        self.log("Clé API OpenAI non fournie ou invalide. Arrêt de l'application.")
                        return
                elif self.selected_api_provider == "gemini":
                    if not self.get_gemini_api_key():
                        self.log("Clé API Gemini non fournie ou invalide. Arrêt de l'application.")
                        return
                if self.selected_api_provider == "ollama":
                    self.select_synthese_mode()
                    self.select_ollama_model()
                    self.reglages_ollama()
            self.log("\n🚀 --- Lancement du traitement ---")
            self.traitement_principal()
        except KeyboardInterrupt:
            self.log("\n🛑 Traitement interrompu par l'utilisateur.")
        except Exception as e:
            self.log(f"\n❌ Erreur critique inattendue : {str(e)}")
            self.log(f"Détails : {traceback.format_exc()}")
            print("\nUne erreur majeure est survenue. Veuillez consulter le journal ci-dessus.")
        finally:
            self.log("\n--- FIN DU PROGRAMME ---")

    def traitement_principal(self):
        fichier = self.fichier_selectionne
        base_name = Path(fichier).stem
        transcription_txt = f"{base_name}_transcription.txt"
        synthese_txt = f"{base_name}_synthese.txt"
        self.log(f"\n📋 Fichier à traiter : {os.path.basename(fichier)}")
        texte = ""
        modele_whisper = self.niveau_whisper or "base"
        try:
            if self.operation_mode == "transcribe_synthesize":
                self.log(f"\n🎵 [ÉTAPE 6] Transcription avec Whisper (modèle : {modele_whisper})")
                self.log("Explication : plus le modèle est précis, plus le traitement sera lent et la mémoire requise importante.")
                start = time.time()
                texte, modele_whisper, device_used = self.transcrire_fichier(fichier, transcription_txt, modele_whisper)
                elapsed = time.time() - start
                if not texte:
                    self.log("❌ La transcription n'a pas pu être réalisée ou a été interrompue.")
                    return
                self.log(f"✅ Transcription sauvegardée : {transcription_txt}")
                self.log(f"⏱️ Temps de transcription : {elapsed:.1f} sec.")
            else:
                self.log("\n📄 Lecture de la transcription existante...")
                try:
                    with open(fichier, "r", encoding="utf-8") as f:
                        texte = f.read()
                    self.log(f"✅ Transcription chargée : {os.path.basename(fichier)} ({len(texte)} caractères)")
                except Exception as e:
                    self.log(f"❌ Erreur lors de la lecture du fichier texte : {str(e)}")
                    print(f"Impossible de lire le fichier de transcription :\n{str(e)}")
                    return
            if self.operation_mode in ["transcribe_synthesize", "synthesize_only"]:
                self.log(f"\n🤖 [ÉTAPE 7] Synthèse IA")
                self.log("Explication : la synthèse se fait à partir de la transcription, selon le type choisi.")
                start = time.time()
                self.generer_synthese(texte, synthese_txt)
                elapsed = time.time() - start
                self.log(f"✅ Synthèse sauvegardée : {synthese_txt}")
                self.log(f"⏱️ Temps de synthèse : {elapsed:.1f} sec.")
            self.log("\n🎉 --- TRAITEMENT TERMINÉ ---")
            if self.operation_mode == "transcribe_synthesize":
                self.log(f"📝 Transcription : {transcription_txt}")
                self.log(f"🔧 Modèle Whisper : {modele_whisper}")
            else:
                self.log(f"📄 Transcription source : {os.path.basename(fichier)}")
            self.log(f"📋 Synthèse : {synthese_txt}")
        except Exception as e:
            self.log(f"\n❌ Erreur lors du traitement : {str(e)}")
            self.log(f"Détails : {traceback.format_exc()}")
            print("\nLe traitement a échoué. Veuillez consulter le journal ci-dessus pour plus de détails.")

    def transcrire_fichier(self, fichier, nom_sortie, modele_whisper):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_name = "GPU (CUDA)" if device == "cuda" else "CPU"
            self.log(f"💻 Chargement du modèle Whisper sur {device_name}...")
            model = whisper.load_model(modele_whisper, device=device)
            self.log("🎵 Début de la transcription...")
            result = model.transcribe(fichier, verbose=True)
            with open(nom_sortie, "w", encoding="utf-8") as f:
                f.write(result["text"])
            self.log(f"📄 Transcription de {len(result['text'])} caractères générée.")
            return result["text"], modele_whisper, device_name
        except Exception as e:
            self.log(f"❌ Erreur de transcription : {str(e)}")
            return None, None, None

    def generer_synthese(self, texte, nom_sortie):
        try:
            prompt = self.prompt_synthese(texte)
            if self.selected_api_provider == "ollama":
                self.generer_synthese_ollama(prompt, nom_sortie)
                return
            if self.selected_api_provider == "openai":
                client = OpenAI(api_key=self.api_key_openai)
                self.log("🌐 Envoi de la requête à OpenAI pour la synthèse...")
                response = client.chat.completions.create(
                    model=self.modele_synth,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.ollama_max_tokens,
                    temperature=self.ollama_temperature,
                )
                synthese = response.choices[0].message.content.strip()
            elif self.selected_api_provider == "gemini":
                genai.configure(api_key=self.api_key_gemini)
                model = genai.GenerativeModel(self.modele_synth)
                self.log("🌐 Envoi de la requête à Google Gemini pour la synthèse...")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self.ollama_max_tokens,
                        temperature=self.ollama_temperature
                    )
                )
                synthese = response.text.strip()
            else:
                self.log("[ERREUR] Fournisseur d'API de synthèse non reconnu.")
                return
            with open(nom_sortie, "w", encoding="utf-8") as f:
                f.write(synthese)
            self.log(f"📋 Synthèse de {len(synthese)} caractères générée.")
        except Exception as e:
            self.log(f"❌ Erreur de synthèse : {str(e)}")
            raise

    def prompt_synthese(self, texte):
        mode = self.synthese_modes[self.synthese_mode]['nom']
        if self.synthese_mode == "1":
            consigne = "Fais un résumé ultra-court du texte ci-dessous (3 phrases maximum, va à l'essentiel) :\n"
        elif self.synthese_mode == "2":
            consigne = "Fais une synthèse standard du texte suivant, en 1 ou 2 paragraphes structurés avec des titres :\n"
        elif self.synthese_mode == "3":
            consigne = (
                "Fais une synthèse détaillée du texte ci-dessous : organise avec un plan, développe l'analyse, liste points forts/faibles si pertinent.\n"
            )
        elif self.synthese_mode == "4":
            consigne = (
                "Analyse argumentée du texte suivant : liste les points de discussion, développe les arguments et critiques possibles.\n"
            )
        else:
            consigne = "Fais une synthèse claire et structurée du texte suivant :\n"
        return consigne + "\nTEXTE À SYNTHÉTISER :\n" + texte[:12000]  # coupe à 12000 caractères max par précaution

    def generer_synthese_ollama(self, prompt, nom_sortie):
        data = {
            "model": self.modele_synth,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.ollama_temperature,
                "num_predict": self.ollama_max_tokens
            }
        }
        try:
            self.log(f"🖥️ Requête à Ollama ({self.modele_synth}) en cours...")
            r = requests.post("http://localhost:11434/api/generate", data=json.dumps(data))
            if r.status_code != 200:
                raise Exception(f"Ollama API renvoie le code {r.status_code}: {r.text}")
            synthese = r.json().get("response", "").strip()
            with open(nom_sortie, "w", encoding="utf-8") as f:
                f.write(synthese)
            self.log(f"📋 Synthèse locale générée ({len(synthese)} caractères).")
        except Exception as e:
            self.log(f"❌ Erreur lors de la synthèse locale Ollama : {e}")

    def get_openai_api_key(self):
        print("\n--- Configuration OpenAI ---")
        print("La synthèse nécessite une clé API OpenAI.\nVous pouvez l'obtenir ici : https://platform.openai.com/api-keys")
        while True:
            key = self.get_user_input("Veuillez saisir votre clé API OpenAI")
            if not key:
                print("[ATTENTION] La clé API ne peut pas être vide.")
                continue
            self.log("🌐 Test de la clé API OpenAI...")
            try:
                client = OpenAI(api_key=key)
                client.models.list()
                self.api_key_openai = key
                self.log("✅ Clé API validée avec succès.")
                return True
            except AuthenticationError:
                print("[ERREUR] Clé API OpenAI invalide. Veuillez vérifier votre clé.")
            except Exception as e:
                print(f"[ERREUR] Une erreur est survenue lors du test de la clé API OpenAI : {e}")
            return False

    def get_gemini_api_key(self):
        print("\n--- Configuration Google Gemini ---")
        print("La synthèse nécessite une clé API Gemini.\nVous pouvez l'obtenir ici : https://makersuite.google.com/app/apikey")
        while True:
            key = self.get_user_input("Veuillez saisir votre clé API Gemini")
            if not key:
                print("[ATTENTION] La clé API ne peut pas être vide.")
                continue
            self.log("🌐 Test de la clé API Gemini...")
            try:
                genai.configure(api_key=key)
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        self.log(f"✅ Clé API Gemini validée avec succès.")
                        self.api_key_gemini = key
                        return True
                print("[ERREUR] Aucune méthode de génération de contenu supportée trouvée avec cette clé API Gemini. Vérifiez votre clé.")
                return False
            except InvalidArgument as e:
                print(f"[ERREUR] Clé API Gemini invalide ou problème de configuration : {e}")
            except GoogleAPIError as e:
                print(f"[ERREUR] Erreur Google API avec la clé Gemini : {e}")
            except Exception as e:
                print(f"[ERREUR] Une erreur inattendue est survenue lors du test de la clé API Gemini : {e}")
            return False

if __name__ == "__main__":
    app = TranscripteurTerminal()
    app.run()

