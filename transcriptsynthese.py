import whisper
import os
import sys
import torch
<<<<<<< HEAD
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
=======
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font
import webbrowser
from pathlib import Path
import traceback

# Vérification des dépendances
try:
    from openai import OpenAI
except ImportError:
    root_temp = tk.Tk()
    root_temp.withdraw()
    result = messagebox.askyesno(
        "Module manquant",
        "Le module 'openai' n'est pas installé.\n\n"
        "Voulez-vous ouvrir la page d'installation ?\n"
        "Commande : pip install openai",
        icon="warning"
    )
    if result:
        webbrowser.open("https://pypi.org/project/openai/")
    root_temp.destroy()
    sys.exit(1)

class TranscripteurModerne:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_variables()
        self.create_interface()
        self.center_window()

    def setup_window(self):
        """Configuration de la fenêtre principale"""
        self.root.title("🎬 Transcripteur Audio/Vidéo + Synthèse IA")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)

        # Style moderne
        self.root.configure(bg='#f0f0f0')

        # Icône de la fenêtre (si disponible)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass

    def setup_variables(self):
        """Initialisation des variables"""
        self.api_key = None
        self.fichier_selectionne = None
        self.is_processing = False
        self.operation_mode = tk.StringVar(value="transcribe_synthesize") # Nouvelle variable pour le mode d'opération

    def create_interface(self):
        """Création de l'interface utilisateur"""
        # Frame principal avec padding
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=30, pady=20)
        main_frame.pack(fill="both", expand=True)

        # Titre principal
        title_font = font.Font(family="Arial", size=16, weight="bold")
        title_label = tk.Label(
            main_frame,
            text="🎬 Transcripteur Audio/Vidéo + Synthèse IA",
            font=title_font,
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))

        # Section choix d'opération
        self.create_operation_mode_section(main_frame)

        # Section sélection de fichier
        self.create_file_section(main_frame)

        # Section configuration
        self.create_config_section(main_frame)

        # Section contrôles
        self.create_controls_section(main_frame)

        # Section progression
        self.create_progress_section(main_frame)

        # Section logs
        self.create_logs_section(main_frame)

        # Section informations
        self.create_info_section(main_frame)

    def create_operation_mode_section(self, parent):
        """Section de sélection du mode d'opération"""
        mode_frame = tk.LabelFrame(
            parent,
            text=" ⚙️ Mode d'opération ",
            bg='#f0f0f0',
            fg='#34495e',
            font=("Arial", 10, "bold"),
            padx=15,
            pady=10
        )
        mode_frame.pack(fill="x", pady=(0, 15))

        # Radio buttons for operation mode
        rb_transcribe = tk.Radiobutton(
            mode_frame,
            text="Transcrire et synthétiser (Audio/Vidéo)",
            variable=self.operation_mode,
            value="transcribe_synthesize",
            command=self.on_operation_mode_change,
            bg='#f0f0f0',
            fg='#2c3e50',
            font=("Arial", 9)
        )
        rb_transcribe.pack(anchor="w", pady=(0, 5))

        rb_synthesize = tk.Radiobutton(
            mode_frame,
            text="Synthétiser une transcription existante (.txt)",
            variable=self.operation_mode,
            value="synthesize_only",
            command=self.on_operation_mode_change,
            bg='#f0f0f0',
            fg='#2c3e50',
            font=("Arial", 9)
        )
        rb_synthesize.pack(anchor="w")

    def on_operation_mode_change(self):
        """Met à jour l'interface en fonction du mode d'opération choisi"""
        self.fichier_selectionne = None # Réinitialiser le fichier sélectionné
        self.lbl_fichier.config(text="📂 Aucun fichier sélectionné", fg='#7f8c8d')
        self.btn_lancer.config(state="disabled")

        if self.operation_mode.get() == "transcribe_synthesize":
            self.btn_fichier.config(
                text="🎵 Sélectionner un fichier audio/vidéo"
            )
        else: # synthesize_only
            self.btn_fichier.config(
                text="📄 Sélectionner un fichier texte (.txt)"
            )
        self.log(f"Mode d'opération changé : {self.operation_mode.get().replace('_', ' ').title()}")

    def create_file_section(self, parent):
        """Section de sélection de fichier"""
        file_frame = tk.LabelFrame(
            parent,
            text=" 📁 Sélection du fichier ",
            bg='#f0f0f0',
            fg='#34495e',
            font=("Arial", 10, "bold"),
            padx=15,
            pady=10
        )
        file_frame.pack(fill="x", pady=(0, 15))

        self.btn_fichier = tk.Button(
            file_frame,
            text="🎵 Sélectionner un fichier audio/vidéo",
            command=self.choisir_fichier,
            bg='#3498db',
            fg='white',
            font=("Arial", 10, "bold"),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.btn_fichier.pack(fill="x", pady=(0, 10))

        self.lbl_fichier = tk.Label(
            file_frame,
            text="📂 Aucun fichier sélectionné",
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 9),
            wraplength=600
        )
        self.lbl_fichier.pack(fill="x")

    def create_config_section(self, parent):
        """Section de configuration"""
        config_frame = tk.LabelFrame(
            parent,
            text=" ⚙️ Configuration GPT ", # Renommé pour plus de clarté
            bg='#f0f0f0',
            fg='#34495e',
            font=("Arial", 10, "bold"),
            padx=15,
            pady=10
        )
        config_frame.pack(fill="x", pady=(0, 15))

        # Modèle GPT
        gpt_frame = tk.Frame(config_frame, bg='#f0f0f0')
        gpt_frame.pack(fill="x")

        tk.Label(
            gpt_frame,
            text="🤖 Modèle GPT :",
            bg='#f0f0f0',
            fg='#2c3e50',
            font=("Arial", 9, "bold")
        ).pack(anchor="w", pady=(0, 5))

        self.modele_gpt = tk.StringVar(value="gpt-4o-mini")
        self.combo_gpt = ttk.Combobox(
            gpt_frame,
            textvariable=self.modele_gpt,
            values=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            state="readonly",
            font=("Arial", 9)
        )
        self.combo_gpt.pack(fill="x")

        # Informations sur les modèles
        info_modeles = tk.Label(
            gpt_frame,
            text="💡 gpt-4o-mini : Rapide et économique | gpt-4o : Plus précis mais plus coûteux",
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 8)
        )
        info_modeles.pack(anchor="w", pady=(5, 0))

    def create_controls_section(self, parent):
        """Section des contrôles"""
        controls_frame = tk.Frame(parent, bg='#f0f0f0')
        controls_frame.pack(fill="x", pady=(0, 15))

        self.btn_lancer = tk.Button(
            controls_frame,
            text="🚀 Lancer le traitement",
            command=self.lancer_traitement,
            state="disabled",
            bg='#27ae60',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=30,
            pady=12,
            cursor='hand2'
        )
        self.btn_lancer.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_stop = tk.Button(
            controls_frame,
            text="⏹️ Arrêter",
            command=self.arreter_traitement,
            state="disabled",
            bg='#e74c3c',
            fg='white',
            font=("Arial", 10, "bold"),
            relief='flat',
            padx=20,
            pady=12,
            cursor='hand2'
        )
        self.btn_stop.pack(side="right")

    def create_progress_section(self, parent):
        """Section de progression"""
        progress_frame = tk.LabelFrame(
            parent,
            text=" 📊 Progression ",
            bg='#f0f0f0',
            fg='#34495e',
            font=("Arial", 10, "bold"),
            padx=15,
            pady=10
        )
        progress_frame.pack(fill="x", pady=(0, 15))

        self.progress_label = tk.Label(
            progress_frame,
            text="En attente...",
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 9)
        )
        self.progress_label.pack(anchor="w", pady=(0, 5))

        self.progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.progress.pack(fill="x", pady=(0, 5))
        self.progress["value"] = 0

        self.progress_percent = tk.Label(
            progress_frame,
            text="0%",
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 8)
        )
        self.progress_percent.pack(anchor="e")

    def create_logs_section(self, parent):
        """Section des logs"""
        logs_frame = tk.LabelFrame(
            parent,
            text=" 📝 Journal d'activité ",
            bg='#f0f0f0',
            fg='#34495e',
            font=("Arial", 10, "bold"),
            padx=15,
            pady=10
        )
        logs_frame.pack(fill="both", expand=True, pady=(0, 15))

        # Frame pour le texte et scrollbar
        text_frame = tk.Frame(logs_frame, bg='#f0f0f0')
        text_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            text_frame,
            height=8,
            state="disabled",
            bg='#2c3e50',
            fg='#ecf0f1',
            font=("Consolas", 9),
            wrap="word"
        )

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bouton pour effacer les logs
        clear_btn = tk.Button(
            logs_frame,
            text="🗑️ Effacer",
            command=self.clear_logs,
            bg='#95a5a6',
            fg='white',
            font=("Arial", 8),
            relief='flat',
            padx=10,
            pady=2
        )
        clear_btn.pack(anchor="e", pady=(5, 0))

    def create_info_section(self, parent):
        """Section d'informations"""
        info_frame = tk.Frame(parent, bg='#f0f0f0')
        info_frame.pack(fill="x")

        # Détection GPU/CPU
        device_info = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        device_label = tk.Label(
            info_frame,
            text=f"Matériel détecté : {device_info}",
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 8)
        )
        device_label.pack(anchor="w")

        # Version info
        version_label = tk.Label(
            info_frame,
            text="Version 2.1 | Formats supportés : MP4, AVI, MOV, MP3, WAV, FLAC, TXT", # Ajout de TXT
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 8)
        )
        version_label.pack(anchor="w")

    def center_window(self):
        """Centrer la fenêtre à l'écran"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def choisir_fichier(self):
        """Sélection du fichier audio/vidéo ou texte"""
        current_mode = self.operation_mode.get()

        if current_mode == "transcribe_synthesize":
            types_fichiers = [
                ("Fichiers Audio/Vidéo", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.opus *.webm *.3gp *.ogv *.mpg *.mpeg"),
                ("Vidéos", "*.mp4 *.mkv *.avi *.mov *.webm *.3gp *.ogv *.mpg *.mpeg"),
                ("Audio", "*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.opus"),
                ("Tous les fichiers", "*.*")
            ]
            title = "Sélectionner un fichier audio ou vidéo"
        else: # synthesize_only
            types_fichiers = [
                ("Fichiers Texte", "*.txt"),
                ("Tous les fichiers", "*.*")
            ]
            title = "Sélectionner un fichier de transcription (.txt)"

        fichier = filedialog.askopenfilename(
            title=title,
            filetypes=types_fichiers
        )

        if fichier:
            self.fichier_selectionne = fichier
            nom_fichier = os.path.basename(fichier)
            taille = os.path.getsize(fichier) / (1024*1024)  # MB

            self.lbl_fichier.config(
                text=f"✅ {nom_fichier} ({taille:.1f} MB)",
                fg='#27ae60'
            )
            self.btn_lancer.config(state="normal")
            self.log(f"📁 Fichier sélectionné : {nom_fichier}")
        else:
            self.lbl_fichier.config(
                text="📂 Aucun fichier sélectionné",
                fg='#7f8c8d'
            )
            self.btn_lancer.config(state="disabled")

    def update_progress(self, value, text=""):
        """Mise à jour de la barre de progression"""
        self.progress["value"] = value
        self.progress_percent.config(text=f"{int(value)}%")
        if text:
            self.progress_label.config(text=text)
        self.root.update_idletasks()

    def log(self, message):
        """Ajout d'un message dans les logs"""
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def clear_logs(self):
        """Effacer les logs"""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.config(state="disabled")

    def demander_api_key(self):
        """Demande de la clé API OpenAI"""
        dialog = ApiKeyDialog(self.root)
        self.root.wait_window(dialog.dialog) # This line makes the main window wait for the dialog to close.
        return dialog.api_key

    def lancer_traitement(self):
        """Lancement du traitement principal"""
        if not self.api_key:
            self.api_key = self.demander_api_key()
            if not self.api_key:
                messagebox.showwarning(
                    "Clé API requise",
                    "La clé API OpenAI est obligatoire pour la synthèse."
                )
                return

        if not self.fichier_selectionne:
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier.")
            return

        # Désactiver les contrôles
        self.is_processing = True
        self.btn_lancer.config(state="disabled")
        self.btn_fichier.config(state="disabled")
        self.combo_gpt.config(state="disabled")
        self.btn_stop.config(state="normal")

        # Désactiver les radio buttons pendant le traitement
        # Access the radio buttons by iterating through the children of the mode_frame
        # This access is slightly brittle, a more robust way is to store references to them.
        try:
            for widget in self.root.winfo_children()[0].winfo_children()[2].winfo_children():
                if isinstance(widget, tk.Radiobutton):
                    widget.config(state="disabled")
        except IndexError:
            # Fallback if structure changes, log an error or handle gracefully
            self.log("Avertissement: Impossible de désactiver les boutons radio. La structure de l'interface a peut-être changé.")


        # Réinitialiser la progression
        self.update_progress(0, "Initialisation...")
        self.clear_logs()

        # Lancer le traitement dans un thread séparé
        self.thread_traitement = threading.Thread(
            target=self.traitement_principal,
            daemon=True
        )
        self.thread_traitement.start()

    def arreter_traitement(self):
        """Arrêt du traitement"""
        self.is_processing = False
        self.log("🛑 Arrêt demandé par l'utilisateur...")
        self.reactiver_controles()

    def reactiver_controles(self):
        """Réactivation des contrôles"""
        self.btn_lancer.config(state="normal" if self.fichier_selectionne else "disabled")
        self.btn_fichier.config(state="normal")
        self.combo_gpt.config(state="readonly")
        self.btn_stop.config(state="disabled")
        self.update_progress(0, "En attente...")

        # Réactiver les radio buttons
        try:
            for widget in self.root.winfo_children()[0].winfo_children()[2].winfo_children():
                if isinstance(widget, tk.Radiobutton):
                    widget.config(state="normal")
        except IndexError:
            self.log("Avertissement: Impossible de réactiver les boutons radio. La structure de l'interface a peut-être changé.")

    def traitement_principal(self):
        """Traitement principal de transcription et synthèse"""
        try:
            fichier = self.fichier_selectionne
            modele_gpt = self.modele_gpt.get()
            operation_mode = self.operation_mode.get()

            # Génération des noms de fichiers de sortie
            base_name = Path(fichier).stem
            transcription_txt = f"{base_name}_transcription.txt"
            synthese_txt = f"{base_name}_synthese.txt"

            self.log("🎬 === DÉBUT DU TRAITEMENT ===")
            self.log(f"📁 Fichier : {os.path.basename(fichier)}")
            self.log(f"🤖 Modèle GPT : {modele_gpt}")
            self.log(f"⚙️ Mode d'opération : {operation_mode.replace('_', ' ').title()}")

            texte = ""
            modele_whisper = "N/A"
            device_used = "N/A"

            if operation_mode == "transcribe_synthesize":
                # Étape 1: Transcription
                if not self.is_processing:
                    self.log("Traitement interrompu.")
                    return

                self.log("🎵 Phase 1 : Transcription avec Whisper...")
                self.update_progress(10, "Chargement du modèle Whisper...")

                texte, modele_whisper, device_used = self.transcrire_fichier(
                    fichier, transcription_txt
                )

                if not texte or not self.is_processing:
                    if self.is_processing: # Only log error if not explicitly stopped
                        self.log("❌ Erreur lors de la transcription ou traitement interrompu.")
                    return

                self.update_progress(60, "Transcription terminée")
                self.log(f"✅ Transcription sauvegardée : {transcription_txt}")
            else: # synthesize_only
                self.log("📄 Phase 1 : Lecture de la transcription existante...")
                self.update_progress(10, "Lecture du fichier texte...")
                try:
                    with open(fichier, "r", encoding="utf-8") as f:
                        texte = f.read()
                    self.update_progress(30, "Fichier texte lu.")
                    self.log(f"✅ Transcription chargée : {os.path.basename(fichier)} ({len(texte)} caractères)")
                except Exception as e:
                    self.log(f"❌ Erreur lors de la lecture du fichier texte : {str(e)}")
                    messagebox.showerror("Erreur", f"Impossible de lire le fichier de transcription :\n{str(e)}")
                    return

            # Étape 2: Synthèse
            if not self.is_processing:
                self.log("Traitement interrompu.")
                return

            self.log("🤖 Phase 2 : Synthèse avec GPT...")
            self.update_progress(70 if operation_mode == "transcribe_synthesize" else 50, "Génération de la synthèse...") # Ajustement de la progression

            self.generer_synthese(texte, synthese_txt, modele_gpt)

            if not self.is_processing:
                self.log("Traitement interrompu.")
                return

            self.update_progress(100, "Traitement terminé !")

            # Résumé final
            self.log("🎉 === TRAITEMENT TERMINÉ ===")
            if operation_mode == "transcribe_synthesize":
>>>>>>> b2896277f61a1a9a1e6bb43dcaf28080487fa447
                self.log(f"📝 Transcription : {transcription_txt}")
                self.log(f"🔧 Modèle Whisper : {modele_whisper}")
            else:
                self.log(f"📄 Transcription source : {os.path.basename(fichier)}")
            self.log(f"📋 Synthèse : {synthese_txt}")
<<<<<<< HEAD
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
=======
            self.log(f"💻 Matériel : {device_used}")

            messagebox.showinfo(
                "Succès !",
                f"Traitement terminé avec succès !\n\n"
                f"📝 Transcription : {transcription_txt if operation_mode == 'transcribe_synthesize' else os.path.basename(fichier)}\n"
                f"📋 Synthèse : {synthese_txt}"
            )

        except Exception as e:
            self.log(f"❌ Erreur critique : {str(e)}")
            self.log(f"Détails : {traceback.format_exc()}")
            messagebox.showerror("Erreur", f"Une erreur est survenue :\n{str(e)}")

        finally:
            self.is_processing = False
            self.reactiver_controles()

    def transcrire_fichier(self, fichier, nom_sortie):
        """Transcription du fichier avec Whisper"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_name = "GPU (CUDA)" if device == "cuda" else "CPU"

            self.log(f"💻 Chargement du modèle Whisper sur {device_name}...")
            model = whisper.load_model("base", device=device)

            if not self.is_processing:
                return None, None, None

            self.update_progress(20, "Analyse du fichier audio...")
            self.log("🎵 Début de la transcription...")

            # Transcription
            result = model.transcribe(fichier)

            if not self.is_processing:
                return None, None, None

            # Sauvegarde
            with open(nom_sortie, "w", encoding="utf-8") as f:
                f.write(result["text"])

            self.log(f"📄 Transcription de {len(result['text'])} caractères générée")

            return result["text"], "base", device_name

        except Exception as e:
            self.log(f"❌ Erreur de transcription : {str(e)}")
            raise

    def generer_synthese(self, texte, nom_sortie, modele_gpt):
        """Génération de la synthèse avec GPT"""
        try:
            client = OpenAI(api_key=self.api_key)

            # Limitation du texte pour éviter les dépassements de tokens
            # Ajustement de la limite pour la synthèse : un texte plus long peut être synthétisé,
            # mais il faut rester vigilant sur les tokens et le coût.
            # 8000 caractères est une bonne base, mais on peut aller jusqu'à 16k pour gpt-4o-mini
            # en fonction du modèle et du prompt.
            texte_limite = texte[:12000] if len(texte) > 12000 else texte

            prompt = (
                "Fais une synthèse complète, claire et structurée du texte suivant. "
                "Organise-la avec des titres, des points clés et des conclusions. "
                "La synthèse doit être professionnelle et facilement lisible.\n\n"
                "TEXTE À SYNTHÉTISER :\n"
                + texte_limite
            )

            self.log("🌐 Envoi de la requête à OpenAI...")
            self.update_progress(80 if self.operation_mode.get() == "transcribe_synthesize" else 70, "Génération par l'IA...") # Ajustement progression

            response = client.chat.completions.create(
                model=modele_gpt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, # La synthèse aura au maximum 2000 tokens
                temperature=0.6,
            )

            if not self.is_processing:
                return

            synthese = response.choices[0].message.content.strip()

            # Sauvegarde
            with open(nom_sortie, "w", encoding="utf-8") as f:
                f.write(synthese)

            self.update_progress(95, "Sauvegarde de la synthèse...")
            self.log(f"📋 Synthèse de {len(synthese)} caractères générée")

>>>>>>> b2896277f61a1a9a1e6bb43dcaf28080487fa447
        except Exception as e:
            self.log(f"❌ Erreur de synthèse : {str(e)}")
            raise

<<<<<<< HEAD
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

=======
class ApiKeyDialog:
    """Dialog pour saisir la clé API OpenAI"""
    def __init__(self, parent):
        self.api_key = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Clé API OpenAI")
        self.dialog.geometry("500x250")
        self.dialog.resizable(False, False)
        self.dialog.configure(bg='#f0f0f0')

        # Centrer le dialog
        self.dialog.transient(parent) # Rend le dialogue au-dessus de la fenêtre parente
        self.dialog.grab_set()      # Rend le dialogue modal (bloque l'interaction avec la fenêtre parente)

        # Force la mise à jour des tâches pour obtenir les dimensions du dialogue avant le positionnement
        self.dialog.update_idletasks()

        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()

        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()

        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)

        self.dialog.geometry(f'+{x}+{y}')

        # Contenu
        frame = tk.Frame(self.dialog, bg='#f0f0f0', padx=30, pady=20)
        frame.pack(fill="both", expand=True)

        # Titre
        title_label = tk.Label(
            frame,
            text="🔑 Clé API OpenAI requise",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 15))

        # Instructions
        info_text = (
            "Pour utiliser la synthèse GPT, vous devez fournir votre clé API OpenAI.\n"
            "Vous pouvez l'obtenir sur : https://platform.openai.com/api-keys"
        )
        info_label = tk.Label(
            frame,
            text=info_text,
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 9),
            wraplength=400,
            justify="left"
        )
        info_label.pack(pady=(0, 15))

        # Champ de saisie
        tk.Label(
            frame,
            text="Clé API :",
            bg='#f0f0f0',
            fg='#2c3e50',
            font=("Arial", 9, "bold")
        ).pack(anchor="w")

        self.entry = tk.Entry(
            frame,
            show="*", # Masque la saisie pour la sécurité
            font=("Arial", 10),
            width=50
        )
        self.entry.pack(fill="x", pady=(5, 15))

        # Assurez-vous que le focus est mis sur le champ de saisie
        self.entry.focus_set()
        # Et tentez de lier l'événement de clic droit pour le collage.
        # Sur la plupart des systèmes, le clic droit est déjà géré par l'OS,
        # mais on peut forcer un peu pour s'assurer.
        self.entry.bind("<Button-3>", self.show_context_menu) # Clic droit (Button-3)

        # Boutons
        btn_frame = tk.Frame(frame, bg='#f0f0f0')
        btn_frame.pack(fill="x")

        tk.Button(
            btn_frame,
            text="Annuler",
            command=self.annuler,
            bg='#95a5a6',
            fg='white',
            font=("Arial", 9),
            relief='flat',
            padx=20,
            pady=5
        ).pack(side="right", padx=(10, 0))

        tk.Button(
            btn_frame,
            text="Valider",
            command=self.valider,
            bg='#27ae60',
            fg='white',
            font=("Arial", 9, "bold"),
            relief='flat',
            padx=20,
            pady=5
        ).pack(side="right")

        # Raccourcis clavier pour valider et annuler
        self.dialog.bind('<Return>', lambda e: self.valider())
        self.dialog.bind('<Escape>', lambda e: self.annuler())

    def show_context_menu(self, event):
        """Affiche un menu contextuel personnalisé avec l'option Coller."""
        menu = tk.Menu(self.dialog, tearoff=0)
        # Ajoute l'option "Coller" au menu
        menu.add_command(label="Coller", command=self.paste_from_clipboard)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def paste_from_clipboard(self):
        """Colle le contenu du presse-papiers dans le champ d'entrée."""
        try:
            text = self.dialog.clipboard_get()
            self.entry.insert(tk.INSERT, text)
        except tk.TclError:
            messagebox.showwarning("Coller", "Le presse-papiers est vide ou inaccessible.")

    def valider(self):
        """Validation de la clé API"""
        key = self.entry.get().strip()
        if key:
            self.api_key = key
            self.dialog.destroy()
        else:
            messagebox.showwarning("Clé requise", "Veuillez saisir votre clé API.")

    def annuler(self):
        """Annulation"""
        self.dialog.destroy()

def main():
    """Fonction principale"""
    try:
        # Configuration de l'interface
        root = tk.Tk()

        # Gestion de la fermeture
        def on_closing():
            if messagebox.askokcancel("Quitter", "Voulez-vous vraiment fermer l'application ?"):
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        # Création de l'application
        app = TranscripteurModerne(root)

        # Message de bienvenue
        app.log("🎬 Transcripteur Audio/Vidéo + Synthèse IA")
        app.log("✅ Application prête à l'emploi")
        app.log("💡 Sélectionnez un mode d'opération et un fichier pour commencer")

        # Lancement de l'interface
        root.mainloop()

    except Exception as e:
        messagebox.showerror(
            "Erreur critique",
            f"Impossible de démarrer l'application :\n{str(e)}"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
>>>>>>> b2896277f61a1a9a1e6bb43dcaf28080487fa447
