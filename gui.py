cat > gui.py << 'EOF'
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import threading
from pathlib import Path

from config import PUBLICS_CIBLES, CONTEXTES
from core import transcrire_fichier_audio, build_prompt, generate_synthese_mistral, export_word

def run_gui():
    fichier_selectionne = None
    
    def selectionner_fichier():
        nonlocal fichier_selectionne
        filetypes = [
            ("Fichiers audio/vidéo", "*.mp3 *.wav *.mp4 *.mkv *.avi *.mov *.flac *.aac *.ogg *.m4a"),
            ("Fichiers texte", "*.txt"),
            ("Tous les fichiers", "*.*")
        ]
        fichier = filedialog.askopenfilename(title="Sélectionner un fichier", filetypes=filetypes)
        if fichier:
            fichier_selectionne = fichier
            label_fichier.config(text=f"📄 {Path(fichier).name}")
            btn_lancer.config(state="normal")
            log("Fichier sélectionné : " + fichier)
    
    def log(message):
        zone_texte.config(state="normal")
        zone_texte.insert(tk.END, message + "\n")
        zone_texte.see(tk.END)
        zone_texte.config(state="disabled")
    
    def lancer_traitement():
        if not fichier_selectionne:
            messagebox.showwarning("Attention", "Veuillez d'abord sélectionner un fichier.")
            return
        
        btn_lancer.config(state="disabled", text="⏳ Traitement en cours...")
        
        def traitement():
            try:
                log("\n" + "="*50)
                log("Démarrage du traitement...")
                
                ext = Path(fichier_selectionne).suffix.lower()
                
                if ext in [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".mp4", ".mkv", ".avi", ".mov"]:
                    log("[Whisper] Transcription en cours...")
                    texte, _ = transcrire_fichier_audio(fichier_selectionne, modele="base")
                    log(f"[Whisper] Transcription terminée ({len(texte)} caractères)")
                else:
                    with open(fichier_selectionne, "r", encoding="utf-8") as f:
                        texte = f.read()
                    log(f"[Texte] Fichier chargé ({len(texte)} caractères)")
                
                params = {
                    "public": combo_public.get(),
                    "contexte": combo_contexte.get(),
                    "niveau_langue": "Professionnel courant (style rapport de service)",
                    "ton": "Neutre",
                    "objectifs": "Informer de façon neutre, lister des actions concrètes si possible.",
                    "synthese_types": [True, True, False, False, False, False],
                    "structuration": [False, True, False]
                }
                
                log("[IA] Construction du prompt...")
                prompt = build_prompt(texte, params)
                
                log("[IA] Génération de la synthèse avec Mistral...")
                synthese = generate_synthese_mistral(prompt, temperature=0.5, max_tokens=2048)
                log(f"[IA] Synthèse générée ({len(synthese)} caractères)")
                
                nom_sortie = Path(fichier_selectionne).stem + "_synthese.docx"
                export_word(synthese, nom_sortie)
                log(f"✅ Fichier Word généré : {nom_sortie}")
                
                log("="*50)
                log("TRAITEMENT TERMINÉ")
                
                messagebox.showinfo("Succès", f"Synthèse générée !\n\nFichier : {nom_sortie}")
                
            except Exception as e:
                log(f"❌ ERREUR : {e}")
                messagebox.showerror("Erreur", str(e))
            finally:
                btn_lancer.config(state="normal", text="🚀 Lancer")
        
        thread = threading.Thread(target=traitement)
        thread.start()
    
    fenetre = tk.Tk()
    fenetre.title("TranscriptSynthèse - IA Qualité Santé")
    fenetre.geometry("600x500")
    fenetre.resizable(True, True)
    
    frame_main = ttk.Frame(fenetre, padding="10")
    frame_main.pack(fill=tk.BOTH, expand=True)
    
    frame_fichier = ttk.LabelFrame(frame_main, text="1. Fichier source", padding="10")
    frame_fichier.pack(fill=tk.X, pady=(0, 10))
    
    btn_parcourir = ttk.Button(frame_fichier, text="📁 Parcourir...", command=selectionner_fichier)
    btn_parcourir.pack(side=tk.LEFT)
    
    label_fichier = ttk.Label(frame_fichier, text="Aucun fichier sélectionné")
    label_fichier.pack(side=tk.LEFT, padx=10)
    
    frame_params = ttk.LabelFrame(frame_main, text="2. Paramètres", padding="10")
    frame_params.pack(fill=tk.X, pady=(0, 10))
    
    ttk.Label(frame_params, text="Public cible :").grid(row=0, column=0, sticky=tk.W, pady=2)
    combo_public = ttk.Combobox(frame_params, values=PUBLICS_CIBLES, state="readonly", width=40)
    combo_public.set(PUBLICS_CIBLES[2])
    combo_public.grid(row=0, column=1, pady=2, padx=5)
    
    ttk.Label(frame_params, text="Contexte :").grid(row=1, column=0, sticky=tk.W, pady=2)
    combo_contexte = ttk.Combobox(frame_params, values=CONTEXTES, state="readonly", width=40)
    combo_contexte.set(CONTEXTES[0])
    combo_contexte.grid(row=1, column=1, pady=2, padx=5)
    
    btn_lancer = ttk.Button(frame_main, text="🚀 Lancer", command=lancer_traitement, state="disabled")
    btn_lancer.pack(pady=10)
    
    frame_log = ttk.LabelFrame(frame_main, text="3. Journal", padding="10")
    frame_log.pack(fill=tk.BOTH, expand=True)
    
    zone_texte = scrolledtext.ScrolledText(frame_log, height=12, state="disabled", wrap=tk.WORD)
    zone_texte.pack(fill=tk.BOTH, expand=True)
    
    zone_texte.config(state="normal")
    zone_texte.insert(tk.END, "Bienvenue dans TranscriptSynthèse !\n")
    zone_texte.insert(tk.END, "1. Sélectionnez un fichier audio/vidéo ou texte\n")
    zone_texte.insert(tk.END, "2. Ajustez les paramètres si besoin\n")
    zone_texte.insert(tk.END, "3. Cliquez sur Lancer\n")
    zone_texte.config(state="disabled")
    
    fenetre.mainloop()
EOF 
