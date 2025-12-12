cat > config.py << 'EOF'
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

TONS = ["Autoritaire", "Engageant", "Neutre", "Pédagogique", "Autre"]

MODELES_MISTRAL = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-v0.1"
]
EOF 
