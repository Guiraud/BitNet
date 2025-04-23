# BitNet Demo

Ce projet est une démonstration de l'utilisation du modèle Microsoft BitNet, un modèle LLM basé sur la quantification de bit. BitNet utilise une approche innovante de quantification 1-bit pour réduire considérablement la taille des modèles tout en préservant leurs capacités.

## Prérequis

Ce projet nécessite :
- Python 3.12 ou supérieur
- Un gestionnaire de paquets Python moderne comme `uv` (recommandé) ou `pip`
- Accès à Internet pour télécharger les modèles
- Minimum 8GB de RAM recommandé

## Configuration

Ce projet utilise :
- Python 3.12
- PyTorch >= 2.0.0
- Transformers >= 4.34.0
- Accelerate >= 0.23.0
- uv (gestionnaire de paquets Python)

## Installation

### 1. Configuration de l'environnement virtuel

#### Avec uv (recommandé)
```bash
# Créer un environnement virtuel
uv venv -p python3.12 .venv-uv

# Activer l'environnement
source .venv-uv/bin/activate  # Sur Unix/macOS
# ou
.venv-uv\Scripts\activate      # Sur Windows
```

#### Avec venv (alternative)
```bash
# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate      # Sur Unix/macOS
# ou
.venv\Scripts\activate         # Sur Windows
```

### 2. Installation des dépendances

#### Avec uv (recommandé)
```bash
uv pip install -r requirements.txt
```

#### Avec pip (alternative)
```bash
pip install -r requirements.txt
```

## Utilisation des modèles

### Modèles supportés

Ce projet a été testé avec les modèles suivants :
- `gpt2` - Fonctionne parfaitement, idéal pour tester l'installation
- Modèles BitNet - Voir la note ci-dessous

### Note importante sur BitNet

Les modèles BitNet nécessitent d'utiliser le paramètre `trust_remote_code=True` lors du chargement du modèle et du tokenizer. Ceci est nécessaire car BitNet utilise du code personnalisé pour implémenter la quantification 1-bit.

Si vous rencontrez des erreurs avec BitNet, vérifiez ces points :
- Assurez-vous d'utiliser `trust_remote_code=True`
- Vérifiez que vous utilisez le bon identifiant de modèle
- La version actuelle des modèles BitNet peut nécessiter une version spécifique de Transformers

## Exécution

Pour exécuter la démonstration :

```bash
python bitnet_demo.py
```

Ce script téléchargera le modèle configuré et générera une réponse à une question simple.

### Modification du modèle utilisé

Vous pouvez modifier la variable `model_id` dans le fichier `bitnet_demo.py` pour utiliser un autre modèle. Par exemple :

```python
# Pour utiliser GPT-2 (recommandé pour les tests)
model_id = "gpt2"

# Pour utiliser BitNet (nécessite trust_remote_code=True)
model_id = "microsoft/bitnet-b1.58-2B-4T-bf16"
```

## Paramètres de génération

Vous pouvez ajuster les paramètres de génération dans le script pour obtenir des réponses de meilleure qualité :

```python
# Paramètres ajustables
outputs = model.generate(
    **input_ids,
    max_new_tokens=50,        # Longueur de la réponse
    temperature=0.7,          # Contrôle la créativité (0.0 = déterministe, 1.0 = créatif)
    top_p=0.9,                # Sampling des tokens les plus probables
    repetition_penalty=1.2    # Évite les répétitions
)
```

## Dépannage

### Problème : "Cannot locate configuration_bitnet.py"
Solution : C'est une erreur connue avec certains modèles BitNet. Essayez d'utiliser un modèle alternatif comme "gpt2" pour vérifier que votre installation fonctionne correctement.

### Problème : "CUDA out of memory"
Solution : Réduisez la taille du batch, utilisez un modèle plus petit, ou ajoutez le paramètre `device_map="auto"` lors du chargement du modèle.

### Problème : Réponses répétitives
Solution : Ajustez les paramètres de génération, notamment `repetition_penalty` et `temperature`.

## Installation de llama.cpp pour macOS Silicon

Ce projet peut nécessiter l'utilisation de llama.cpp pour certaines fonctionnalités, notamment avec les modèles BitNet. Voici comment installer et configurer llama.cpp sur macOS avec puce Apple Silicon (M1/M2/M3) :

### 1. Cloner le dépôt llama.cpp
```bash
cd ~/Documents
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

### 2. Compiler avec CMake pour Apple Silicon

> ⚠️ **Note importante** : La méthode de compilation avec Makefile est désormais dépréciée. Utilisez CMake comme recommandé ci-dessous.

```bash
# Installer CMake si ce n'est pas déjà fait
brew install cmake

# Créer et accéder au répertoire de build
mkdir build
cd build

# Configurer avec support Metal pour Apple Silicon
cmake .. -DLLAMA_METAL=ON

# Compiler le projet
cmake --build . --config Release
```

### 3. Configurer BitNet pour utiliser llama.cpp

Pour intégrer llama.cpp avec BitNet, vous devez créer la structure de répertoires correcte :

```bash
# Créer le répertoire 3rdparty s'il n'existe pas
mkdir -p /Users/mguiraud/Documents/github/BitNet/3rdparty/

# Créer un lien symbolique vers llama.cpp
ln -sf ~/Documents/llama.cpp /Users/mguiraud/Documents/github/BitNet/3rdparty/
```

### 4. Problèmes connus et solutions

#### Problème de casse dans les chemins
Si vous rencontrez des erreurs liées à des chemins introuvables, vérifiez la casse des répertoires. macOS peut être sensible à la casse dans certains contextes. Par exemple, si votre dépôt est dans `/Documents/gitHub/` mais que le script recherche dans `/Documents/github/`, créez un lien symbolique :

```bash
mkdir -p /Users/mguiraud/Documents/github
ln -sf /Users/mguiraud/Documents/gitHub/BitNet /Users/mguiraud/Documents/github/
```

#### Fichiers requirements manquants
Si vous rencontrez une erreur concernant des fichiers requirements manquants dans llama.cpp :

```bash
# Vérifiez que le dossier requirements de llama.cpp contient tous les fichiers nécessaires
ls -la ~/Documents/llama.cpp/requirements/
```

#### Environnement conda
Si vous utilisez conda et rencontrez l'erreur "CondaError: Run 'conda init' before 'conda activate'" :

```bash
# Initialisez conda pour votre shell (zsh par défaut sur macOS récent)
conda init zsh

# Redémarrez votre terminal ou exécutez
source ~/.zshrc
```

### 5. Variables d'environnement utiles

Pour faciliter l'utilisation de llama.cpp, ajoutez ces lignes à votre `~/.zshrc` ou au script d'activation de votre environnement conda :

```bash
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/Documents/llama.cpp/build/bin
export PATH=$PATH:~/Documents/llama.cpp/build/bin
```

## Ressources

- [Documentation officielle Microsoft BitNet](https://github.com/microsoft/unilm/tree/master/bitnet)
- [Documentation Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Documentation uv](https://github.com/astral-sh/uv)
- [Documentation llama.cpp](https://github.com/ggerganov/llama.cpp)

## Licence

Veuillez consulter les licences des modèles que vous utilisez. Les modèles BitNet sont soumis aux conditions d'utilisation de Microsoft.
