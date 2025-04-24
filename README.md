# BitNet Demo

Ce projet est une démonstration de l'utilisation du modèle Microsoft BitNet, un modèle LLM basé sur la quantification de bit. BitNet utilise une approche innovante de quantification 1-bit pour réduire considérablement la taille des modèles tout en préservant leurs capacités.

## Prérequis

Ce projet nécessite :
- **Python 3.11** (recommandé ; TorchDynamo/BitNet ne supporte pas encore 3.12)
- `rust` (compilation de *tokenizers* ≥ 0.21)
- Un gestionnaire de paquets moderne, idéalement **uv**
- Accès Internet pour télécharger les modèles
- 8 Go RAM minimum (CPU ; +GPU = mieux)
- Xcode Command Line Tools + `brew install llvm libomp` (nécessaire si vous laissez Torch compiler en C++)

## Configuration

Ce projet utilise :
- Python 3.11
- PyTorch 2.3.1
- Transformers 4.52.0.dev0 (fork *bitnet*)
- tokenizers ≥ 0.21,<0.22
- Accelerate 1.6.0
- uv

> **Attention :** certaines bibliothèques (ex. *colpali‑engine*) exigent `transformers <4.47` et `numpy <2`.  
> Installez BitNet dans un environnement virtuel séparé pour éviter les conflits.

## Installation

### 1. Configuration de l'environnement virtuel

#### Avec uv (recommandé)
```bash
# Créer un environnement virtuel
uv venv -p python3.11 .venv-bitnet

# Activer l'environnement
source .venv-bitnet/bin/activate  # Sur Unix/macOS
# ou
.venv-bitnet\Scripts\activate      # Sur Windows
```

#### Avec venv (alternative)
```bash
# Créer un environnement virtuel
python3.11 -m venv .venv-bitnet

# Activer l'environnement
source .venv-bitnet/bin/activate      # Sur Unix/macOS
# ou
.venv-bitnet\Scripts\activate         # Sur Windows
```

### 2. Installation des dépendances

```bash
# Dépendances natives (une seule fois)
brew install rust

# Installation BitNet
uv pip install --no-binary :all: "regex!=2019.12.17" "tokenizers>=0.21,<0.22"
uv pip install torch==2.3.1 accelerate
uv pip install --no-binary :all: --no-cache-dir \
  git+https://github.com/shumingma/transformers.git@bitnet
```

### Fork Transformers à utiliser

Le support officiel de BitNet n’est pas encore fusionné dans `huggingface/transformers`.  
Utilisez la branche *bitnet* du dépôt suivant :

```bash
git+https://github.com/shumingma/transformers.git@bitnet
```

*Ne passez plus* `trust_remote_code=True` ; la classe `BitNetForCausalLM` est incluse dans ce fork.

## Utilisation des modèles

### Modèles supportés

Ce projet a été testé avec les modèles suivants :
- `gpt2` - Fonctionne parfaitement, idéal pour tester l'installation
- Modèles BitNet - Voir la note ci-dessous

### Note importante sur BitNet

Les modèles BitNet nécessitaient autrefois `trust_remote_code=True`.  
Depuis le fork dédié (cf. ci‑dessus), il suffit de :

```python
from transformers import BitNetForCausalLM, AutoTokenizer
model_id = "microsoft/bitnet-b1.58-2B-4T"
tok = AutoTokenizer.from_pretrained(model_id)
model = BitNetForCausalLM.from_pretrained(model_id).to("cpu")  # ou "mps"
```

Le script *fast-demo.py* intègre déjà ce fork et charge le modèle sans `trust_remote_code`.

## Exécution

### Exécution rapide avec *fast-demo.py*

```bash
# CPU (par défaut)
uv run python fast-demo.py -p "Cite trois faits surprenants sur Marie Curie"

# Essayer le GPU Apple (MPS) – encore expérimental
uv run python fast-demo.py -d mps -p "Cite trois faits surprenants sur Marie Curie"
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

- **`weight_scale is on the meta device…`**  
  Désactivez `device_map="auto"` et forcez le chargement sur CPU :  
  ```python
  model = BitNetForCausalLM.from_pretrained(model_id).to("cpu")
  ```

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
