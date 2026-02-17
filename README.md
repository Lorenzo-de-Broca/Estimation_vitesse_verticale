# Estimation de la Vitesse Verticale de l'Air

## ��� Description du Projet

Ce projet vise à **estimer la vitesse verticale de l'air** en utilisant des données satellites de **températures de brillance** (brightness temperature). L'approche combine des techniques de régression machine learning pour établir une relation entre les variations spatiales et temporelles des températures de brillance et la vitesse verticale mesurée.

### Objectifs
- Extraire et filtrer les données de mesures satellites (netCDF)
- Créer des filtres pour éviter les zones de convection
- Entraîner différents modèles de régression (Linéaire, Random Forest, Réseaux de Neurones)
- Évaluer et visualiser les performances des modèles
- Générer des prédictions de vitesse verticale en 2D/3D

---

## ��� Guide d'Installation et d'Utilisation

### Prérequis - Installation des Librairies

Installez les dépendances requises :

```bash
pip install numpy matplotlib scikit-learn netCDF4 pyyaml tensorflow pandas seaborn scipy
```

**Détail des librairies :**
- `numpy` : Opérations numériques sur les tableaux
- `matplotlib` : Visualisation des données et résultats
- `scikit-learn` : Modèles de régression (linéaire, Random Forest)
- `netCDF4` : Lecture des fichiers de données en format netCDF
- `pyyaml` : Gestion des fichiers de configuration YAML
- `tensorflow` : Réseaux de neurones (optionnel, pour `try_tensorflow.py`)
- `pandas` : Manipulation de données (optionnel)
- `seaborn` : Visualisation avancée (optionnel)
- `scipy` : Outils scientifiques (interpolation, optimisation)

### Configuration Initiale

1. **Créer le fichier `inputs/paths.yaml`** pour spécifier le chemin du fichier de données :

```yaml
# filepath: inputs/paths.yaml
data_file: "/chemin/vers/votre/fichier/MesoNH-ice3_CADDIWAF7_1km_projectHB.nc"
```

2. **Configurer `inputs/inputs.yaml`** pour personnaliser l'entraînement (voir section Configuration ci-dessous)

3. **Créer le dossier `figures/`** pour stocker les résultats :

```bash
mkdir figures
```

### Lancer une Analyse

#### Option 1 : Régression Linéaire ou Random Forest

```bash
# Avec configuration par défaut
python main.py

# Avec configuration personnalisée
python main.py --input inputs/inputs.yaml --paths inputs/paths.yaml
```

#### Option 2 : Réseau de Neurones (TensorFlow)

Pour le réseau de neuronne la fonction **neural_network** se trouve dans le fichier **main.py** et il faut la lancer manuellement dans une console python:

`neural_network()`

---

## ⚙️ Configuration via les Fichiers d'Input

### `inputs/inputs.yaml`

Ce fichier contrôle les paramètres d'entraînement :

```yaml
### Linear regressions and Random Forest parameters
training_ratio : 0.6

compute_delta_T : False            # if true, train on delta T else on absolute T

compute_random_forest : False      # if true, use Random Forest Regressor else use Multilinear Regression
n_trees : 10                       # number of trees for Random Forest Regressor

compute_PCA : False                # if random forest is True, it will not compute PCA no matter what
PCA_components : 5                 # if true, compute PCA with this number of components

### Neural Network parameters
train_ratio : 0.6                  # ratio of data points to use for Neural Network training
pop_size : 10000                   # number of data points to use for Neural Network training

learning_rate : 0.0001             # learning rate for Neural Network training
monitor_metric : 'val_loss'        # metric to monitor for early stopping
nn_epochs : 5000                   # number of epochs for Neural Network training
nn_early_stopping : True           # if true, use early stopping during Neural Network training
nn_early_stopping_patience : 70    # patience for early stopping
```

### `inputs/paths.yaml`

Définit le chemin d'accès aux données :

```yaml
data_file: "/chemin/complet/vers/MesoNH-ice3_CADDIWAF7_1km_projectHB.nc"
```

**Important** : Utilisez des chemins absolus pour éviter les erreurs.

---

## ��� Structure du Code

```
Estimation_vitesse_verticale/
├── main.py                          # Point d'entrée principal
├── README.md                        # Documentation
├── inputs/
│   ├── inputs.yaml                 # Configuration des paramètres
│   ├── paths.yaml                  # Chemins d'accès aux données
│   └── paths.example.yaml          # Exemple de configuration
├── figures/                        # Dossier de sortie des graphiques
├── SRC/                            # Code source
│   ├── __pycache__/                # Cache Python
│   ├── extract_data.py             # Extraction et prétraitement
│   ├── filtre_convection.py        # Création du filtre de convection
│   ├── regression.py               # Modèles de régression
│   ├── plots.py                    # Fonctions de visualisation
│   ├── utils.py                    # Fonctions utilitaires
```

---

## ��� Détail des Modules

### ��� `main.py`
**Rôle** : Orchestrateur principal du pipeline d'analyse

**Fonctions principales** :
- `load_input(file_path)` : Charge les fichiers YAML de configuration
- `main(input_file, paths_file)` : Exécute le pipeline complet
  1. Charge la configuration
  2. Extrait les données
  3. Crée les filtres de convection
  4. Entraîne le modèle
  5. Évalue les performances
  6. Génère les visualisations

**Flux d'exécution** :
```
Chargement config → Extraction données → Filtrage convection 
→ Division train/test → Entraînement modèle → Tests → Visualisation
```

---

### ��� `SRC/extract_data.py`
**Rôle** : Extraction et prétraitement des données

**Fonctions principales** :

| Fonction | Description |
|----------|-------------|
| `extract_data()` | Lit le fichier netCDF et retourne un dictionnaire avec temps, longitude, latitude, températures de brillance, vitesse verticale |
| `create_train_test_matrix(train_ratio)` | Crée des masques binaires pour diviser les données en ensembles d'entraînement/test |
| `create_reg_array1(freq, frame, filter)` | Crée des arrays de ΔT filtés uniquement par le filtre de convection |
| `create_reg_array2(freq, frame, filter, train_matrix)` | Crée des arrays de ΔT avec filtrage convection + train/test |
| `create_reg_array3(freq, frame, filter, train_matrix)` | Crée des arrays de T brut (non différencié) |
| `create_combined_regression_array(frame, filter, train_matrix)` | Combine tous les canaux de fréquence en une seule matrice |
| `create_combined_regression_array_delta_t(...)` | Version ΔT de la fonction précédente |
| `create_PCA(combined_x, combined_y, pca_components)` | Applique l'ACP pour réduire la dimensionnalité |

**Variables clés** :
- `default_train = np.ones((500, 500))` : Masque d'entraînement par défaut (tout inclus)

---

### ��� `SRC/filtre_convection.py`
**Rôle** : Créer un filtre pour identifier et exclure les zones de convection

**Fonctions principales** :

| Fonction | Description |
|----------|-------------|
| `create_convection_filter()` | Génère un filtre 3D basé sur la différence entre TB(183±1) et TB(183±7). Les zones où TB(183±1) > TB(183±7) - 10 sont masquées (convection) |
| `plot_filter()` | Visualise le filtre à différents pas de temps (3×3 grille) |

**Critère de convection** : 
```
Si aos_1830BT > aos_1837BT - 10 → convection → masque à 0
Sinon → pas de convection → masque à 1
```

---

### ��� `SRC/regression.py`
**Rôle** : Implémentation des modèles de régression

**Fonctions principales** :

| Fonction | Description |
|----------|-------------|
| `multi_lin_reg(x_data, y_data)` | Entraîne une régression linéaire multiple avec scikit-learn |
| `test_model(model, X_test, y_test)` | Évalue le modèle : retourne prédictions, RMSE, résidus, R² |
| `random_forest_reg(x_data, y_data, n_estimators, ...)` | Entraîne une forêt aléatoire avec imprimé des importances de features |

**Métriques** :
- RMSE (Root Mean Square Error)
- R² (Coefficient de détermination)
- Résidus

---

### ��� `SRC/plots.py`
**Rôle** : Toutes les visualisations et graphiques

**Fonctions principales** :

| Fonction | Description |
|----------|-------------|
| `create_c_maps()` | Crée des colormaps personnalisées (noir, blanc) et des normalisations pour les visualisations |
| `plot_filter(x_data, filter, train_matrix, t)` | Affiche ΔT avec superposition du filtre et des points d'entraînement |
| `plot_test_model(y_data, y_pred, title, R2, rmse)` | Nuage de points : vraies vs prédites valeurs |
| `plot_residuals(y_pred, residuals, title)` | Histogramme des résidus + graphique résidu vs prédiction |
| `plot_reconstructed_velocity_map(x_data, model, filter)` | Carte 2D de la vitesse prédite |
| `plot_real_velocity_map(y_data, filter)` | Carte 2D de la vitesse mesurée |
| `plot_velocity_comparison(x_data, y_data, y_data_pred, ...)` | Comparaison côte à côte mesuré vs prédit avec colorbar partagée |
| `plot_difference_velocity_map(w_data, w_pred_tot, ...)` | Analyse des points où signes divergent (prédiction incorrecte) |

**Constantes** :
- `alpha = 0.1` : Transparence des points

---

### ��� `SRC/utils.py`
**Rôle** : Fonctions utilitaires réutilisables

**Fonctions principales** :

| Fonction | Description |
|----------|-------------|
| `linear(x, m, b)` | f(x) = mx + b |
| `exponential(x, a, b, c)` | f(x) = a · e^(bx) + c |
| `logarithmic(x, a, b)` | f(x) = a ln(x) + b |

**Usage** : Utilisées avec `scipy.optimize.curve_fit` pour fit personnalisés.

---

## ��� Flux de Données Typique

```
netCDF File
    ↓
extract_data() → dict(time, lat, lon, TB channels, W)
    ↓
create_convection_filter() → Filter 3D
    ↓
create_train_test_matrix() → Train/Test masks
    ↓
create_combined_regression_array() → X, y
    ↓
Split train/test → X_train, y_train, X_test, y_test
    ↓
[Optional] create_PCA() → X_pca
    ↓
multi_lin_reg() ou random_forest_reg() → model
    ↓
test_model() → predictions, metrics
    ↓
plot_*() → PNG files in figures/
```

---

## ��� Cas d'Usage Courants

### 1️⃣ Régression Linéaire Simple
```bash
# inputs/inputs.yaml
compute_random_forest: False
compute_PCA: False
training_ratio: 0.6
```

### 2️⃣ Random Forest avec ACP
```bash
compute_random_forest: True
n_trees: 50
compute_PCA: False  # Ignoré si Random Forest = True
```

### 3️⃣ Régression Linéaire + ACP
```bash
compute_random_forest: False
compute_PCA: True
PCA_components: 10
```

### 4️⃣ Réseau de Neurones

Dans main.neural_network directement:

`model = keras.Sequential()`
    `model.add(keras.layers.Dense(units = 128, activation = 'relu', input_shape=x_data_train.shape[1:]))`
    `model.add(keras.layers.Dense(units = 128, activation = 'relu'))`
    `model.add(keras.layers.Dense(units = 128, activation = 'relu'))`
    `model.add(keras.layers.Dense(units = 128, activation = 'relu'))`
    `model.add(keras.layers.Dense(units = 1, activation = 'linear')`

Et dans **inputs.yaml**:

Pour un test rapide

```yaml
train_ratio : 0.6                  # ratio of data points to use for Neural Network training
pop_size : 10000                   # number of data points to use for Neural Network training

learning_rate : 0.0001             # learning rate for Neural Network training
monitor_metric : 'val_loss'        # metric to monitor for early stopping
nn_epochs : 5000                   # number of epochs for Neural Network training
nn_early_stopping : True           # if true, use early stopping during Neural Network training
nn_early_stopping_patience : 70    # patience for early stopping
```

Pour les meilleurs résultats présentés ici

```yaml
train_ratio : 0.6                  # ratio of data points to use for Neural Network training
pop_size : 100000                  # number of data points to use for Neural Network training

learning_rate : 0.000001           # learning rate for Neural Network training
monitor_metric : 'val_loss'        # metric to monitor for early stopping
nn_epochs : 5000                   # number of epochs for Neural Network training
nn_early_stopping : True           # if true, use early stopping during Neural Network training
nn_early_stopping_patience : 100   # patience for early stopping
```

---

## ��� Interprétation des Résultats

Les figures générées dans `figures/` incluent :

| Fichier | Signification |
|---------|---------------|
| `Data_Vs_Prediction_*.png` | Nuage train/test : points alignés = bon modèle |
| `Residual_histograms_*.png` | Distribution résidus : centrée à 0 = sans biais |
| `Residual_analysis_*.png` | Résidus vs prédictions : pas de pattern = homoscédasticité |
| `Velocity_comparison_*.png` | Cartes 2D mesurée vs prédite |
| `Difference_velocity_map_*.png` | Zones avec désaccord signes (erreurs qualitatives) |

---

## ⚠️ Notes Importantes

- **Données** : Assurez-vous que le fichier netCDF est au bon chemin dans `paths.yaml`
- **Mémoire** : Avec 88 temps × 500×500 pixels × 10 canaux = ~1.7 GB données brutes
- **Random Forest** : Peut être très lent avec beaucoup de données, utilisez `training_ratio` et `pop_size` bas
- **ACP** : Réduit la dimensionnalité mais peut perdre l'interprétabilité physique

---

## ��� Dépannage

| Problème | Solution |
|----------|----------|
| `FileNotFoundError: netCDF file` | Vérifier chemin dans `paths.yaml` |
| `Memory Error` | Réduire `training_ratio` dans `inputs.yaml` |
| `KeyError: 'aos_1830BT'` | S'assurer que le fichier netCDF contient ces variables |
| `Shape mismatch in reshape` | Vérifier que les temps sont 87 ou 88 selon le contexte |

---

## ��� Références Techniques

### Canaux de Fréquence Disponibles
- `aos_1830BT`, `aos_1833BT`, `aos_1835BT`, `aos_1837BT`, `aos_183TBT`
- `aos_3250BT`, `aos_3253BT`, `aos_3255BT`, `aos_3257BT`, `aos_325BT`

### Données de Sortie
- `W_at_BT` : Vitesse verticale (variable cible)
- `time` : 88 pas de temps tous les 30 secondes
- Résolution spatiale : 500 × 500 pixels

---

**Auteur** : Titouan Renaud, Lorenzo de Broca et Louis Capelle 
**Dernière mise à jour** : 2026  
**Classe** : M2 ECLAT - Méthodes Statistiques de Climat
