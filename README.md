# Movie Intelligence — Analyse, Classification, Clustering et Recommandation de Films

Plateforme d'analyse intelligente d'un catalogue de films à partir de données TMDB : classification des films à fort engagement, clustering non supervisé, et recommandation par contenu (TF-IDF).

## Contexte et objectifs métier

Une plateforme de contenus souhaite mieux comprendre son catalogue de films et proposer des recommandations pertinentes. Le projet répond à trois problématiques :
1. **Classification** — prédire si un film appartient à une catégorie à fort engagement (`high_engagement`).
2. **Clustering** — identifier automatiquement des groupes de films similaires (K-Means).
3. **Recommandation** — proposer des films similaires à partir du contenu textuel (TF-IDF + similarité cosinus).

## Source des données — API TMDB

Extraction automatisée via l'API publique TMDB (`/discover/movie` pour la liste, `/movie/{id}` pour les détails + mots-clés). 1 719 films exploitables après nettoyage (sur ~1 971 extraits), filtrés sur `vote_count >= 20`.

Champs extraits : `movie_id, title, overview, release_date, runtime, original_language, genres, keywords, budget, revenue, popularity, vote_average, vote_count`.

Les réponses JSON brutes sont sauvegardées dans `data/raw/` avant tout traitement, pour éviter de solliciter l'API à nouveau en cas d'erreur de nettoyage. L'extraction gère la pagination, les erreurs HTTP (429 avec retry, 404, timeouts), et reprend automatiquement si interrompue (fichiers déjà présents ignorés).

## Nettoyage et préparation des données

- `budget`, `revenue`, `runtime` à `0` → convertis en `NaN` (TMDB utilise 0 pour "inconnu", pas une vraie valeur).
- `release_date` converti en datetime ; lignes sans date ou sans overview supprimées.
- Les valeurs manquantes (budget/revenue/runtime) ne sont **pas** remplies lors du nettoyage — elles sont imputées plus tard, à l'intérieur du pipeline scikit-learn (médiane calculée sur le train uniquement), pour éviter toute fuite de données (data leakage).
- Stockage dans **MongoDB** (`movie_db.movies`), un document par film, upsert sur `movie_id`. Requêtes et agrégation (`$unwind`, `$group`, `$match`, `$sort`, `$project`) : note moyenne par genre.

## Analyse exploratoire (EDA)

Visualisations principales (voir `notebooks/01_eda.ipynb`) :
- Distribution des notes (symétrique, centrée 6–8).
- Popularité (fortement asymétrique à droite, nombreux outliers).
- Films par genre (Action ~600, Documentary/TV Movie ~10 — fort déséquilibre).
- Sorties par année (croissance nette depuis 1980).
- Durée (90–120 min typique).
- Budget vs Revenue (corrélation 0.64).
- `vote_count` vs `popularity` (corrélation ~0.03 — signaux indépendants).
- `vote_count` vs `vote_average` (corrélation 0.39 — le futur target ne duplique pas la note).

## Feature Engineering

- `year`, `month`, `decade` (depuis `release_date`).
- `n_genres`, `n_keywords` (comptage des listes).
- `runtime_category` (short / medium / long / unknown, seuils basés sur l'EDA).
- `overview_length` (nombre de mots du synopsis).
- Genres encodés en multi-label (`genre_<Nom>`, via `MultiLabelBinarizer`) pour la classification.

Aucune variable créée à partir de `vote_count` n'est utilisée ailleurs que pour construire la cible, afin d'éviter le data leakage.

## Traitement NLP — TF-IDF

`TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")` appliqué sur `overview` nettoyé (minuscules, ponctuation retirée). Matrice finale : `(1719, 5000)`.

Analyse : les mots génériques ("world", "life", "family") obtiennent un score moyen faible (pénalisés par l'IDF car présents dans beaucoup de films), tandis que les mots spécifiques à un film donné ("inducing heights", "thailand" pour un film d'escalade) obtiennent un score élevé pour ce film précis — comportement attendu du TF-IDF.

## Classification — `high_engagement`

**Cible** : top 25% des films par `vote_count` (seuil = 75e percentile, ≈ 8566 votes). Choix d'un seuil basé sur percentile plutôt que fixe, car `vote_count` est fortement asymétrique.

**Règle anti-leakage** : `vote_count` (variable source du target) et `vote_average` (fortement lié à l'engagement) sont exclus des features.

**Features** : numériques (`runtime, budget, revenue, popularity, n_genres, n_keywords, overview_length`), catégorielles (`original_language, runtime_category`), et genres multi-label.

**Modèles comparés** (test set 20%, stratifié) :

| Modèle | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.77 | 0.47 | 0.58 | 0.84 |
| Linear SVM | 0.76 | 0.44 | 0.56 | 0.84 |
| **Random Forest** | 0.65 | 0.72 | 0.68 | 0.89 |

Random Forest retenu : meilleur recall sur la classe minoritaire grâce à `class_weight="balanced"` et à sa nature non-linéaire/ensembliste.

**Validation croisée** (StratifiedKFold, 5 folds) : F1 moyen 0.669 (std 0.016), ROC-AUC moyen 0.882 (std 0.012) — résultats stables.

**GridSearchCV** (grille sur `n_estimators`, `max_depth`, `min_samples_leaf`, scoring F1) :

| | Avant (défaut) | Après (optimisé) |
|---|---|---|
| Recall (1) | 0.72 | 0.83 |
| F1 (1) | 0.68 | 0.70 |
| ROC-AUC | 0.8885 | 0.8889 |

**Features supplémentaires testées** : l'ajout des genres (multi-label) a amélioré les résultats (signal réel). L'ajout du TF-IDF complet comme feature a au contraire dégradé la performance (overfitting lié au ratio élevé features/lignes sur ce jeu de données de taille modeste) — le modèle final conserve les genres mais exclut le TF-IDF des features de classification.

## Clustering — K-Means

Clustering sur 7 variables structurées scalées (`runtime, budget, revenue, popularity, vote_average, n_genres, n_keywords`). Test de K=2 à K=10 via le Silhouette Score ; K=2 obtient le meilleur score (0.274) mais produit une séparation trop grossière. **K=5 retenu** (score 0.192), meilleur compromis entre qualité et interprétabilité.

| Cluster | Taille | Profil |
|---|---|---|
| 0 | 600 | Budget modéré, note plus faible |
| 1 | 607 | Budget plus faible, meilleure note, le plus de mots-clés |
| 2 | 263 | Budget élevé, le plus de genres |
| 3 | 14 | Popularité extrême (outliers viraux) |
| 4 | 235 | Budget le plus élevé, films premium |

Visualisation PCA 2D (seulement ~49% de variance expliquée — le cluster 3, défini par la popularité, ne se distingue pas visuellement malgré une séparation claire dans les données réelles).

## Recommandation (Bonus)

Similarité cosinus sur la matrice TF-IDF. Pour un film donné : récupération de son vecteur, calcul de similarité avec tous les autres films, retour des 5 plus proches (titre, genres, année, score). Testé sur plusieurs films (ex. "The Godfather" → sa suite directe en premier résultat, score 0.461).

## Application Streamlit

4 pages : Dashboard (EDA interactive via Plotly), Classification (formulaire de prédiction live), Clusters (profils + navigation par cluster), Recommandation (sélection d'un film → 5 similaires).

## Automatisation — Airflow

DAG `movie_intelligence_pipeline` : `extract >> clean >> features >> mongo_store >> train`. Chaque étape est un script Python indépendant (`src/`), orchestré via Airflow en conteneur Docker.

## Conteneurisation — Docker

Services Docker Compose : `mongo`, Airflow (apiserver, scheduler, worker, triggerer, dag-processor, postgres, redis), `streamlit`. Un seul `docker compose up -d` démarre l'ensemble de la stack.

**Important** : les dépendances scikit-learn sont figées (`scikit-learn==1.9.1`) dans `requirements.txt` pour garantir la compatibilité des modèles sauvegardés (`.pkl`) entre les environnements.

## Installation et exécution

### Prérequis
- Docker et Docker Compose
- Clé API TMDB (`.env` : `TMDB_API_KEY=...`)

### Lancer toute la stack
```bash
docker compose up -d --build
```

### Accès
- Streamlit : http://localhost:8501
- Airflow : http://localhost:8080 (login: airflow / airflow)
- MongoDB : localhost:27017

### Exécuter le pipeline manuellement
Déclencher le DAG `movie_intelligence_pipeline` depuis l'interface Airflow, ou :
```bash
python src/extract.py
python src/clean.py
python src/features.py
python src/mongo_store.py
python src/train.py
```

## Structure du projet