flowchart TD
    A[TMDB API] -->|Requests<br/>pagination, retry, rate-limit| B[Extraction<br/>src/extract.py]
    B -->|JSON brut| C[(data/raw/)]
    C --> D[Nettoyage<br/>src/clean.py]
    D -->|budget/revenue/runtime 0 to NaN<br/>dates parsées| E[(data/processed/<br/>movies_clean.pkl)]
    E --> F[(MongoDB<br/>movie_db.movies)]
    E --> G[Feature Engineering<br/>src/features.py]
    G -->|year, decade, n_genres,<br/>runtime_category, overview_clean| H[(movies_features.pkl)]

    H --> I[TF-IDF<br/>TfidfVectorizer]
    I --> I2[(tfidf_matrix.npz<br/>tfidf_vectorizer.pkl)]

    H --> J[Classification<br/>Random Forest<br/>GridSearchCV]
    J --> J2[(classifier_rf.pkl)]

    H --> K[Clustering<br/>K-Means K=5]
    K --> K2[(kmeans_model.pkl<br/>cluster_scaler.pkl)]

    I2 --> L[Recommandation<br/>Cosine Similarity<br/>src/recommend.py]

    J2 --> M[Streamlit App]
    K2 --> M
    L --> M
    H --> M

    M --> M1[Dashboard]
    M --> M2[Classification]
    M --> M3[Clusters]
    M --> M4[Recommandation]

    subgraph Orchestration
        N[Airflow DAG<br/>extract >> clean >> features >> mongo_store >> train]
    end
    N -.orchestre.-> B
    N -.orchestre.-> D
    N -.orchestre.-> G
    N -.orchestre.-> F
    N -.orchestre.-> J
    N -.orchestre.-> K

    subgraph Docker[Docker Compose]
        F
        N
        M
    end