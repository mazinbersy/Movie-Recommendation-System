# 🎬 Movie Recommendation Systems (MovieLens 100K)

This repository contains three **Jupyter Notebooks** implementing different movie recommendation system approaches on the [MovieLens 100K dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset?select=ml-100k).  
Each notebook demonstrates a different recommendation strategy and evaluates performance.

---

## 📂 Notebooks in the Repository

1. **`model_user-based.ipynb`**
   - **User-Based Collaborative Filtering**
   - Uses cosine similarity between users.
   - Normalizes ratings by subtracting each user’s mean.
   - Generates recommendations from the weighted ratings of nearest neighbors.

2. **`model_item-based.ipynb`**
   - **Item-Based Collaborative Filtering**
   - Uses cosine similarity between items.
   - Normalizes ratings by centering items on column means.
   - Recommends items similar to those already rated by the user.

3. **`model_matrix-factorization.ipynb`**
   - **Latent Factor Model with Truncated SVD**
   - Decomposes the user–item matrix into latent features.
   - Predicts ratings via matrix factorization.
   - Generates top-k recommendations from reconstructed predictions.

---

## ⚙️ Requirements

Install the required packages:

```bash
pip install pandas numpy scikit-learn
```

## 📊 Dataset

### Dataset: MovieLens 100K
- Files used:
  - ua.base → training set
  - ua.test → test set
  - u.item → movie metadata (titles, genres, etc.)
- Make sure the dataset files are placed in a folder named ml-100k/ in the repo root.

## 📈 Evaluation
### We evaluate all models using Precision@K:

- For each user in the test set:
  - Generate recommendations.
  - Relevant items = movies rated ≥ 3 in the test set.
  - Precision@K = (Relevant Recommended Items) / K.
- The final metric is averaged across all users.

### Results
|           Method            | Precision@10 |
|-----------------------------|---------------|
| User-Based CF               | 0.1013        |
| Item-Based CF               | 0.0862        |
| Matrix Factorization (SVD)  | 0.2137        |

## 📌 Notes
- Results may vary depending on parameters:
  - n_neighbors in user-based CF.
  - n_components in SVD.
  - k for top-k recommendations.

- These implementations are educational and not optimized for large-scale production.
