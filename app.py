import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Sample Dataset
# -----------------------------

content_data = pd.DataFrame({
    'content_id': [1,2,3,4,5],
    'title': ['Action Movie', 'Romantic Movie', 'Sci-Fi Movie', 'Comedy Show', 'Horror Movie'],
    'genre': ['Action', 'Romance', 'Sci-Fi', 'Comedy', 'Horror'],
    'description': [
        'Explosive action and adventure',
        'Love story and emotions',
        'Space travel and future technology',
        'Funny and entertaining show',
        'Scary and thrilling experience'
    ]
})

user_data = pd.DataFrame({
    'user_id': [101,101,102,102,103],
    'content_id': [1,3,2,4,5],
    'rating': [5,4,5,3,4]
})

# -----------------------------
# Content-Based Filtering
# -----------------------------

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_data['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommend(title, top_n=3):
    idx = content_data[content_data['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n+1]
    content_indices = [i[0] for i in similarity_scores]
    return content_data['title'].iloc[content_indices].values

# -----------------------------
# Collaborative Filtering
# -----------------------------

user_item_matrix = user_data.pivot_table(index='user_id',
                                         columns='content_id',
                                         values='rating').fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity,
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.index)

def collaborative_recommend(user_id, top_n=3):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    weighted_ratings = pd.Series(dtype='float64')

    for other_user, similarity_score in similar_users.items():
        user_ratings = user_item_matrix.loc[other_user]
        weighted_ratings = weighted_ratings.add(user_ratings * similarity_score, fill_value=0)

    recommended_items = weighted_ratings.sort_values(ascending=False)
    watched = user_item_matrix.loc[user_id]
    recommended_items = recommended_items[watched == 0]
    top_content_ids = recommended_items.head(top_n).index

    return content_data[content_data['content_id'].isin(top_content_ids)]['title'].values

# -----------------------------
# Hybrid Recommendation
# -----------------------------

def hybrid_recommend(user_id, favorite_title, top_n=3):
    content_rec = content_based_recommend(favorite_title, top_n)
    collab_rec = collaborative_recommend(user_id, top_n)
    final_rec = list(set(content_rec).union(set(collab_rec)))
    return final_rec[:top_n]

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🎬 AI Personalized Content Recommendation System")

st.write("Select User and Last Watched Content")

user_id = st.selectbox("Select User ID", user_data['user_id'].unique())
last_watched = st.selectbox("Select Last Watched Movie", content_data['title'])

if st.button("Get Recommendations"):
    recommendations = hybrid_recommend(user_id, last_watched, top_n=5)
    
    st.subheader("Recommended Content:")
    for rec in recommendations:
        st.write("👉", rec)