import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from plotly import express as px
import json
from sentence_transformers import SentenceTransformer
import textwrap

def str_to_float_list(s):
    return json.loads(s)

@st.cache_data
def get_reduced_data(method, n_components, perplexity, data):
    if method == "PCA": 
        pca = PCA(n_components=n_components)
        return pca.fit_transform(np.stack(data['embeddings'].values))
    else: 
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        return tsne.fit_transform(np.stack(data['embeddings'].values))

@st.cache_data 
def get_clusters(method, n_clusters, eps, min_samples, reduced_data):
    if method == "K-Means":
        kmeans = KMeans(n_clusters=n_clusters)
        return kmeans.fit_predict(reduced_data)
    else:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(reduced_data)

@st.cache_data
def get_embeddings(_model, model_name, sentence):
    print(model_name)
    embeddings = _model.encode([sentence])[0].tolist()
    print(embeddings)
    return embeddings

def main():
    model_name = 'paraphrase-multilingual-mpnet-base-v2'
    model = SentenceTransformer(model_name)

    st.text("Persee's documents' abstracts' 3D reduced embeddings and topic clustering")

    st.sidebar.title("Upload CSV")
    csv_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    
    if csv_file is not None:
        data = pd.read_csv(csv_file)
        data = data[data['year'].notna()]
        data['embeddings'] = data['embeddings'].apply(str_to_float_list)

        st.sidebar.write(data.head())

        added_sentence = st.sidebar.text_input("Recherche textuelle", value = "La culture Russe vue de la France")
        embeddings = get_embeddings(model, model_name, added_sentence)
        new_row = pd.DataFrame({'title': ["Recherche textuelle"], 'year': [None], 'content': [added_sentence], 'embeddings': [embeddings]})
        print(new_row)
        data = pd.concat([data, new_row], ignore_index=True)

        data['content_formatted'] = data['content'].apply(lambda text: '</br>'.join(textwrap.wrap(text, width=40)))

        st.sidebar.title("Dimensionality Reduction")
        method = st.sidebar.radio("Choose a method", ("PCA", "t-SNE"))
        n_components = 3

        perplexity = None
        if method == "t-SNE":
            perplexity = st.sidebar.number_input("Perplexity", value=30)

        reduced_data = get_reduced_data(method, n_components, perplexity, data)

        st.sidebar.title("Clustering")
        clustering_method = st.sidebar.radio("Choose a method", ("K-Means", "DBSCAN"))
        n_clusters = None
        min_samples = None
        eps = None
        if clustering_method == "K-Means":
            n_clusters = st.sidebar.number_input("Number of clusters", value=4) 
        else:
            eps = st.sidebar.number_input("Epsilon", value=0.5)
            min_samples = st.sidebar.number_input("Minimum samples", value=5)
        
        data['cluster'] = get_clusters(clustering_method, n_clusters, eps, min_samples, reduced_data)
        st.sidebar.title("Year Range")
        min_year = st.sidebar.number_input("Minimum year", value=data['year'].min())
        max_year = st.sidebar.number_input("Maximum year", value=data['year'].max())

        filter_cluster = st.checkbox('Show research cluster only', value=False)
        max_cluster = data['cluster'].max()
        search_cluster = data.loc[data['year'].isnull(), 'cluster'].iloc[0]
        print(search_cluster)

        data.loc[data['year'].isnull(), 'cluster'] = max_cluster * 2
        filtered_data = data
        if filter_cluster:
            filtered_data = data[(data['cluster'] == search_cluster) & (((data['year'] >= min_year) & (data['year'] <= max_year)) | (data['year'].isnull()))]
        else:
            filtered_data = data[(data['cluster'] != -1) & (((data['year'] >= min_year) & (data['year'] <= max_year)) | (data['year'].isnull()))]
        
        # Get the row indices of the filtered data
        filtered_indices = filtered_data.index

        # Filter the same rows in the reduced_data ndarray
        filtered_reduced_data = reduced_data[filtered_indices, :]

        fig = px.scatter_3d(
            filtered_data,
            x=filtered_reduced_data[:, 0],
            y=filtered_reduced_data[:, 1],
            z=filtered_reduced_data[:, 2],
            custom_data=['content_formatted', 'year'],
            color='cluster',
        )

        fig.update_traces(
            hovertemplate="<br>".join([
                "Year: %{customdata[1]}",
                "Content %{customdata[0]}",
            ])
        )

        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
