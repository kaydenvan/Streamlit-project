# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:04:58 2021

@author: van_s
"""

import pandas as pd
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from kneed import KneeLocator
from func.download_file import download_file
from func.upload_file import upload_file

def get_optimal_features(df, **kwargs):
    plot_pca = kwargs.get('plot_pca', False)
    pca = PCA(n_components=len(df.columns))
    pca.fit(df)
    
    # calculate how many features required
    ratio = kwargs.get('ratio', .8)
    ratio = ratio * 100
    variance = pca.explained_variance_ratio_ 
    var=np.cumsum(np.round(variance, 3)*100)
    keep_feature = np.count_nonzero(var < ratio)+1
    keep_feature = keep_feature if keep_feature < len(df.columns) else len(df.columns)
    
    if plot_pca:
        fig, ax = plt.subplots()
        ax = plt.plot(range(1, len(df.columns)+1),
                      pca.explained_variance_ratio_.cumsum(),
                      marker='o',
                      linestyle= '--')
        plt.ylabel('Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        st.write(fig)
        
    return_df_pca = kwargs.get('df_pca', False)
    df_pca = pd.DataFrame()
    if return_df_pca:
        pca = PCA(n_components=keep_feature)
        df_pca = pca.fit_transform(df)
        df_pca = pd.DataFrame(data=df_pca, 
                              columns=[f'pca_component_{i}' for i in range(1, keep_feature+1)])
        return keep_feature, df_pca
    return keep_feature

def computer_kmeans(df, clusters, **kwargs):
    elbow = kwargs.get('elbow', False)
    silouette_score = kwargs.get('silouette_score', False)
    sse = []
    sil = []
    for cluster in range(2, clusters+1):
        kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state=1)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        if silouette_score:
            sil.append(silhouette_score(df, kmeans.labels_, metric='euclidean'))
    
    # get optimal cluster number
    k1 = KneeLocator(
        range(2, clusters+1), sse, curve='convex', direction='decreasing'
        )
    optimal_k = k1.elbow
    
    kmeans = KMeans(n_clusters=optimal_k).fit(df)
    labels = kmeans.labels_
    if not elbow:
        sse =[]
    return optimal_k, labels, kmeans, sse, sil
     
@st.cache
def computer_dbscan(df, **kwargs):
    eps_ = kwargs.get('eps', .3)
    min_samples_ = kwargs.get('min_samples', 5)
    dbs = DBSCAN(eps=eps_, min_samples=min_samples_)
    dbs.fit(df)
    # core_samples_mask = np.zeros_like(dbs.labels_, dtype=bool)
    # core_samples_mask[dbs.core_sample_indices_] = True
    labels = dbs.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    optimal_k = len(set(labels)) - (1 if -1 in labels else 0)
    
    n_noise_ = dbs.labels_[dbs.labels_ == -1].size
    return optimal_k, labels, dbs, n_noise_

# @st.cache
def plot_elbow(clusters, sse):
    fig, ax = plt.subplots()
    ax = plt.plot(range(2, clusters+1), sse)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    return fig

def plot_silhouette_score(clusters, sil):
    fig, ax = plt.subplots()
    ax = plt.plot(range(2, clusters+1), sil)
    plt.title('Silhouette Coefficients')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    return fig

def plot_kmean_correlation(df, kmean_model, x_axis, y_axis):
    fig, ax = plt.subplots(figsize=(12,8))
    sns.scatterplot(data=df, x=x_axis, y=y_axis,
                   hue='y_pred', ax=ax)
    plt.scatter(x=kmean_model.cluster_centers_[:,df.columns.to_list().index(x_axis)],
               y=kmean_model.cluster_centers_[:,df.columns.to_list().index(y_axis)],
               c='red', s=100)
    plt.title(f'Clustering correlation: {x_axis} against {y_axis}')
    plt.suptitle('Clustering', fontsize=20, weight='bold', y=.95)
    return fig

def plot_dbscan(df, model):
    fig, ax = plt.subplots()
    unique_labels = set(model.labels_)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (model.labels_ == k)
        
        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
        
        xy = df[class_member_mask & core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
    
        xy = df[class_member_mask & ~core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    return fig

def download_pred_df(df):
    """
    Purpose: create a link to download result file
    """
    st.markdown('**disable download function at the moment**')
    # download(df)

def auto_clustering():
    st.title('Auto Clustering')
    st.write('This app is powered by Streamlit, Sklearn.')
    df, uploaded = upload_file(file_type = ['csv', 'xlsx', 'xls'], show_file_info = True)
    
    if not uploaded:
        st.stop()
        
    preview_df = st.checkbox('Preview dataframe')
    if preview_df:
        st.subheader('Preview uploaded dataframe') if uploaded else st.subheader('Preview demo dataframe')
        st.dataframe(df.head())
    
    st.write('If you would like to do EDA for the dataset, please reach to the EDA page accordingly')
    
    
    # data cleaning for KMeans
    imp_median = SimpleImputer(strategy='median')
    imp_freq = SimpleImputer(strategy='most_frequent')
    oe = OrdinalEncoder()
    standard = StandardScaler()
    
    for col in df.select_dtypes(include=['number']):
        df[col] = imp_median.fit_transform(np.array(df[col]).reshape(-1, 1))
        df[col] = standard.fit_transform(np.array(df[col]).reshape(-1, 1))
    for col in df.select_dtypes(include=['object']):
        df[col] = imp_freq.fit_transform(np.array(df[col]).reshape(-1, 1))
        df[col] = oe.fit_transform(np.array(df[col]).reshape(-1, 1))
    
    model_option = st.sidebar.radio('Choose the methodology to cluster the data', 
                                    ('KMeans', 'DBSCAN'),
                                    help='Select clustering method')
    
    pca_ = st.sidebar.checkbox('Enable PCA', help='Use PCA (1) to analysis and (2) transform and reduce dimension')
    df_pca = pd.DataFrame()
    if pca_:
        ratio_ = st.sidebar.slider('% of PCA you want to cover', min_value=.1, max_value=1., value=.8, step=.01,
                          help='PCA is used to minimize dimension in the dataset and by default to cover 80% of feature.')
        # can add slider to ask what % variance has to be covered
        optimal_feature, df_pca = get_optimal_features(df, plot_pca=True, ratio=ratio_, df_pca=True)
        st.write(f'{optimal_feature} feature(s) is/are used for clustering. It can cover {ratio_*100}% of original features')
    
    if not df_pca.empty:
        df = df_pca

    if st.checkbox('Preview cleaned dataset'):
        st.dataframe(df)
    
    if model_option == 'KMeans':
        elbow_ = st.sidebar.checkbox('Elbow Method', help='Check it if you want to show Elbow Chart')
        silouette_score_ = st.sidebar.checkbox('Silhouette Coefficients', help='Check it if you want to show Silhouette Coefficients')
        plot_cluster_correlation_ = st.sidebar.checkbox('Plot clustering verus dimensions', help='Check it if you want to select how dimensions correlate with clustering')
        
        clusters = st.number_input('Number of Clusters you want to test',
                               min_value = 2,
                               max_value = 20,
                               value=10,
                               step=1,
                               help='At least 2 cluster has to be formed')
        clusters = int(clusters)
        # st.write(f'The model will be tested in {clusters} clusters')
        
        optimal_k, df['y_pred'], model, sse, sil = computer_kmeans(df, clusters, elbow=elbow_, silouette_score=silouette_score_)
        st.write(f'Optimal number of cluster under Elbow Method: {optimal_k}')
        download_pred_df(df)
        
        # plot Elbow Method
        if elbow_:
            st.write(plot_elbow(clusters, sse))
        
        # plot Silhouette Coefficient
        if silouette_score_:
            st.write(plot_silhouette_score(clusters, sil))
    
        if plot_cluster_correlation_:
            col1, col2 = st.columns(2)
            plot_columns = df.columns.to_list()
            plot_columns.remove('y_pred')
            with col1:
                x_axis = st.selectbox('Select x axis', plot_columns)
            with col2:
                y_axis = st.selectbox('Select y axis', plot_columns)
        
            st.write(plot_kmean_correlation(df, model, x_axis, y_axis))
        
    elif model_option == 'DBSCAN':
        eps_ = st.sidebar.slider('ε(eps)', min_value=.01, max_value=1., value=.5)
        min_samples_ = st.sidebar.number_input('Min sample', min_value=1, value=5, step=1)
        
        optimal_k, df['y_pred'], model, n_noise_ = computer_dbscan(df, eps=eps_, min_samples=min_samples_)
        st.write(f'Optimal number of cluster under DBSCAN: {optimal_k}')
        download_pred_df(df)
        
        fig = plot_dbscan(df, model)
        plt.title('Estimated number of clusters: %d' % optimal_k)
        st.write(fig)
        st.sidebar.write(f'Number of noise data: {n_noise_}')