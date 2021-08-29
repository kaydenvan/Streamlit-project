# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:04:58 2021

@author: van_s
"""

# yet to finalize the demo

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

def iris_dataset():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    return df

def get_optimal_features(df, **kwargs):
    """
    find optimal features from data
    df: pd.DataFrame, input dataset
    plot_pca: boolean, plot for pca against feature coverage
    ratio: float, pca ratio
    df_pca: boolean, return transformed pca dataframe or not
    """
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
    """
    df: pd.DataFrame, input dataset
    clusters: int, number of clusters to be tested
    elbow: boolean, return elbow result 
    silouette_score: boolean, return silouette score
    """
    sse = []
    sil = []
    for cluster in range(2, clusters+1):
        kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state=1)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        sil.append(silhouette_score(df, kmeans.labels_, metric='euclidean'))
    
    # get optimal cluster number
    k1 = KneeLocator(
        range(2, clusters+1), sse, curve='convex', direction='decreasing'
        )
    optimal_k = k1.elbow
    
    kmeans = KMeans(n_clusters=optimal_k).fit(df)
    labels = kmeans.labels_
    return optimal_k, labels, kmeans, sse, sil
     
@st.cache
def computer_dbscan(df, **kwargs):
    eps_ = kwargs.get('eps', .3)
    min_samples_ = kwargs.get('min_samples', 5)
    dbs = DBSCAN(eps=eps_, min_samples=min_samples_)
    dbs.fit(df)
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
    show_result = st.beta_expander('Preview result dataset', expanded=False)
    show_result.dataframe(df.head(50))
    # download_file(df)

def auto_clustering():
    st.title('Auto Clustering')
    st.write('This app is powered by Streamlit, Sklearn.')
    df = pd.DataFrame()
    
    df, uploaded, file_name = upload_file(file_type = ['csv', 'xlsx', 'xls'], return_file_name = True)
        
    if not uploaded:
        demo = st.sidebar.radio('Enable Demo', ('Yes', 'No'), index=1,
                                help='Iris dataset is used for demonstration purpose')
        if demo == 'Yes':
            df = iris_dataset()
            df.drop(columns = 'target', inplace=True)
    else:
        demo = 'No'
        
    # check if the uploaded file refreshed or not
    # if the file refreshed, clean the model created before
    if 'file_name' not in st.session_state:
        st.session_state.file_name = file_name
    elif st.session_state.file_name != file_name:
        if 'model' in st.session_state:
            del st.session_state.model
        st.session_state.file_name = file_name
    
    # if dataframe is empty, stop program
    if df.empty:
        st.stop()
        
    show_upload = st.beta_expander('Preview uploaded dataframe', expanded=True) if uploaded else st.beta_expander('Preview demo dataframe', expanded=True)
    show_upload.dataframe(df.head(50))

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
    
    model_option_ = st.sidebar.selectbox('Choose the methodology to cluster the data', 
                                        ('KMeans', 'DBSCAN'), index=0,
                                        help='Select clustering method')
    train_model_ = st.sidebar.button('Confirm', help='Confirm model to train') if demo == 'No' else True
    
    # Layout
    # create sidebar expander
    if model_option_ == 'KMeans':
        model_selection_ = st.sidebar.beta_expander('KMeans Configuration', expanded=False)  
    elif model_option_ == 'DBSCAN':
        model_selection_ = st.sidebar.beta_expander('DBSCAN Configuration', expanded=False)  

    # Layout
    pca_ = model_selection_.checkbox('Enable PCA', help='Use PCA (1) to analysis and (2) transform and reduce dimension')\
        if demo == 'No' else True
            
    if model_option_ == 'KMeans':
        elbow_ = model_selection_.checkbox('Elbow Method', 
                                           help='Check it if you want to show Elbow Chart') if demo == 'No' else True
        silouette_score_ = model_selection_.checkbox('Silhouette Coefficients', 
                                                     help='Check it if you want to show Silhouette Coefficients') if demo == 'No' else True
        plot_cluster_correlation_ = model_selection_.checkbox('Plot Clusters vs Dimensions', 
                                                              help='Check it if you want to select how dimensions correlate with clustering') if demo == 'No' else True
    elif model_option_ == 'DBSCAN':
        eps_ = model_selection_.slider('ε(eps)', min_value=.01, max_value=1., value=.5)
        min_samples_ = model_selection_.number_input('Min sample', min_value=1, value=5, step=1)
    # pca analysis
    
    df_pca = pd.DataFrame()
    if pca_ and df_pca.empty:
        ratio_ = st.sidebar.slider('% of PCA you want to cover', min_value=.1, max_value=1., value=.8, step=.01,
                          help='PCA is used to minimize dimension in the dataset and by default to cover 80% of feature.')
        # can add slider to ask what % variance has to be covered
        optimal_feature, df_pca = get_optimal_features(df, plot_pca=True, ratio=ratio_, df_pca=True)
        st.write(f'{optimal_feature} feature(s) is/are used for clustering. It can cover {ratio_*100}% of original features')
        show_transform = st.beta_expander('Preview transformed dataset', expanded=False) \
            if demo == 'No' else st.beta_expander('Preview transformed dataset', expanded=True) 
        show_transform.write(df.head(50))
    
    if not df_pca.empty:
        df = df_pca
    
    if model_option_ == 'KMeans':
        # clusters = st.number_input('Number of Clusters you want to test', min_value = 2,max_value = 20,
        #                        value=10, step=1, help='At least 2 cluster has to be formed')
        # for simplicity, default to test 20 clusters
        clusters = int(20)
        
    if train_model_:
        if model_option_ == 'KMeans':
            optimal_k, df['y_pred'], model, sse, sil = computer_kmeans(df, clusters)
            st.session_state.optimal_k = optimal_k
            st.session_state.df = df
            st.session_state.model = model
            st.session_state.sse = sse
            st.session_state.sil = sil
            st.session_state.model_option_ = model_option_
            st.write('run completed')
        if model_option_ == 'DBSCAN':
        #     optimal_k, df['y_pred'], model, n_noise_ = computer_dbscan(df, eps=eps_, min_samples=min_samples_)
        #     st.session_state.optimal_k = optimal_k
        #     st.session_state.df = df
        #     st.session_state.model = model
        #     st.session_state.n_noise_ = n_noise_
            st.session_state.model_option_ = model_option_
    
    if 'model' not in st.session_state:
        st.stop()
    if st.session_state.model_option_ != model_option_:
        st.warning('The trained model is not aligned to the selected model. Please click confirm to retrain the model')
        
    st.write(f'Optimal number of cluster: {st.session_state.optimal_k}')
    download_pred_df(st.session_state.df)
        
    if model_option_ == 'KMeans':
        # plot Elbow Method
        col1, col2 = st.beta_columns(2)
        if elbow_ or demo == 'Yes':
            col1.write('Elbow Method Graph')
            col1.write(plot_elbow(clusters, st.session_state.sse))
        
        # plot Silhouette Coefficient
        if silouette_score_ or demo == 'Yes':
            col2.write('Silhouette Score Graph')
            col2.write(plot_silhouette_score(clusters, st.session_state.sil))
    
        # plot interactive clustering chart
        if plot_cluster_correlation_ or demo == 'Yes':
            col3, col4 = st.beta_columns(2)
            plot_cols_x = st.session_state.df.columns.to_list()
            plot_cols_x.remove('y_pred')
            with col3:
                x_axis = st.selectbox('Select x axis', plot_cols_x)
            with col4:
                plot_cols_y = plot_cols_x.copy()
                plot_cols_y.remove(x_axis)
                y_axis = st.selectbox('Select y axis', plot_cols_y)
            st.write(plot_kmean_correlation(st.session_state.df, 
                                            st.session_state.model, x_axis, y_axis))
        
    if model_option_ == 'DBSCAN':  
        # since it is more interactive to see the graph changed with the parameters changed
        # it enables real time clustering at the moment.
        st.warning('Real time clustering is used for more interactive response')
        optimal_k, df['y_pred'], model, n_noise_ = computer_dbscan(df, eps=eps_, min_samples=min_samples_)
        fig = plot_dbscan(df, model)
        plt.title('Estimated number of clusters: %d' % optimal_k)
        st.write(fig)
        st.sidebar.write(f'Number of noise data: {n_noise_}')