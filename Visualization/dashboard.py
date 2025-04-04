import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.stats as stats

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("../data/preprocessed_genes/all_genes/brain_count_overlap_hvg_labeled.csv")
    data['involve_cancer'] = data['involve_cancer'].astype(str)  # Ensure consistent labels
    return data

df = load_data()


# UMAP Visualization
def plot_umap(df):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings = reducer.fit_transform(df.drop('involve_cancer', axis=1))

    df_umap = df.copy()
    df_umap['UMAP_1'] = umap_embeddings[:, 0]
    df_umap['UMAP_2'] = umap_embeddings[:, 1]

    fig = px.scatter(
        df_umap,
        x='UMAP_1',
        y='UMAP_2',
        color='involve_cancer',
        color_discrete_map={'True': '#FF4B4B', 'False': '#4CAF50'},
        title='UMAP Clustering Visualization'
    )
    return fig

# t-SNE Visualization
def plot_tsne(df):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_embeddings = tsne.fit_transform(df.drop('involve_cancer', axis=1))

    df_tsne = df.copy()
    df_tsne['TSNE_1'] = tsne_embeddings[:, 0]
    df_tsne['TSNE_2'] = tsne_embeddings[:, 1]

    fig = px.scatter(
        df_tsne,
        x='TSNE_1',
        y='TSNE_2',
        color='involve_cancer',
        color_discrete_map={'True': '#FF4B4B', 'False': '#4CAF50'},
        title='t-SNE Clustering Visualization'
    )
    return fig

# PCA Visualization
def plot_pca(df):
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(df.drop('involve_cancer', axis=1))

    df_pca = df.copy()
    df_pca['PCA_1'] = pca_embeddings[:, 0]
    df_pca['PCA_2'] = pca_embeddings[:, 1]

    fig = px.scatter(
        df_pca,
        x='PCA_1',
        y='PCA_2',
        color='involve_cancer',
        color_discrete_map={'True': '#FF4B4B', 'False': '#4CAF50'},
        title='PCA Clustering Visualization'
    )
    return fig

# KMeans Clustering Visualization
def plot_kmeans(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_kmeans = df.copy()
    df_kmeans['Cluster'] = kmeans.fit_predict(df_kmeans.drop('involve_cancer', axis=1))

    fig = px.scatter(
        df_kmeans,
        x='PCA_1',  # Can be UMAP_1, TSNE_1, etc.
        y='PCA_2',  # Can be UMAP_2, TSNE_2, etc.
        color='Cluster',
        title=f'KMeans Clustering (n={n_clusters})'
    )
    return fig

# Filter gene columns (excluding computed ones)
def get_gene_columns(df):
    computed_cols = ['PCA_1', 'PCA_2', 'UMAP_1', 'UMAP_2', 'TSNE_1', 'TSNE_2', 'Total_Gene_Expression']
    return [col for col in df.select_dtypes(include=[np.number]).columns if col not in computed_cols]


def plot_gene_expression_histogram(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df.sort_values('involve_cancer'),
        x=df.select_dtypes(include=[np.number]).iloc[:, 1:].sum(axis=1), 
        bins=30, 
        kde=True, 
        hue='involve_cancer'
    )
    plt.title("Distribution of Gene Expression Across Samples by Cancer Status")
    plt.xlabel("Total Gene Expression")
    plt.ylabel("Frequency")
    plt.legend(title="Cancer Status", labels=["Cancerous", "Non-Cancerous"])
    st.pyplot(plt)


# Main Page
def main_page():
    st.title("🧬 Cancer Prediction Dashboard")
    st.markdown("This dashboard visualizes the **gene expression data** for cancer prediction.")

    # Overview Section
    with st.expander("📋 Data Overview"):
        st.info(f"**Dataset Shape:** {df.shape}")
        st.write(df.head())

    # Cancer Status Distribution Section
    with st.expander("🩺 Cancer Status Distribution"):
        st.write("### Cancer Status Distribution")
        fig = px.histogram(
            df, 
            x='involve_cancer', 
            color='involve_cancer', 
            color_discrete_map={'True': '#FF4B4B', 'False': '#4CAF50'},
            title='Cancer Status Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cancer Status Distribution Section
    with st.expander("🩺 Cancer Histogram"):
        st.write("### Cancer Histogram")
        plot_gene_expression_histogram(df)

    # Gene Distribution Section
    with st.expander("📈 Gene Distributions"):
        st.write("### KDE Plots for Gene Distribution")
        selected_genes = st.multiselect(
            "Select Genes for KDE Plots", 
            df.columns[:-1], 
            default=["0.1", "1", "2"]
        )
        if selected_genes:
            sampled_df = df[selected_genes].sample(500).reset_index()
            fig = px.line(sampled_df, x='index', y=selected_genes,
                            labels={'value': 'Gene Expression'},
                            title='KDE Plot of Selected Genes')
            st.plotly_chart(fig)


    # Correlation Heatmap
    with st.expander("🌡️ Correlation Heatmap"):
        st.write("### 🌡️ Correlation Heatmap (Top 20 Genes)")
        top_corr_genes = df.select_dtypes(include=[np.number]).corr()['involve_cancer'].abs().nlargest(21).index
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df[top_corr_genes].corr(), cmap='coolwarm', center=0, annot=False, cbar_kws={'shrink': 0.8})
        st.pyplot(fig)

    # Outlier Analysis Section
    with st.expander("📦 Outlier Analysis"):
        st.write("### Boxplots for Outlier Detection")
        selected_gene = st.selectbox("📋 Select a Gene for Boxplot", df.select_dtypes(include=[np.number]).columns[:-1])
        fig = px.box(df, x='involve_cancer', y=selected_gene, color='involve_cancer')
        st.plotly_chart(fig)

    # Volcano Plot
    with st.expander("🌡️ Volcano Plot"):
        st.write("### 🌋 Volcano Plot - Gene Expression Analysis")
        p_values, fold_changes = [], []
        genes = df.select_dtypes(include=[np.number]).columns[:-1]

        for gene in genes:
            cancer_values = df[df['involve_cancer'] == 1][gene]
            non_cancer_values = df[df['involve_cancer'] == 0][gene]
            _, p = mannwhitneyu(cancer_values, non_cancer_values, alternative='two-sided')
            p_values.append(p)
            fold_changes.append(np.log2(cancer_values.mean() / non_cancer_values.mean()))

        volcano_df = pd.DataFrame({
            'Gene': genes,
            'p_value': p_values,
            'log_p_value': -np.log10(p_values),
            'log2_fold_change': fold_changes
        })

        fig = go.Figure()
        significant_genes = volcano_df[(volcano_df['p_value'] < 0.05) & (abs(volcano_df['log2_fold_change']) > 1)]
        non_significant_genes = volcano_df[~volcano_df.index.isin(significant_genes.index)]

        fig.add_trace(go.Scatter(x=non_significant_genes['log2_fold_change'],
                                    y=non_significant_genes['log_p_value'],
                                    mode='markers',
                                    marker=dict(color='gray', size=6),
                                    name='Non-significant'))

        fig.add_trace(go.Scatter(x=significant_genes['log2_fold_change'],
                                    y=significant_genes['log_p_value'],
                                    mode='markers',
                                    marker=dict(color='red', size=8),
                                    name='Significant Genes'))

        fig.update_layout(title='Volcano Plot - Gene Expression Analysis',
                            xaxis_title='Log2 Fold Change',
                            yaxis_title='-Log10 p-value',
                            template='plotly_white')

        st.plotly_chart(fig, use_container_width=True)

    # Clustering Visualizations Section
    with st.expander("🧩 Clustering Visualizations"):
        st.write("### Select Clustering Visualization")

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]))

        df['PCA_1'] = pca_result[:, 0]
        df['PCA_2'] = pca_result[:, 1]

        cluster_option = st.selectbox(
            "Choose a Clustering Technique:",
            ["UMAP", "t-SNE", "PCA", "KMeans"]
        )

        if cluster_option == "UMAP":
            st.plotly_chart(plot_umap(df.select_dtypes(include=[np.number])), use_container_width=True)
        elif cluster_option == "t-SNE":
            st.plotly_chart(plot_tsne(df.select_dtypes(include=[np.number])), use_container_width=True)
        elif cluster_option == "PCA":
            st.plotly_chart(plot_pca(df.select_dtypes(include=[np.number])), use_container_width=True)
        elif cluster_option == "KMeans":
            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            st.plotly_chart(plot_kmeans(df.select_dtypes(include=[np.number]), n_clusters), use_container_width=True)


def statistical_page():
    st.title("📈 Statistical Analysis")
    st.markdown("### 🔍 Descriptive Statistics")
    data = df.select_dtypes(include=[np.number])
    st.write(data.describe())

    # Skewness & Kurtosis
    st.write("### 📐 Skewness & Kurtosis")
    stats_df = pd.DataFrame({
        'Skewness': data.skew(),
        'Kurtosis': data.kurtosis()
    }).style.background_gradient(cmap='coolwarm')
    st.write(stats_df)

    # Top Correlated Genes
    st.write("### 🔬 Top 5 Genes Most Correlated with Cancer")
    correlation_with_cancer = data.corr()['involve_cancer'].drop('involve_cancer').sort_values(ascending=False)
    st.write(correlation_with_cancer.head(5).to_frame().style.background_gradient(cmap='Blues'))

    # Outlier Analysis
    st.write("### 🚨 Outlier Analysis")
    outliers = data[(data > data.quantile(0.99)).any(axis=1)]
    st.write(outliers)

# Sidebar Navigation
st.sidebar.title("🚀 Dashboard Navigation")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Main Page", "📊 Statistical Analysis"]
)

if page == "🏠 Main Page":
    main_page()
else:
    statistical_page()
