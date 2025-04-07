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

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(r"../data/preprocessed_genes/all_genes/brain_count_overlap_hvg_labeled.csv")
    # data['involve_cancer'] = data['involve_cancer'].astype(str)  # Ensure consistent labels
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
    plt.legend(title="Cancer Status", labels=["Non-Cancerous", "Cancerous"])
    st.pyplot(plt)


# Main Page
def main_page():
    st.title("🧬 Cancer Prediction Dashboard")
    st.markdown("This dashboard visualizes the **gene expression data** for cancer prediction.")

    # Overview Section
    col1, col2 = st.columns([1, 3])
    with col1:
        st.info(f"**Dataset Shape:** {df.shape}")
    with col2:
        st.write(df.head())

    # Cancer Status Distribution Section
    st.markdown("## Cancer Status Distribution")
    fig = px.histogram(
        df, 
        x='involve_cancer', 
        color='involve_cancer', 
        color_discrete_map={'True': '#FF4B4B', 'False': '#4CAF50'},
        title='Cancer Status Distribution'
    )
    fig.update_layout(
        xaxis_title="Cancer Status",
        yaxis_title="Count",
        legend_title="Cancer Status",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("ℹ️ View Detailed Description"):
        st.markdown("""
        ### Description
        The bar chart titled **"Cancer Status Distribution"** visualizes the distribution of samples based on their cancer status. There are two categories:
        - **Category 0 (Non-Cancerous)**: Represented by a lighter red bar, this category has a count of **9,703** samples. It indicates the number of samples that are not associated with cancer.
        - **Category 1 (Cancerous)**: Represented by a darker red bar, this category has a count of **12,560** samples. It indicates the number of samples that are associated with cancer.
        The chart shows that there are more cancerous samples (**12.56K**) compared to non-cancerous samples (**9.703K**) in the dataset. This distribution is important for understanding the balance of classes in the dataset, which can impact the analysis and modeling processes.
        """)
    st.markdown("---")

    # Cancer Histogram Section
    st.markdown("## 🩺 Cancer Histogram")
    plot_gene_expression_histogram(df)
    
    with st.expander("ℹ️ View Detailed Description"):
        st.markdown("""
        ### Description
        The **"Cancer Histogram"** visualizes the distribution of total gene expression across samples, categorized by cancer status.

        - **Cancerous Samples**: Shown in orange, these samples have a distinct distribution pattern, indicating higher gene expression levels in some cases.

        - **Non-Cancerous Samples**: Shown in blue, these samples generally exhibit lower gene expression levels compared to cancerous samples.

        ### Conclusion
        The histogram reveals that cancerous samples tend to have higher gene expression levels, suggesting potential biomarkers for cancer detection. This difference in distribution highlights the importance of gene expression analysis in understanding cancer biology and developing diagnostic tools.
        """)
    st.markdown("---")

    # Gene Distribution Section
    st.markdown("## 📈 Gene Distributions")
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
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # Correlation Heatmap
    st.markdown("## 🌡️ Correlation Heatmap")
    top_corr_genes = df.select_dtypes(include=[np.number]).corr()['involve_cancer'].abs().nlargest(21).index
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[top_corr_genes].corr(), cmap='coolwarm', center=0, annot=False, cbar_kws={'shrink': 0.8})
    plt.title("Correlation Heatmap (Top 20 Genes)", fontsize=16)
    st.pyplot(fig)
    with st.expander("ℹ️ View Detailed Description"):
        st.markdown("""
        ### Description
        The **"Correlation Heatmap"** displays the correlation between the top 20 genes and their relationship with cancer involvement.

        - **Color Scale**: The heatmap uses a color scale from blue (negative correlation) to red (positive correlation). Darker colors indicate stronger correlations.

        - **Diagonal**: The diagonal represents perfect correlation (1.0) of each gene with itself.

        ### Conclusion
        This heatmap helps identify genes that have strong correlations with cancer involvement. Genes with high positive or negative correlations may be significant in understanding cancer mechanisms and could serve as potential targets for further research.
        """)
    st.markdown("---")

    # Outlier Analysis Section
    st.markdown("## 📦 Outlier Analysis")
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_gene = st.selectbox("📋 Select a Gene for Boxplot", df.select_dtypes(include=[np.number]).columns[:-1])
    with col2:
        fig = px.box(df, x='involve_cancer', y=selected_gene, color='involve_cancer')
        fig.update_layout(
            title=f"Boxplot for {selected_gene}",
            xaxis_title="Cancer Status",
            yaxis_title="Expression Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # Volcano Plot
    st.markdown("## 🌋 Volcano Plot - Gene Expression Analysis")
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

    fig.add_trace(go.Scatter(
        x=non_significant_genes['log2_fold_change'],
        y=non_significant_genes['log_p_value'],
        mode='markers',
        marker=dict(color='gray', size=6),
        name='Non-significant',
        text=non_significant_genes['Gene'],  # Add gene names for hover text
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=significant_genes['log2_fold_change'],
        y=significant_genes['log_p_value'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Significant Genes',
        text=significant_genes['Gene'],  # Add gene names for hover text
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Volcano Plot - Gene Expression Analysis',
        xaxis_title='Log2 Fold Change',
        yaxis_title='-Log10 p-value',
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    with st.expander("ℹ️ View Detailed Description"):
        st.markdown("""
        ### Description
        The **"Volcano Plot"** visualizes the differential expression of genes, highlighting both the magnitude and significance of changes.

        - **X-axis (Log2 Fold Change)**: Represents the magnitude of change in gene expression. Points to the left indicate downregulation, while points to the right indicate upregulation.

        - **Y-axis (-Log10 P-value)**: Represents the statistical significance of the change. Higher values indicate greater significance.

        - **Color Coding**: 
        - **Red Dots**: Significant genes with both high fold change and high significance.
        - **Gray Dots**: Non-significant genes.

        ### Conclusion
        The volcano plot helps identify genes that are both significantly and substantially differentially expressed. These genes are potential candidates for further investigation, as they may play crucial roles in the biological processes being studied.
        """)
    st.markdown("---")

    # Clustering Visualizations Section
    st.markdown("## 🧩 Clustering Visualizations")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]))

    df['PCA_1'] = pca_result[:, 0]
    df['PCA_2'] = pca_result[:, 1]

    col1, col2 = st.columns([1, 3])
    with col1:
        cluster_option = st.selectbox(
            "Choose a Clustering Technique:",
            ["PCA", "t-SNE", "UMAP", "KMeans"]
        )
        
        if cluster_option == "KMeans":
            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
    
    with col2:
        if cluster_option == "UMAP":
            st.plotly_chart(plot_umap(df.select_dtypes(include=[np.number])), use_container_width=True)
        elif cluster_option == "t-SNE":
            st.plotly_chart(plot_tsne(df.select_dtypes(include=[np.number])), use_container_width=True)
        elif cluster_option == "PCA":
            st.plotly_chart(plot_pca(df.select_dtypes(include=[np.number])), use_container_width=True)
        elif cluster_option == "KMeans":
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
