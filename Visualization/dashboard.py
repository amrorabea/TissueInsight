import streamlit as st # type: ignore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from scipy.stats import mannwhitneyu # type: ignore
import umap
from sklearn.decomposition import PCA
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import math 
from matplotlib import cm
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "brain_count_overlap_hvg_labeled.csv"))

    meta_path = os.path.join(os.path.dirname(__file__), "../data/meta_all_gene.csv")
    meta = pd.read_csv(meta_path)
    # data['involve_cancer'] = data['involve_cancer'].astype(str)  # Ensure consistent labels
    return data , meta

df, meta = load_data()

def plot_species_distribution(meta):
    # Ensure we don't modify the original DataFrame
    species_counts = meta['species'].value_counts().reset_index()
    species_counts.columns = ['Species', 'Count']

    # Create interactive pie chart
    fig = px.pie(
        species_counts,
        names="Species",
        values="Count",
        title="🧬 Distribution of Species in the Dataset",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    return fig


def plot_species_distribution_doughnut_plotly(meta, tissue="brain"):
    # Filter for selected tissue
    brain_meta = meta[meta['tissue'] == tissue]
    
    # Count species
    species_counts = brain_meta['species'].value_counts().reset_index()
    species_counts.columns = ['Species', 'Count']
    
    # Create the doughnut chart using Plotly Express
    fig = px.pie(
        species_counts,
        names='Species',
        values='Count',
        hole=0.4,  # Creates a doughnut chart
        title=f"🧠 Species Distribution in {tissue.title()} Tissue",
        color='Species',  # Add color differentiation based on species
        color_discrete_sequence=px.colors.qualitative.Set2  # Choose a nice color palette
    )

    # Add actual numbers at the top corner of the chart
    total_count = species_counts['Count'].sum()
    fig.add_annotation(
        x=0.5,  # Position of the annotation in the center of the plot
        y=1.1,  # Slightly above the chart
        text=f"Total Samples: {total_count}",
        showarrow=False,
        font=dict(size=16, color="black"),
        align="center",
    )

    # Update layout for better spacing and presentation
    fig.update_layout(
        title_x=0.5,  # Title in the center
        title_font_size=16,
        title_font_family="Arial",
    )

    return fig


def plot_tissue_distribution(meta):
    if 'tissue' not in meta.columns:
        raise ValueError("The 'tissue' column is missing.")
    
    tissue_counts = meta['tissue'].value_counts().sort_values(ascending=False)
    
    if tissue_counts.empty:
        raise ValueError("No tissue data found.")

    num_groups = 4
    group_size = math.ceil(len(tissue_counts) / num_groups)
    tissue_groups = [tissue_counts[i * group_size: (i + 1) * group_size] for i in range(num_groups)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, (ax, tissue_group) in enumerate(zip(axes, tissue_groups)):
        sns.barplot(x=tissue_group.index, y=tissue_group.values, ax=ax, palette="viridis")
        ax.set_title(f"Group {i+1}: Tissue Sample Count")
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Tissue Type")
        ax.set_xticklabels(tissue_group.index, rotation=45, ha="right")
        for bar, count in zip(ax.patches, tissue_group.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count}",
                ha='center', va='bottom', fontsize=10, color='black'
            )

    plt.tight_layout()
    return fig

def plot_tissue_distribution_circular(meta):
    # Count tissues
    top_n = 20  # or 15
    tissue_counts = meta['tissue'].value_counts().head(top_n).reset_index()

    # tissue_counts = meta['tissue'].value_counts().reset_index()
    tissue_counts.columns = ['Tissue', 'Count']

    num_bars = len(tissue_counts)
    angles = np.linspace(0, 2 * np.pi, num_bars, endpoint=False).tolist()
    tissue_counts["Angle"] = angles
    tissue_counts["LabelAngle"] = np.degrees(tissue_counts["Angle"])
    tissue_counts['Height'] = tissue_counts['Count'] / tissue_counts['Count'].max() * 100

    # Color palette
    color_palette = cm.get_cmap('tab20c', num_bars)
    bar_colors = [color_palette(i) for i in range(num_bars)]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    bars = ax.bar(
        tissue_counts["Angle"],
        tissue_counts["Height"],
        width=0.35,
        color=bar_colors,
        edgecolor='black'
    )

    # Add labels
    for i, row in tissue_counts.iterrows():
        rotation = row["LabelAngle"]
        alignment = 'left' if rotation < 180 else 'right'
        ax.text(
            row["Angle"],
            row["Height"] + 5,
            f'{row["Tissue"]} ({row["Count"]})',
            ha=alignment,
            va='center',
            fontsize=9,
            rotation=rotation if rotation < 180 else rotation - 180,
            rotation_mode='anchor'
        )

    ax.set_title(f"Tissue Sample Distribution for the first {top_n}", size=15)
    ax.set_yticklabels([])
    ax.set_xticks([])
    plt.tight_layout()
    return fig

def plot_cancer_percentage_by_tissue(meta):
    # Ensure 'involve_cancer' is string and normalized
    meta = meta.copy()  # To avoid modifying the original
    meta['involve_cancer'] = meta['involve_cancer'].astype(str).str.lower()

    # Mapping to readable labels
    cancer_mapping = {"true": "Cancerous", "false": "Non-Cancerous"}
    meta['involve_cancer'] = meta['involve_cancer'].map(cancer_mapping)
    meta = meta[meta['involve_cancer'].notna()]

    # Filter tissues with more than 10 samples
    tissue_counts = meta['tissue'].value_counts()
    valid_tissues = tissue_counts[tissue_counts > 10].index
    filtered_meta = meta[meta['tissue'].isin(valid_tissues)]

    # Group and calculate percentages
    cancer_counts = filtered_meta.groupby(['tissue', 'involve_cancer']).size().unstack(fill_value=0)
    cancer_percentages = cancer_counts.div(cancer_counts.sum(axis=1), axis=0) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = ["Non-Cancerous", "Cancerous"]
    colors = ["#1f77b4", "#d62728"]
    bottom = np.zeros(len(cancer_percentages))
    bars = []
    for label, color in zip(labels, colors):
        bars.append(ax.bar(cancer_percentages.index, cancer_percentages[label], label=label, color=color, bottom=bottom))
        bottom += cancer_percentages[label].values

    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f"{height:.1f}%", 
                        ha='center', va='center', fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel("Percentage of Samples")
    ax.set_xlabel("Tissue Type")
    ax.set_title("Percentage of Cancerous vs. Non-Cancerous Samples per Tissue")
    ax.legend(title="Involve Cancer")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig


def plot_cancerous_vs_non_cancerous(meta, tissue="brain"):
  
    # Filter the data for the selected tissue
    meta_filtered = meta[meta['tissue'] == tissue]
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create the countplot for cancerous vs non-cancerous samples
    sns.countplot(data=meta_filtered, x='species', hue='involve_cancer', palette=['blue', 'red'], ax=ax)

    # Add counts on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

    # Labels and title
    ax.set_title(f"Comparison of Cancerous vs Non-Cancerous Samples in {tissue.capitalize()} Tissue")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    ax.legend(labels=["Non-Cancerous", "Cancerous"])

    # Return the figure object for use with Streamlit
    return fig

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



def explore_data_page():
    st.title("🔬 Explore Data")
    st.markdown("Interactively explore gene expression features and cancer labels.")

def general_data_page():
    st.title("📊 General Data Analysis")
    st.markdown("Explore overall patterns in the gene expression dataset. This page provides insights into the distribution and activity of genes across tissues and cancer types.")
    
    # 🐭 Species Breakdown
    st.subheader("🐭 Species Representation")
    st.markdown("This chart shows how samples are distributed across different species.")
    st.plotly_chart(plot_species_distribution(meta), use_container_width=True)
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

    # 🌍 Tissue Distribution
    st.subheader("🌍 Tissue Sample Distribution")
    try:
        tissue_fig = plot_tissue_distribution(meta)  # Call your function here
        st.pyplot(tissue_fig)  # Display the plot in Streamlit
    except Exception as e:
        st.error(f"Error generating tissue distribution: {e}")
        
    try:
        tissue_fig2 = plot_tissue_distribution_circular(meta)  # Call your function here
        st.pyplot(tissue_fig2)  # Display the plot in Streamlit
    except Exception as e:
        st.error(f"Error generating tissue distribution: {e}")

    st.markdown("---")
    # 🧪 Cancer Composition by Tissue
    st.subheader("🧪 Cancerous vs Non-Cancerous Proportions by Tissue")
    st.markdown("This shows how different tissues are affected by cancer in terms of sample proportion.")
    fig = plot_cancer_percentage_by_tissue(meta)  # 👈 Pass your global `meta` here
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧬 Species Distribution in Brain Tissue ")
    st.markdown("This circular bar chart shows the breakdown of species within brain tissue samples.")
   
    # Call the doughnut chart function for brain tissue
    fig = plot_species_distribution_doughnut_plotly(meta, tissue="brain")
    st.plotly_chart(fig)  # Display Plotly chart in Streamlit

    # New Section: Cancerous vs Non-Cancerous Samples in Brain Tissue
    st.subheader("🧬 Cancerous vs Non-Cancerous Samples in Brain Tissue")
    st.markdown("This countplot compares the distribution of cancerous vs non-cancerous samples for brain tissue.")
    
    # Generate the plot using the function
    try:
        fig = plot_cancerous_vs_non_cancerous(meta, tissue="brain")
        st.pyplot(fig)  # Display the plot in Streamlit
    except Exception as e:
        st.error(f"Error generating cancerous vs non-cancerous distribution: {e}")
    
# Sidebar Navigation
st.sidebar.title("🚀 Dashboard Navigation")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Main Page", "🧪 General Analysis","🔬 Explore Data", "📊 Statistical Analysis"]
)

if page == "🏠 Main Page":
    main_page()

elif page == "🧪 General Analysis":
    general_data_page()
elif page == "🔬 Explore Data":
    explore_data_page()
else:
    statistical_page()