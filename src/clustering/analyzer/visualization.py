from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

class ClusterVisualizer:
    """
    A class to handle all visualization aspects of the clustering analysis.
    
    Attributes:
        color_sequence (List[str]): Color sequence for plots
        template (str): Plotly template to use
    """
    
    def __init__(
        self,
        color_sequence: Optional[List[str]] = None,
        template: str = "plotly_white"
    ):
        """
        Initialize the ClusterVisualizer.
        
        Args:
            color_sequence: Custom color sequence for plots
            template: Plotly template name
        """
        self.color_sequence = color_sequence or px.colors.qualitative.Set3
        self.template = template
    
    def plot_cluster_distribution(
        self,
        clusters: np.ndarray,
        title: str = "Cluster Size Distribution"
    ) -> go.Figure:
        """
        Create a bar plot of cluster sizes.
        
        Args:
            clusters: Array of cluster assignments
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=cluster_sizes.index,
                y=cluster_sizes.values,
                text=cluster_sizes.values,
                textposition='auto',
                marker_color=self.color_sequence[:len(cluster_sizes)]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Cluster",
            yaxis_title="Number of Employees",
            template=self.template
        )
        
        return fig
    
    def plot_dimension_reduction(
        self,
        features: np.ndarray,
        clusters: np.ndarray,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create a scatter plot using PCA for dimensionality reduction."""
        if len(features) != len(clusters):
            raise ValueError("Features and clusters must have the same length")
        
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(features)
        
        n_clusters = len(np.unique(clusters))
        cluster_colors = {str(i): self.color_sequence[i % len(self.color_sequence)] 
                        for i in range(n_clusters)}
        colors = [cluster_colors[str(c)] for c in clusters]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=dict(
                color=colors,
                size=8
            ),
            text=[f'Cluster {c}' for c in clusters],
            hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ))
        
        explained_var = pca.explained_variance_ratio_ * 100
        fig.update_layout(
            title=title or "Cluster Visualization (PCA)",
            xaxis_title=f'PC1 ({explained_var[0]:.1f}% variance)',
            yaxis_title=f'PC2 ({explained_var[1]:.1f}% variance)',
            template=self.template,
            showlegend=False
        )
        
        return fig
    
    def plot_feature_importance(
        self,
        importance_by_cluster: Dict[int, List[Tuple[str, float]]],
        top_n: int = 5
    ) -> go.Figure:
        """
        Create a heatmap of feature importance by cluster.
        
        Args:
            importance_by_cluster: Dictionary of feature importance by cluster
            top_n: Number of top features to show
            
        Returns:
            Plotly figure object
        """
        features = set()
        for cluster_features in importance_by_cluster.values():
            features.update(feature for feature, _ in cluster_features[:top_n])
        
        feature_list = sorted(features)
        cluster_ids = sorted(importance_by_cluster.keys())
        
        heatmap_data = np.zeros((len(feature_list), len(cluster_ids)))
        for i, feature in enumerate(feature_list):
            for j, cluster_id in enumerate(cluster_ids):
                cluster_features = dict(importance_by_cluster[cluster_id])
                heatmap_data[i, j] = cluster_features.get(feature, 0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f'Cluster {i}' for i in cluster_ids],
            y=feature_list,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Feature Importance by Cluster",
            xaxis_title="Cluster",
            yaxis_title="Feature",
            template=self.template
        )
        
        return fig
    
    def plot_risk_profiles(
        self,
        risk_scores: pd.DataFrame,
        clusters: np.ndarray
    ) -> Tuple[go.Figure, go.Figure]:
        """
        Create risk profile visualizations.
        
        Args:
            risk_scores: DataFrame containing risk scores
            clusters: Cluster assignments
            
        Returns:
            Tuple of (radar plot, box plot) figures
        """
        risk_components = [col for col in risk_scores.columns 
                         if col.endswith('_risk') and col != 'overall_risk']
        
        radar_fig = go.Figure()
        for cluster_id in np.unique(clusters):
            cluster_means = risk_scores[risk_scores['cluster'] == cluster_id][risk_components].mean()
            
            radar_fig.add_trace(go.Scatterpolar(
                r=cluster_means.values,
                theta=risk_components,
                name=f'Cluster {cluster_id}',
                fill='toself'
            ))
        
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            title="Risk Components by Cluster",
            showlegend=True,
            template=self.template
        )
        
        box_fig = go.Figure()
        for cluster_id in np.unique(clusters):
            cluster_risks = risk_scores[risk_scores['cluster'] == cluster_id]['overall_risk']
            
            box_fig.add_trace(go.Box(
                y=cluster_risks,
                name=f'Cluster {cluster_id}',
                boxpoints='outliers'
            ))
        
        box_fig.update_layout(
            title="Overall Risk Distribution by Cluster",
            yaxis_title="Risk Score",
            showlegend=True,
            template=self.template
        )
        
        return radar_fig, box_fig
    
    def plot_demographic_distribution(
        self,
        employee_data: pd.DataFrame,
        clusters: np.ndarray,
        demographic_cols: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create stacked bar plots of demographic distributions by cluster.
        
        Args:
            employee_data: DataFrame containing employee information
            clusters: Cluster assignments
            demographic_cols: List of demographic columns to plot
            
        Returns:
            Plotly figure object
        """
        demographic_cols = demographic_cols or [
            'department_name', 'generation', 'gender', 'company_tenure'
        ]
        
        fig = make_subplots(
            rows=len(demographic_cols),
            cols=1,
            subplot_titles=[col.replace('_', ' ').title() for col in demographic_cols],
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(demographic_cols, 1):
            if col not in employee_data.columns:
                continue
                
            dist = pd.crosstab(
                clusters,
                employee_data[col],
                normalize='index'
            )
            
            for category in dist.columns:
                fig.add_trace(
                    go.Bar(
                        name=category,
                        x=dist.index,
                        y=dist[category],
                        text=np.round(dist[category], 2),
                        textposition='auto'
                    ),
                    row=i,
                    col=1
                )
        
        fig.update_layout(
            height=300 * len(demographic_cols),
            title="Demographic Distribution by Cluster",
            barmode='stack',
            showlegend=True,
            template=self.template
        )
        
        return fig
    
    def create_dashboard(
        self,
        features: np.ndarray,
        clusters: np.ndarray,
        risk_scores: pd.DataFrame,
        employee_data: pd.DataFrame,
        importance_by_cluster: Dict[int, List[Tuple[str, float]]]
    ) -> bool:
        """
        Create and display a comprehensive dashboard of visualizations.
        
        Args:
            features: Feature array
            clusters: Cluster assignments
            risk_scores: DataFrame containing risk scores
            employee_data: DataFrame containing employee information
            importance_by_cluster: Dictionary of feature importance by cluster
            
        Returns:
            bool: True if dashboard creation was successful, False otherwise
        """
        try:
            print("Creating visualization dashboard...")
            
            self.plot_cluster_distribution(clusters).show()
            self.plot_dimension_reduction(features, clusters).show()
            self.plot_feature_importance(importance_by_cluster).show()
            
            radar_fig, box_fig = self.plot_risk_profiles(risk_scores, clusters)
            radar_fig.show()
            box_fig.show()
            
            self.plot_demographic_distribution(employee_data, clusters).show()
            
            return True
            
        except Exception as e:
            print(f"Dashboard creation failed: {str(e)}")
            return False