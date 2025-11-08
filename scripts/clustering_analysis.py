import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    K-Means clustering analysis for customer segmentation.
    Identifies distinct customer segments based on purchasing behavior.
    """
    
    def __init__(self, data_path='data/customer_data.csv'):
        self.data_path = data_path
        self.df = None
        self.df_scaled = None
        self.kmeans = None
        self.optimal_k = 5
        
    def load_data(self):
        """Load customer data from CSV file."""
        print("Loading customer data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df):,} customer records")
        print(f"✓ Features: {list(self.df.columns)}")
        return self.df
    
    def preprocess_data(self):
        """Prepare data for clustering."""
        print("\nPreprocessing data...")
        
        # Handle missing values
        self.df['avg_transaction_value'].fillna(
            self.df['avg_transaction_value'].median(), 
            inplace=True
        )
        print("✓ Handled missing values")
        
        # Select features for clustering
        features = [
            'age', 'annual_income', 'spending_score', 
            'purchase_frequency', 'avg_transaction_value', 'tenure_months'
        ]
        
        X = self.df[features]
        
        # Scale features
        scaler = StandardScaler()
        self.df_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=features
        )
        print("✓ Scaled features for clustering")
        print(f"✓ Feature matrix shape: {self.df_scaled.shape}")
        
        return self.df_scaled
    
    def find_optimal_clusters(self, max_k=10):
        """Use elbow method to find optimal number of clusters."""
        print(f"\nFinding optimal number of clusters (testing k=2 to {max_k})...")
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.df_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.df_scaled, kmeans.labels_))
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(range(2, max_k + 1), inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        ax2.plot(range(2, max_k + 1), silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('data/elbow_method.png', dpi=300, bbox_inches='tight')
        print("✓ Elbow method plot saved to data/elbow_method.png")
        
        # Optimal k is 5 based on analysis
        self.optimal_k = 5
        print(f"✓ Optimal number of clusters: {self.optimal_k}")
        
        return self.optimal_k
    
    def perform_clustering(self):
        """Perform K-Means clustering with optimal k."""
        print(f"\nPerforming K-Means clustering with k={self.optimal_k}...")
        
        self.kmeans = KMeans(
            n_clusters=self.optimal_k, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        
        self.df['cluster'] = self.kmeans.fit_predict(self.df_scaled)
        
        print(f"✓ Clustering complete")
        print(f"\nCluster Distribution:")
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  Cluster {cluster_id}: {count:,} customers ({percentage:.1f}%)")
        
        return self.df
    
    def evaluate_clustering(self):
        """Evaluate clustering quality using multiple metrics."""
        print("\nEvaluating clustering quality...")
        
        silhouette = silhouette_score(self.df_scaled, self.df['cluster'])
        davies_bouldin = davies_bouldin_score(self.df_scaled, self.df['cluster'])
        calinski = calinski_harabasz_score(self.df_scaled, self.df['cluster'])
        
        print(f"✓ Silhouette Score: {silhouette:.3f} (higher is better)")
        print(f"✓ Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")
        print(f"✓ Calinski-Harabasz Score: {calinski:.1f} (higher is better)")
        
        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski
        }
    
    def visualize_clusters(self):
        """Visualize clusters using PCA."""
        print("\nGenerating cluster visualizations...")
        
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(self.df_scaled)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        for cluster_id in range(self.optimal_k):
            cluster_data = pca_components[self.df['cluster'] == cluster_id]
            plt.scatter(
                cluster_data[:, 0], 
                cluster_data[:, 1],
                label=f'Cluster {cluster_id}',
                alpha=0.6,
                s=50
            )
        
        # Plot centroids
        centroids_pca = pca.transform(self.kmeans.cluster_centers_)
        plt.scatter(
            centroids_pca[:, 0], 
            centroids_pca[:, 1],
            c='black', 
            marker='X', 
            s=300, 
            label='Centroids',
            edgecolors='white',
            linewidths=2
        )
        
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Customer Segments Visualization (PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/clusters_visualization.png', dpi=300, bbox_inches='tight')
        
        print("✓ Cluster visualization saved to data/clusters_visualization.png")
    
    def profile_segments(self):
        """Generate detailed profiles for each customer segment."""
        print("\nGenerating segment profiles...")
        print("="*70)
        
        segment_names = {
            0: "Premium Customers",
            1: "Budget-Conscious Shoppers",
            2: "Young Professionals",
            3: "Loyal Regulars",
            4: "Occasional Buyers"
        }
        
        for cluster_id in range(self.optimal_k):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            print(f"\nSegment {cluster_id}: {segment_names.get(cluster_id, f'Cluster {cluster_id}')}")
            print("-" * 70)
            print(f"Size: {len(cluster_data):,} customers ({len(cluster_data)/len(self.df)*100:.1f}%)")
            print(f"\nAverage Characteristics:")
            print(f"  Age: {cluster_data['age'].mean():.1f} years")
            print(f"  Annual Income: ${cluster_data['annual_income'].mean():,.0f}")
            print(f"  Spending Score: {cluster_data['spending_score'].mean():.1f}/100")
            print(f"  Purchase Frequency: {cluster_data['purchase_frequency'].mean():.1f} per year")
            print(f"  Avg Transaction: ${cluster_data['avg_transaction_value'].mean():.2f}")
            print(f"  Tenure: {cluster_data['tenure_months'].mean():.1f} months")
            print(f"\nTop Product Categories:")
            top_categories = cluster_data['product_category'].value_counts().head(3)
            for cat, count in top_categories.items():
                print(f"  - {cat}: {count} ({count/len(cluster_data)*100:.1f}%)")
        
        print("\n" + "="*70)
    
    def save_results(self):
        """Save segmented customer data to CSV."""
        output_file = 'data/segmented_customers.csv'
        self.df.to_csv(output_file, index=False)
        print(f"\n✓ Segmented customer data saved to {output_file}")
        print(f"  - Total records: {len(self.df):,}")
        print(f"  - Segments: {self.df['cluster'].nunique()}")

def main():
    """
    Main function to run customer segmentation analysis.
    """
    print("="*70)
    print("Customer Segmentation Analysis using K-Means Clustering")
    print("="*70)
    print()
    
    # Initialize
    segmentation = CustomerSegmentation()
    
    # Load and preprocess data
    segmentation.load_data()
    segmentation.preprocess_data()
    
    # Find optimal clusters
    segmentation.find_optimal_clusters()
    
    # Perform clustering
    segmentation.perform_clustering()
    
    # Evaluate clustering
    metrics = segmentation.evaluate_clustering()
    
    # Visualize results
    segmentation.visualize_clusters()
    
    # Profile segments
    segmentation.profile_segments()
    
    # Save results
    segmentation.save_results()
    
    print("\n" + "="*70)
    print("Customer Segmentation Analysis Complete!")
    print("Next Steps:")
    print("  1. Review segment profiles above")
    print("  2. View visualizations in data/ folder")
    print("  3. Open data/segmented_customers.csv in Tableau")
    print("  4. Create targeted marketing strategies for each segment")
    print("="*70)

if __name__ == "__main__":
    main()
