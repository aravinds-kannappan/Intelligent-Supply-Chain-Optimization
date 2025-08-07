"""
Supplier optimization using clustering and multi-criteria decision analysis
Demonstrates advanced ML techniques for supplier selection and risk assessment
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplierOptimizationModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.risk_model = None
        self.supplier_network = None
        self.optimization_results = {}
        
    def prepare_supplier_data(self, suppliers_df, performance_df):
        """Prepare comprehensive supplier dataset"""
        # Aggregate performance metrics
        perf_agg = performance_df.groupby('supplier_id').agg({
            'quality_score': ['mean', 'std', 'count'],
            'delivery_performance': ['mean', 'std'],
            'cost_performance': ['mean', 'std'],
            'responsiveness': ['mean', 'std'],
            'innovation_score': ['mean', 'std'],
            'defect_rate': ['mean', 'max'],
            'on_time_delivery_rate': ['mean', 'std'],
            'cost_savings_achieved': ['sum', 'mean'],
            'sustainability_compliance': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        perf_agg.columns = ['_'.join(col).strip() for col in perf_agg.columns]
        perf_agg = perf_agg.reset_index()
        
        # Merge with supplier data
        supplier_data = suppliers_df.merge(perf_agg, on='supplier_id', how='left')
        
        # Fill missing performance data with supplier baseline scores
        performance_cols = [col for col in perf_agg.columns if col != 'supplier_id']
        for col in performance_cols:
            if 'quality' in col and supplier_data[col].isna().any():
                supplier_data[col] = supplier_data[col].fillna(supplier_data['quality_score'])
            elif 'delivery' in col and supplier_data[col].isna().any():
                supplier_data[col] = supplier_data[col].fillna(supplier_data['delivery_score'])
        
        # Create composite scores
        supplier_data['overall_performance'] = (
            supplier_data['quality_score_mean'].fillna(supplier_data['quality_score']) * 0.3 +
            supplier_data['delivery_performance_mean'].fillna(supplier_data['delivery_score']) * 0.25 +
            supplier_data['cost_performance_mean'].fillna(0.75) * 0.2 +
            supplier_data['responsiveness_mean'].fillna(0.8) * 0.15 +
            supplier_data['sustainability_compliance_mean'].fillna(supplier_data['sustainability_score']) * 0.1
        )
        
        # Risk-adjusted performance
        supplier_data['risk_adjusted_performance'] = (
            supplier_data['overall_performance'] * (1 - supplier_data['risk_score'])
        )
        
        # Capacity utilization efficiency
        supplier_data['capacity_efficiency'] = np.where(
            supplier_data['capacity'] > 0,
            supplier_data['quality_score_count'].fillna(1) / supplier_data['capacity'] * 10000,
            0
        )
        
        return supplier_data
    
    def perform_supplier_clustering(self, supplier_data):
        """Cluster suppliers based on performance and characteristics"""
        logger.info("Performing supplier clustering analysis...")
        
        # Select features for clustering
        clustering_features = [
            'quality_score', 'delivery_score', 'cost_competitiveness',
            'risk_score', 'sustainability_score', 'financial_stability',
            'lead_time_days', 'capacity', 'overall_performance'
        ]
        
        # Prepare data
        X = supplier_data[clustering_features].fillna(supplier_data[clustering_features].mean())
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method and silhouette score
        silhouette_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Select optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Final clustering
        self.clustering_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        supplier_data['cluster'] = self.clustering_model.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_analysis = supplier_data.groupby('cluster').agg({
            'quality_score': 'mean',
            'delivery_score': 'mean',
            'cost_competitiveness': 'mean',
            'risk_score': 'mean',
            'sustainability_score': 'mean',
            'overall_performance': 'mean',
            'supplier_id': 'count'
        }).round(3)
        
        cluster_analysis.columns = ['avg_quality', 'avg_delivery', 'avg_cost_comp', 
                                   'avg_risk', 'avg_sustainability', 'avg_performance', 'count']
        
        # Label clusters based on performance
        cluster_analysis['cluster_type'] = cluster_analysis.apply(self._label_cluster, axis=1)
        
        logger.info("Cluster Analysis:")
        logger.info(cluster_analysis)
        
        return supplier_data, cluster_analysis
    
    def _label_cluster(self, row):
        """Label clusters based on performance characteristics"""
        if row['avg_performance'] >= 0.8 and row['avg_risk'] <= 0.3:
            return 'Strategic Partners'
        elif row['avg_performance'] >= 0.7 and row['avg_risk'] <= 0.4:
            return 'Preferred Suppliers'
        elif row['avg_performance'] >= 0.6:
            return 'Qualified Suppliers'
        elif row['avg_risk'] >= 0.6:
            return 'High Risk Suppliers'
        else:
            return 'Development Required'
    
    def build_supplier_network(self, supplier_data, logistics_df):
        """Build supplier network graph for relationship analysis"""
        logger.info("Building supplier network graph...")
        
        self.supplier_network = nx.Graph()
        
        # Add supplier nodes with attributes
        for _, supplier in supplier_data.iterrows():
            self.supplier_network.add_node(
                supplier['supplier_id'],
                region=supplier['region'],
                tier=supplier['tier'],
                performance=supplier['overall_performance'],
                risk=supplier['risk_score'],
                cluster=supplier['cluster']
            )
        
        # Add edges based on logistics relationships and geographical proximity
        logistics_connections = logistics_df.groupby(['supplier_id']).agg({
            'destination_facility_id': lambda x: list(set(x)),
            'transport_cost': 'mean',
            'actual_transit_days': 'mean'
        }).reset_index()
        
        # Connect suppliers that serve similar facilities
        for i, supplier1 in logistics_connections.iterrows():
            for j, supplier2 in logistics_connections.iterrows():
                if i < j:  # Avoid duplicate edges
                    common_facilities = set(supplier1['destination_facility_id']) & set(supplier2['destination_facility_id'])
                    if len(common_facilities) > 0:
                        # Edge weight based on similarity
                        weight = len(common_facilities) / max(len(supplier1['destination_facility_id']), 
                                                            len(supplier2['destination_facility_id']))
                        self.supplier_network.add_edge(
                            supplier1['supplier_id'],
                            supplier2['supplier_id'],
                            weight=weight,
                            common_facilities=len(common_facilities)
                        )
        
        # Calculate network metrics
        network_metrics = {
            'nodes': self.supplier_network.number_of_nodes(),
            'edges': self.supplier_network.number_of_edges(),
            'density': nx.density(self.supplier_network),
            'avg_clustering': nx.average_clustering(self.supplier_network),
            'connected_components': nx.number_connected_components(self.supplier_network)
        }
        
        logger.info(f"Network metrics: {network_metrics}")
        return network_metrics
    
    def detect_supply_risk_anomalies(self, supplier_data):
        """Detect suppliers with anomalous risk patterns"""
        logger.info("Detecting supply risk anomalies...")
        
        # Features for anomaly detection
        risk_features = [
            'risk_score', 'quality_score', 'delivery_score', 'financial_stability',
            'defect_rate_mean', 'on_time_delivery_rate_mean', 'lead_time_days'
        ]
        
        X_risk = supplier_data[risk_features].fillna(supplier_data[risk_features].mean())
        X_risk_scaled = StandardScaler().fit_transform(X_risk)
        
        # Isolation Forest for anomaly detection
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        supplier_data['anomaly_score'] = isolation_forest.fit_predict(X_risk_scaled)
        supplier_data['anomaly_score_prob'] = isolation_forest.score_samples(X_risk_scaled)
        
        # Identify high-risk anomalies
        anomalies = supplier_data[supplier_data['anomaly_score'] == -1].copy()
        anomalies = anomalies.sort_values('anomaly_score_prob')
        
        logger.info(f"Detected {len(anomalies)} anomalous suppliers")
        
        return anomalies[['supplier_id', 'name', 'region', 'tier', 'risk_score', 
                         'overall_performance', 'anomaly_score_prob']]
    
    def optimize_supplier_portfolio(self, supplier_data, demand_requirements):
        """Multi-objective optimization for supplier portfolio"""
        logger.info("Optimizing supplier portfolio...")
        
        # Define optimization objectives
        def objective_function(weights, suppliers, requirements):
            """
            Multi-objective function:
            - Maximize performance
            - Minimize risk
            - Minimize cost
            - Ensure capacity coverage
            """
            selected_suppliers = suppliers[weights > 0.1]  # Threshold for selection
            
            if len(selected_suppliers) == 0:
                return float('inf')
            
            # Performance score (maximize)
            performance_score = np.sum(weights * suppliers['overall_performance'])
            
            # Risk score (minimize)
            risk_score = np.sum(weights * suppliers['risk_score'])
            
            # Cost score (minimize - using inverse of cost competitiveness)
            cost_score = np.sum(weights * (1 - suppliers['cost_competitiveness']))
            
            # Capacity coverage
            total_capacity = np.sum(weights * suppliers['capacity'])
            capacity_penalty = max(0, requirements['total_demand'] - total_capacity) / requirements['total_demand']
            
            # Diversification bonus (prefer multiple suppliers)
            diversification_bonus = min(len(selected_suppliers) / 10, 0.1)
            
            # Combined objective (minimize)
            objective = (
                -performance_score * 0.4 +  # Maximize performance
                risk_score * 0.3 +          # Minimize risk
                cost_score * 0.2 +          # Minimize cost
                capacity_penalty * 0.1 -    # Ensure capacity
                diversification_bonus       # Encourage diversification
            )
            
            return objective
        
        # Prepare optimization
        n_suppliers = len(supplier_data)
        initial_weights = np.ones(n_suppliers) / n_suppliers
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: np.sum(w * supplier_data['capacity']) - demand_requirements['total_demand']}  # Capacity constraint
        ]
        
        bounds = [(0, 1) for _ in range(n_suppliers)]
        
        # Optimize
        result = minimize(
            objective_function,
            initial_weights,
            args=(supplier_data, demand_requirements),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract results
        optimal_weights = result.x
        selected_suppliers = supplier_data[optimal_weights > 0.1].copy()
        selected_suppliers['allocation_weight'] = optimal_weights[optimal_weights > 0.1]
        
        self.optimization_results = {
            'selected_suppliers': selected_suppliers,
            'optimization_success': result.success,
            'objective_value': result.fun,
            'total_suppliers_selected': len(selected_suppliers)
        }
        
        logger.info(f"Portfolio optimization completed. Selected {len(selected_suppliers)} suppliers")
        
        return self.optimization_results
    
    def generate_supplier_recommendations(self, supplier_data, cluster_analysis):
        """Generate actionable supplier recommendations"""
        recommendations = []
        
        # Strategic recommendations by cluster
        for cluster_id, cluster_info in cluster_analysis.iterrows():
            cluster_suppliers = supplier_data[supplier_data['cluster'] == cluster_id]
            
            if cluster_info['cluster_type'] == 'Strategic Partners':
                recommendations.append({
                    'cluster': cluster_id,
                    'type': 'Strategic Partnership',
                    'action': 'Expand collaboration and long-term contracts',
                    'suppliers': len(cluster_suppliers),
                    'priority': 'High'
                })
            elif cluster_info['cluster_type'] == 'High Risk Suppliers':
                recommendations.append({
                    'cluster': cluster_id,
                    'type': 'Risk Mitigation',
                    'action': 'Implement risk monitoring and backup suppliers',
                    'suppliers': len(cluster_suppliers),
                    'priority': 'Critical'
                })
            elif cluster_info['cluster_type'] == 'Development Required':
                recommendations.append({
                    'cluster': cluster_id,
                    'type': 'Supplier Development',
                    'action': 'Provide training and support programs',
                    'suppliers': len(cluster_suppliers),
                    'priority': 'Medium'
                })
        
        # Individual supplier recommendations
        top_performers = supplier_data.nlargest(5, 'overall_performance')
        high_risk = supplier_data.nlargest(5, 'risk_score')
        
        for _, supplier in top_performers.iterrows():
            recommendations.append({
                'supplier_id': supplier['supplier_id'],
                'supplier_name': supplier['name'],
                'type': 'Top Performer',
                'action': 'Consider for strategic partnership expansion',
                'performance': supplier['overall_performance'],
                'priority': 'High'
            })
        
        for _, supplier in high_risk.iterrows():
            recommendations.append({
                'supplier_id': supplier['supplier_id'],
                'supplier_name': supplier['name'],
                'type': 'High Risk Alert',
                'action': 'Immediate risk assessment and mitigation plan',
                'risk_score': supplier['risk_score'],
                'priority': 'Critical'
            })
        
        return recommendations
    
    def save_model(self, filepath):
        """Save optimization models and results"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'scaler': self.scaler,
            'clustering_model': self.clustering_model,
            'risk_model': self.risk_model,
            'optimization_results': self.optimization_results
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Supplier optimization model saved to {filepath}")

def run_supplier_optimization():
    """Main function to run supplier optimization analysis"""
    # Load data
    suppliers_df = pd.read_csv('data/suppliers.csv')
    performance_df = pd.read_csv('data/supplier_performance.csv')
    logistics_df = pd.read_csv('data/logistics.csv')
    demand_df = pd.read_csv('data/demand_forecast.csv')
    
    # Initialize model
    model = SupplierOptimizationModel()
    
    # Prepare data
    supplier_data = model.prepare_supplier_data(suppliers_df, performance_df)
    
    # Perform clustering
    supplier_data, cluster_analysis = model.perform_supplier_clustering(supplier_data)
    
    # Build network
    network_metrics = model.build_supplier_network(supplier_data, logistics_df)
    
    # Detect anomalies
    anomalies = model.detect_supply_risk_anomalies(supplier_data)
    
    # Portfolio optimization
    total_demand = demand_df['actual_demand'].sum()
    demand_requirements = {'total_demand': total_demand}
    optimization_results = model.optimize_supplier_portfolio(supplier_data, demand_requirements)
    
    # Generate recommendations
    recommendations = model.generate_supplier_recommendations(supplier_data, cluster_analysis)
    
    # Save model
    import os
    os.makedirs('models', exist_ok=True)
    model.save_model('models/supplier_optimization_model.pkl')
    
    return {
        'supplier_data': supplier_data,
        'cluster_analysis': cluster_analysis,
        'network_metrics': network_metrics,
        'anomalies': anomalies,
        'optimization_results': optimization_results,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = run_supplier_optimization()
    print("Supplier optimization analysis completed!")
    print(f"Identified {len(results['anomalies'])} high-risk suppliers")
    print(f"Generated {len(results['recommendations'])} recommendations")