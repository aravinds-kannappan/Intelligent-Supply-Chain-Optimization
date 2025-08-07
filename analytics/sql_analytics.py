import pandas as pd
import sqlite3
from config.database_config import db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplyChainAnalytics:
    def __init__(self):
        self.db = db_manager
    
    def supplier_performance_analysis(self):
        """Comprehensive supplier performance analysis using advanced SQL"""
        query = """
        WITH supplier_metrics AS (
            SELECT 
                s.supplier_id,
                s.name,
                s.region,
                s.tier,
                s.risk_score,
                COUNT(DISTINCT sp.performance_id) as evaluation_count,
                AVG(sp.quality_score) as avg_quality,
                AVG(sp.delivery_performance) as avg_delivery,
                AVG(sp.cost_performance) as avg_cost,
                AVG(sp.defect_rate) as avg_defect_rate,
                AVG(sp.on_time_delivery_rate) as avg_otd_rate,
                SUM(sp.cost_savings_achieved) as total_cost_savings,
                STDDEV(sp.quality_score) as quality_volatility,
                -- Performance trend (last 6 months vs previous 6 months)
                AVG(CASE WHEN sp.evaluation_date >= date('now', '-6 months') 
                         THEN sp.quality_score END) as recent_quality,
                AVG(CASE WHEN sp.evaluation_date < date('now', '-6 months') 
                         AND sp.evaluation_date >= date('now', '-12 months')
                         THEN sp.quality_score END) as previous_quality
            FROM suppliers s
            LEFT JOIN supplier_performance sp ON s.supplier_id = sp.supplier_id
            GROUP BY s.supplier_id, s.name, s.region, s.tier, s.risk_score
        ),
        performance_rankings AS (
            SELECT *,
                -- Composite performance score
                (avg_quality * 0.3 + avg_delivery * 0.25 + avg_cost * 0.2 + 
                 avg_otd_rate * 0.15 + (1 - risk_score) * 0.1) as composite_score,
                -- Performance improvement
                (recent_quality - previous_quality) as quality_improvement,
                -- Rank suppliers within their tier
                ROW_NUMBER() OVER (PARTITION BY tier ORDER BY 
                    (avg_quality * 0.3 + avg_delivery * 0.25 + avg_cost * 0.2 + 
                     avg_otd_rate * 0.15 + (1 - risk_score) * 0.1) DESC) as tier_rank
            FROM supplier_metrics
            WHERE evaluation_count > 0
        )
        SELECT 
            supplier_id,
            name,
            region,
            tier,
            tier_rank,
            ROUND(composite_score, 3) as performance_score,
            ROUND(avg_quality, 3) as quality_score,
            ROUND(avg_delivery, 3) as delivery_score,
            ROUND(avg_defect_rate, 4) as defect_rate,
            ROUND(avg_otd_rate, 3) as on_time_delivery,
            total_cost_savings,
            ROUND(quality_improvement, 3) as quality_trend,
            evaluation_count,
            CASE 
                WHEN composite_score >= 0.85 AND tier_rank <= 3 THEN 'Strategic Partner'
                WHEN composite_score >= 0.75 THEN 'Preferred Supplier'
                WHEN composite_score >= 0.65 THEN 'Qualified Supplier'
                WHEN risk_score > 0.6 THEN 'High Risk'
                ELSE 'Development Required'
            END as supplier_category
        FROM performance_rankings
        ORDER BY composite_score DESC;
        """
        
        return pd.read_sql_query(query, self.db.engine)
    
    def demand_forecast_accuracy_analysis(self):
        """Analyze demand forecasting accuracy across products and regions"""
        query = """
        WITH forecast_metrics AS (
            SELECT 
                p.category,
                f.region,
                f.type as facility_type,
                COUNT(*) as forecast_count,
                AVG(df.forecast_accuracy) as avg_accuracy,
                AVG(ABS(df.forecasted_demand - df.actual_demand)) as mae,
                AVG(df.forecasted_demand) as avg_forecasted,
                AVG(df.actual_demand) as avg_actual,
                STDDEV(df.actual_demand) as demand_volatility,
                -- Bias calculation
                AVG(df.forecasted_demand - df.actual_demand) as avg_bias,
                -- Seasonal accuracy (Q4 vs rest of year)
                AVG(CASE WHEN strftime('%m', df.date) IN ('10', '11', '12') 
                         THEN df.forecast_accuracy END) as q4_accuracy,
                AVG(CASE WHEN strftime('%m', df.date) NOT IN ('10', '11', '12') 
                         THEN df.forecast_accuracy END) as non_q4_accuracy
            FROM demand_forecast df
            JOIN products p ON df.product_id = p.product_id
            JOIN facilities f ON df.facility_id = f.facility_id
            WHERE df.actual_demand IS NOT NULL
            GROUP BY p.category, f.region, f.type
        ),
        accuracy_rankings AS (
            SELECT *,
                -- Accuracy improvement opportunity
                (1 - avg_accuracy) * avg_actual as improvement_potential,
                -- Rank by accuracy within category
                ROW_NUMBER() OVER (PARTITION BY category ORDER BY avg_accuracy DESC) as accuracy_rank
            FROM forecast_metrics
        )
        SELECT 
            category,
            region,
            facility_type,
            forecast_count,
            ROUND(avg_accuracy, 3) as forecast_accuracy,
            ROUND(mae, 0) as mean_absolute_error,
            ROUND(avg_bias, 0) as forecast_bias,
            ROUND(demand_volatility, 0) as demand_volatility,
            ROUND(q4_accuracy, 3) as seasonal_accuracy_q4,
            ROUND(non_q4_accuracy, 3) as seasonal_accuracy_other,
            ROUND(improvement_potential, 0) as improvement_opportunity,
            accuracy_rank,
            CASE 
                WHEN avg_accuracy >= 0.9 THEN 'Excellent'
                WHEN avg_accuracy >= 0.8 THEN 'Good'
                WHEN avg_accuracy >= 0.7 THEN 'Fair'
                ELSE 'Needs Improvement'
            END as accuracy_grade
        FROM accuracy_rankings
        ORDER BY category, avg_accuracy DESC;
        """
        
        return pd.read_sql_query(query, self.db.engine)
    
    def logistics_optimization_analysis(self):
        """Advanced logistics performance and optimization analysis"""
        query = """
        WITH logistics_metrics AS (
            SELECT 
                l.transport_mode,
                s.region as origin_region,
                f.region as destination_region,
                COUNT(*) as shipment_count,
                AVG(l.transport_cost) as avg_cost,
                AVG(l.actual_transit_days) as avg_transit_time,
                AVG(l.carbon_footprint_kg) as avg_carbon_footprint,
                SUM(l.quantity) as total_quantity,
                SUM(l.weight_kg) as total_weight,
                -- Performance metrics
                COUNT(CASE WHEN l.delivery_status = 'On Time' THEN 1 END) * 100.0 / COUNT(*) as on_time_rate,
                COUNT(CASE WHEN l.delivery_status = 'Delayed' THEN 1 END) * 100.0 / COUNT(*) as delay_rate,
                COUNT(CASE WHEN l.delivery_status IN ('Damaged', 'Lost') THEN 1 END) * 100.0 / COUNT(*) as damage_rate,
                -- Cost efficiency
                AVG(l.transport_cost / l.weight_kg) as cost_per_kg,
                AVG(l.carbon_footprint_kg / l.weight_kg) as carbon_per_kg,
                -- Time efficiency
                AVG(l.actual_transit_days / l.estimated_transit_days) as time_accuracy
            FROM logistics l
            JOIN suppliers s ON l.supplier_id = s.supplier_id
            JOIN facilities f ON l.destination_facility_id = f.facility_id
            GROUP BY l.transport_mode, s.region, f.region
            HAVING shipment_count >= 10  -- Filter for statistical significance
        ),
        route_optimization AS (
            SELECT *,
                -- Efficiency scores (lower is better for cost and time, higher for reliability)
                NTILE(5) OVER (ORDER BY cost_per_kg) as cost_efficiency_quintile,
                NTILE(5) OVER (ORDER BY avg_transit_time) as time_efficiency_quintile,
                NTILE(5) OVER (ORDER BY on_time_rate DESC) as reliability_quintile,
                NTILE(5) OVER (ORDER BY carbon_per_kg) as sustainability_quintile,
                -- Overall route score
                (5 - NTILE(5) OVER (ORDER BY cost_per_kg)) * 0.3 +
                (5 - NTILE(5) OVER (ORDER BY avg_transit_time)) * 0.25 +
                NTILE(5) OVER (ORDER BY on_time_rate DESC) * 0.25 +
                (5 - NTILE(5) OVER (ORDER BY carbon_per_kg)) * 0.2 as route_score
            FROM logistics_metrics
        )
        SELECT 
            transport_mode,
            origin_region,
            destination_region,
            shipment_count,
            ROUND(avg_cost, 0) as avg_transport_cost,
            ROUND(avg_transit_time, 1) as avg_transit_days,
            ROUND(on_time_rate, 1) as on_time_percentage,
            ROUND(cost_per_kg, 2) as cost_efficiency,
            ROUND(carbon_per_kg, 3) as carbon_efficiency,
            ROUND(route_score, 2) as optimization_score,
            CASE 
                WHEN route_score >= 4.0 THEN 'Optimal'
                WHEN route_score >= 3.0 THEN 'Good'
                WHEN route_score >= 2.0 THEN 'Fair'
                ELSE 'Needs Optimization'
            END as route_grade,
            -- Recommendations
            CASE 
                WHEN cost_efficiency_quintile = 5 THEN 'High Cost - Consider Alternative Routes'
                WHEN time_efficiency_quintile = 5 THEN 'Slow Transit - Evaluate Faster Options'
                WHEN reliability_quintile = 1 THEN 'Poor Reliability - Investigate Delays'
                WHEN sustainability_quintile = 5 THEN 'High Carbon - Consider Green Alternatives'
                ELSE 'Performance Acceptable'
            END as optimization_recommendation
        FROM route_optimization
        ORDER BY route_score DESC;
        """
        
        return pd.read_sql_query(query, self.db.engine)
    
    def supply_chain_risk_assessment(self):
        """Comprehensive supply chain risk analysis"""
        query = """
        WITH supplier_risk_profile AS (
            SELECT 
                s.supplier_id,
                s.name,
                s.region,
                s.tier,
                s.risk_score as base_risk,
                s.financial_stability,
                COUNT(DISTINCT l.destination_facility_id) as facility_coverage,
                COUNT(DISTINCT p.category) as product_diversity,
                SUM(l.quantity) as total_volume,
                -- Performance-based risk adjustments
                AVG(sp.quality_score) as avg_quality,
                AVG(sp.delivery_performance) as avg_delivery,
                STDDEV(sp.quality_score) as quality_volatility,
                -- Recent performance trend
                AVG(CASE WHEN sp.evaluation_date >= date('now', '-3 months') 
                         THEN sp.quality_score END) as recent_quality,
                -- Concentration risk
                MAX(l.quantity) * 100.0 / NULLIF(SUM(l.quantity), 0) as max_shipment_concentration
            FROM suppliers s
            LEFT JOIN logistics l ON s.supplier_id = l.supplier_id
            LEFT JOIN supplier_performance sp ON s.supplier_id = sp.supplier_id
            LEFT JOIN products p ON l.product_id = p.product_id
            GROUP BY s.supplier_id, s.name, s.region, s.tier, s.risk_score, s.financial_stability
        ),
        risk_calculations AS (
            SELECT *,
                -- Adjusted risk score based on performance and diversification
                CASE 
                    WHEN avg_quality IS NULL THEN base_risk + 0.2  -- Penalty for no performance data
                    ELSE base_risk + 
                         (1 - COALESCE(avg_quality, 0.5)) * 0.3 +  -- Quality risk
                         (1 - COALESCE(avg_delivery, 0.5)) * 0.2 +  -- Delivery risk
                         COALESCE(quality_volatility, 0.2) * 0.1 +   -- Volatility risk
                         (1 - COALESCE(financial_stability, 0.5)) * 0.2  -- Financial risk
                END as adjusted_risk_score,
                -- Concentration risk
                CASE 
                    WHEN facility_coverage <= 2 THEN 0.3  -- High concentration
                    WHEN facility_coverage <= 5 THEN 0.1  -- Medium concentration
                    ELSE 0.0  -- Well diversified
                END as concentration_risk,
                -- Volume impact (higher volume = higher impact if disrupted)
                NTILE(5) OVER (ORDER BY total_volume DESC) as volume_impact_quintile
            FROM supplier_risk_profile
        ),
        final_risk_assessment AS (
            SELECT *,
                -- Final risk score combining all factors
                LEAST(1.0, adjusted_risk_score + concentration_risk) as final_risk_score,
                -- Risk category
                CASE 
                    WHEN LEAST(1.0, adjusted_risk_score + concentration_risk) >= 0.8 THEN 'Critical Risk'
                    WHEN LEAST(1.0, adjusted_risk_score + concentration_risk) >= 0.6 THEN 'High Risk'
                    WHEN LEAST(1.0, adjusted_risk_score + concentration_risk) >= 0.4 THEN 'Medium Risk'
                    WHEN LEAST(1.0, adjusted_risk_score + concentration_risk) >= 0.2 THEN 'Low Risk'
                    ELSE 'Minimal Risk'
                END as risk_category,
                -- Business impact
                volume_impact_quintile * LEAST(1.0, adjusted_risk_score + concentration_risk) as business_impact_score
            FROM risk_calculations
        )
        SELECT 
            supplier_id,
            name,
            region,
            tier,
            ROUND(base_risk, 3) as base_risk_score,
            ROUND(final_risk_score, 3) as adjusted_risk_score,
            risk_category,
            ROUND(business_impact_score, 3) as business_impact,
            facility_coverage,
            product_diversity,
            total_volume,
            ROUND(COALESCE(avg_quality, 0), 3) as quality_performance,
            ROUND(COALESCE(recent_quality, 0), 3) as recent_quality_trend,
            ROUND(COALESCE(max_shipment_concentration, 0), 1) as shipment_concentration_pct,
            -- Risk mitigation recommendations
            CASE 
                WHEN final_risk_score >= 0.8 THEN 'Immediate Action Required - Find Backup Suppliers'
                WHEN final_risk_score >= 0.6 THEN 'Monitor Closely - Develop Contingency Plans'
                WHEN concentration_risk > 0.2 THEN 'Diversify Supplier Base'
                WHEN quality_volatility > 0.2 THEN 'Implement Quality Improvement Program'
                ELSE 'Continue Regular Monitoring'
            END as risk_mitigation_action
        FROM final_risk_assessment
        ORDER BY business_impact_score DESC, final_risk_score DESC;
        """
        
        return pd.read_sql_query(query, self.db.engine)
    
    def facility_utilization_optimization(self):
        """Analyze facility utilization and optimization opportunities"""
        query = """
        WITH facility_metrics AS (
            SELECT 
                f.facility_id,
                f.name,
                f.type,
                f.region,
                f.capacity,
                f.utilization_rate,
                f.operational_cost_per_unit,
                f.energy_efficiency,
                f.automation_level,
                -- Demand served by facility
                COUNT(DISTINCT df.product_id) as products_served,
                SUM(df.actual_demand) as total_demand_served,
                AVG(df.actual_demand) as avg_demand_per_product,
                -- Logistics connections
                COUNT(DISTINCT l.supplier_id) as supplier_connections,
                SUM(l.quantity) as total_inbound_quantity,
                AVG(l.transport_cost) as avg_inbound_cost,
                -- Utilization calculations
                (SUM(df.actual_demand) * 1.0 / NULLIF(f.capacity, 0)) as demand_utilization,
                f.utilization_rate as reported_utilization
            FROM facilities f
            LEFT JOIN demand_forecast df ON f.facility_id = df.facility_id
            LEFT JOIN logistics l ON f.facility_id = l.destination_facility_id
            GROUP BY f.facility_id, f.name, f.type, f.region, f.capacity, 
                     f.utilization_rate, f.operational_cost_per_unit, 
                     f.energy_efficiency, f.automation_level
        ),
        optimization_analysis AS (
            SELECT *,
                -- Efficiency metrics
                total_demand_served / NULLIF(operational_cost_per_unit * capacity, 0) as cost_efficiency,
                total_demand_served * energy_efficiency as energy_productivity,
                -- Optimization opportunities
                CASE 
                    WHEN reported_utilization < 0.6 THEN capacity * (0.8 - reported_utilization)
                    ELSE 0
                END as capacity_expansion_opportunity,
                CASE 
                    WHEN reported_utilization > 0.9 THEN total_demand_served * 0.2
                    ELSE 0
                END as capacity_constraint_risk,
                -- Performance scores
                NTILE(5) OVER (ORDER BY cost_efficiency DESC) as cost_efficiency_quintile,
                NTILE(5) OVER (ORDER BY energy_productivity DESC) as energy_efficiency_quintile,
                NTILE(5) OVER (ORDER BY automation_level DESC) as automation_quintile
            FROM facility_metrics
            WHERE total_demand_served > 0  -- Only facilities with actual demand
        )
        SELECT 
            facility_id,
            name,
            type,
            region,
            capacity,
            ROUND(reported_utilization, 3) as current_utilization,
            ROUND(demand_utilization, 3) as demand_based_utilization,
            total_demand_served,
            products_served,
            supplier_connections,
            ROUND(cost_efficiency, 4) as cost_efficiency_score,
            ROUND(energy_productivity, 0) as energy_productivity_score,
            ROUND(capacity_expansion_opportunity, 0) as expansion_opportunity,
            ROUND(capacity_constraint_risk, 0) as constraint_risk,
            -- Overall facility grade
            CASE 
                WHEN (cost_efficiency_quintile + energy_efficiency_quintile + automation_quintile) >= 12 THEN 'A'
                WHEN (cost_efficiency_quintile + energy_efficiency_quintile + automation_quintile) >= 9 THEN 'B'
                WHEN (cost_efficiency_quintile + energy_efficiency_quintile + automation_quintile) >= 6 THEN 'C'
                ELSE 'D'
            END as facility_grade,
            -- Optimization recommendations
            CASE 
                WHEN reported_utilization < 0.5 THEN 'Underutilized - Consider Consolidation'
                WHEN reported_utilization > 0.95 THEN 'Over-capacity - Expand or Redistribute'
                WHEN energy_efficiency < 0.6 THEN 'Energy Efficiency Improvement Needed'
                WHEN automation_level < 0.4 THEN 'Automation Upgrade Opportunity'
                WHEN cost_efficiency_quintile <= 2 THEN 'Cost Optimization Required'
                ELSE 'Performance Acceptable'
            END as optimization_recommendation
        FROM optimization_analysis
        ORDER BY 
            CASE type 
                WHEN 'Manufacturing' THEN 1 
                WHEN 'Assembly' THEN 2 
                WHEN 'Distribution' THEN 3 
                ELSE 4 
            END,
            cost_efficiency_score DESC;
        """
        
        return pd.read_sql_query(query, self.db.engine)
    
    def generate_executive_dashboard_data(self):
        """Generate key metrics for executive dashboard"""
        queries = {
            'kpi_summary': """
                SELECT 
                    'Total Suppliers' as metric,
                    COUNT(*) as value,
                    'Active suppliers in network' as description
                FROM suppliers
                UNION ALL
                SELECT 
                    'Average Supplier Performance',
                    ROUND(AVG(
                        (quality_score + delivery_score + cost_competitiveness + 
                         sustainability_score + financial_stability) / 5
                    ), 3),
                    'Composite performance score'
                FROM suppliers
                UNION ALL
                SELECT 
                    'High Risk Suppliers',
                    COUNT(*),
                    'Suppliers with risk score > 0.6'
                FROM suppliers WHERE risk_score > 0.6
                UNION ALL
                SELECT 
                    'Forecast Accuracy',
                    ROUND(AVG(forecast_accuracy), 3),
                    'Average demand forecast accuracy'
                FROM demand_forecast WHERE actual_demand IS NOT NULL
                UNION ALL
                SELECT 
                    'On-Time Delivery Rate',
                    ROUND(COUNT(CASE WHEN delivery_status = 'On Time' THEN 1 END) * 100.0 / COUNT(*), 1),
                    'Percentage of on-time deliveries'
                FROM logistics
            """,
            
            'regional_performance': """
                SELECT 
                    region,
                    COUNT(DISTINCT s.supplier_id) as supplier_count,
                    ROUND(AVG(s.quality_score), 3) as avg_quality,
                    ROUND(AVG(s.risk_score), 3) as avg_risk,
                    SUM(l.quantity) as total_volume,
                    ROUND(AVG(l.transport_cost), 0) as avg_transport_cost
                FROM suppliers s
                LEFT JOIN logistics l ON s.supplier_id = l.supplier_id
                GROUP BY region
                ORDER BY supplier_count DESC
            """,
            
            'monthly_trends': """
                SELECT 
                    strftime('%Y-%m', date) as month,
                    AVG(forecast_accuracy) as avg_accuracy,
                    SUM(actual_demand) as total_demand,
                    COUNT(*) as forecast_count
                FROM demand_forecast 
                WHERE actual_demand IS NOT NULL
                GROUP BY strftime('%Y-%m', date)
                ORDER BY month
            """
        }
        
        dashboard_data = {}
        for key, query in queries.items():
            dashboard_data[key] = pd.read_sql_query(query, self.db.engine)
        
        return dashboard_data
    
    def run_all_analytics(self):
        """Run all analytics and return comprehensive results"""
        logger.info("Running comprehensive supply chain analytics...")
        
        results = {
            'supplier_performance': self.supplier_performance_analysis(),
            'demand_accuracy': self.demand_forecast_accuracy_analysis(),
            'logistics_optimization': self.logistics_optimization_analysis(),
            'risk_assessment': self.supply_chain_risk_assessment(),
            'facility_optimization': self.facility_utilization_optimization(),
            'dashboard_data': self.generate_executive_dashboard_data()
        }
        
        logger.info("Analytics completed successfully")
        return results

if __name__ == "__main__":
    analytics = SupplyChainAnalytics()
    results = analytics.run_all_analytics()
    
    # Display sample results
    print("=== SUPPLIER PERFORMANCE ANALYSIS ===")
    print(results['supplier_performance'].head())
    
    print("\n=== SUPPLY CHAIN RISK ASSESSMENT ===")
    print(results['risk_assessment'].head())
    
    print("\n=== EXECUTIVE DASHBOARD KPIs ===")
    print(results['dashboard_data']['kpi_summary'])
