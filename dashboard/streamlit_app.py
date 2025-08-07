"""
Interactive Supply Chain Optimization Dashboard
Production-ready Streamlit application for business stakeholders
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.sql_analytics import SupplyChainAnalytics
from ml_models.demand_forecasting import DemandForecastingModel
from ml_models.supplier_optimization import SupplierOptimizationModel

# Page configuration
st.set_page_config(
    page_title="Supply Chain Intelligence Platform",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .alert-critical {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background-color: #fffbeb;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .alert-success {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_analytics_data():
    """Load and cache analytics data"""
    try:
        analytics = SupplyChainAnalytics()
        return analytics.run_all_analytics()
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
        return None

@st.cache_data
def load_raw_data():
    """Load raw data files"""
    try:
        data = {}
        files = ['suppliers', 'products', 'facilities', 'demand_forecast', 
                'supplier_performance', 'logistics']
        
        for file in files:
            try:
                data[file] = pd.read_csv(f'data/{file}.csv')
            except FileNotFoundError:
                st.warning(f"Data file {file}.csv not found. Please generate data first.")
                data[file] = pd.DataFrame()
        
        return data
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        return {}

def create_kpi_cards(dashboard_data):
    """Create KPI metric cards"""
    if 'kpi_summary' not in dashboard_data:
        return
    
    kpis = dashboard_data['kpi_summary']
    
    cols = st.columns(len(kpis))
    
    for i, (_, row) in enumerate(kpis.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{row['value']}</div>
                <div class="metric-label">{row['metric']}</div>
            </div>
            """, unsafe_allow_html=True)

def create_supplier_performance_dashboard(analytics_data):
    """Create supplier performance visualizations"""
    st.markdown('<div class="section-header">🏭 Supplier Performance Analysis</div>', 
                unsafe_allow_html=True)
    
    if 'supplier_performance' not in analytics_data:
        st.error("Supplier performance data not available")
        return
    
    supplier_data = analytics_data['supplier_performance']
    
    # Performance distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_perf = px.histogram(
            supplier_data, 
            x='performance_score',
            nbins=20,
            title='Supplier Performance Score Distribution',
            color_discrete_sequence=['#3b82f6']
        )
        fig_perf.update_layout(
            xaxis_title='Performance Score',
            yaxis_title='Number of Suppliers',
            showlegend=False
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        # Supplier categories pie chart
        category_counts = supplier_data['supplier_category'].value_counts()
        fig_cat = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Supplier Categories Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    # Regional performance comparison
    regional_perf = supplier_data.groupby('region').agg({
        'performance_score': 'mean',
        'quality_score': 'mean',
        'delivery_score': 'mean',
        'supplier_id': 'count'
    }).round(3).reset_index()
    
    fig_regional = px.bar(
        regional_perf,
        x='region',
        y='performance_score',
        title='Average Performance Score by Region',
        color='performance_score',
        color_continuous_scale='Blues'
    )
    fig_regional.update_layout(xaxis_title='Region', yaxis_title='Average Performance Score')
    st.plotly_chart(fig_regional, use_container_width=True)
    
    # Top and bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Suppliers")
        top_suppliers = supplier_data.nlargest(10, 'performance_score')[
            ['name', 'region', 'tier', 'performance_score', 'supplier_category']
        ]
        st.dataframe(top_suppliers, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Suppliers Needing Attention")
        bottom_suppliers = supplier_data.nsmallest(10, 'performance_score')[
            ['name', 'region', 'tier', 'performance_score', 'supplier_category']
        ]
        st.dataframe(bottom_suppliers, use_container_width=True)

def create_risk_assessment_dashboard(analytics_data):
    """Create risk assessment visualizations"""
    st.markdown('<div class="section-header">⚠️ Supply Chain Risk Assessment</div>', 
                unsafe_allow_html=True)
    
    if 'risk_assessment' not in analytics_data:
        st.error("Risk assessment data not available")
        return
    
    risk_data = analytics_data['risk_assessment']
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = risk_data['risk_category'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Risk Category Distribution',
            color_discrete_map={
                'Critical Risk': '#dc2626',
                'High Risk': '#ea580c',
                'Medium Risk': '#d97706',
                'Low Risk': '#65a30d',
                'Minimal Risk': '#16a34a'
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Business impact vs risk score scatter
        fig_scatter = px.scatter(
            risk_data,
            x='adjusted_risk_score',
            y='business_impact',
            color='risk_category',
            size='total_volume',
            hover_data=['name', 'region'],
            title='Risk vs Business Impact Analysis',
            color_discrete_map={
                'Critical Risk': '#dc2626',
                'High Risk': '#ea580c',
                'Medium Risk': '#d97706',
                'Low Risk': '#65a30d',
                'Minimal Risk': '#16a34a'
            }
        )
        fig_scatter.update_layout(
            xaxis_title='Risk Score',
            yaxis_title='Business Impact Score'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Critical risk alerts
    critical_risks = risk_data[risk_data['risk_category'] == 'Critical Risk']
    if not critical_risks.empty:
        st.markdown("""
        <div class="alert-critical">
            <h4>🚨 Critical Risk Alerts</h4>
            <p>The following suppliers require immediate attention:</p>
        </div>
        """, unsafe_allow_html=True)
        
        for _, supplier in critical_risks.head(5).iterrows():
            st.markdown(f"""
            **{supplier['name']}** ({supplier['region']})
            - Risk Score: {supplier['adjusted_risk_score']:.3f}
            - Business Impact: {supplier['business_impact']:.3f}
            - Action: {supplier['risk_mitigation_action']}
            """)
    
    # Risk by region
    regional_risk = risk_data.groupby('region').agg({
        'adjusted_risk_score': 'mean',
        'business_impact': 'mean',
        'supplier_id': 'count'
    }).round(3).reset_index()
    
    fig_regional_risk = px.bar(
        regional_risk,
        x='region',
        y='adjusted_risk_score',
        title='Average Risk Score by Region',
        color='adjusted_risk_score',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_regional_risk, use_container_width=True)

def create_demand_forecasting_dashboard(analytics_data):
    """Create demand forecasting analysis dashboard"""
    st.markdown('<div class="section-header">📈 Demand Forecasting Analysis</div>', 
                unsafe_allow_html=True)
    
    if 'demand_accuracy' not in analytics_data:
        st.error("Demand accuracy data not available")
        return
    
    demand_data = analytics_data['demand_accuracy']
    
    # Forecast accuracy by category
    col1, col2 = st.columns(2)
    
    with col1:
        fig_accuracy = px.bar(
            demand_data.groupby('category')['forecast_accuracy'].mean().reset_index(),
            x='category',
            y='forecast_accuracy',
            title='Forecast Accuracy by Product Category',
            color='forecast_accuracy',
            color_continuous_scale='Blues'
        )
        fig_accuracy.update_layout(
            xaxis_title='Product Category',
            yaxis_title='Average Forecast Accuracy'
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col2:
        # Accuracy grades distribution
        grade_counts = demand_data['accuracy_grade'].value_counts()
        fig_grades = px.pie(
            values=grade_counts.values,
            names=grade_counts.index,
            title='Forecast Accuracy Grades',
            color_discrete_map={
                'Excellent': '#16a34a',
                'Good': '#65a30d',
                'Fair': '#d97706',
                'Needs Improvement': '#dc2626'
            }
        )
        st.plotly_chart(fig_grades, use_container_width=True)
    
    # Seasonal accuracy comparison
    seasonal_data = demand_data[['category', 'seasonal_accuracy_q4', 'seasonal_accuracy_other']].melt(
        id_vars=['category'],
        var_name='season',
        value_name='accuracy'
    )
    seasonal_data['season'] = seasonal_data['season'].map({
        'seasonal_accuracy_q4': 'Q4 (Holiday Season)',
        'seasonal_accuracy_other': 'Other Quarters'
    })
    
    fig_seasonal = px.box(
        seasonal_data,
        x='category',
        y='accuracy',
        color='season',
        title='Seasonal Forecast Accuracy Comparison'
    )
    fig_seasonal.update_layout(
        xaxis_title='Product Category',
        yaxis_title='Forecast Accuracy',
        xaxis_tickangle=45
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Improvement opportunities
    st.subheader("🎯 Forecast Improvement Opportunities")
    improvement_opps = demand_data.nlargest(10, 'improvement_opportunity')[
        ['category', 'region', 'facility_type', 'forecast_accuracy', 
         'improvement_opportunity', 'accuracy_grade']
    ]
    st.dataframe(improvement_opps, use_container_width=True)

def create_logistics_optimization_dashboard(analytics_data):
    """Create logistics optimization dashboard"""
    st.markdown('<div class="section-header">🚚 Logistics Optimization</div>', 
                unsafe_allow_html=True)
    
    if 'logistics_optimization' not in analytics_data:
        st.error("Logistics optimization data not available")
        return
    
    logistics_data = analytics_data['logistics_optimization']
    
    # Transport mode performance
    col1, col2 = st.columns(2)
    
    with col1:
        mode_performance = logistics_data.groupby('transport_mode').agg({
            'optimization_score': 'mean',
            'on_time_percentage': 'mean',
            'cost_efficiency': 'mean'
        }).round(2).reset_index()
        
        fig_modes = px.bar(
            mode_performance,
            x='transport_mode',
            y='optimization_score',
            title='Transport Mode Optimization Scores',
            color='optimization_score',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_modes, use_container_width=True)
    
    with col2:
        # Route grades distribution
        grade_counts = logistics_data['route_grade'].value_counts()
        fig_grades = px.pie(
            values=grade_counts.values,
            names=grade_counts.index,
            title='Route Performance Grades',
            color_discrete_map={
                'Optimal': '#16a34a',
                'Good': '#65a30d',
                'Fair': '#d97706',
                'Needs Optimization': '#dc2626'
            }
        )
        st.plotly_chart(fig_grades, use_container_width=True)
    
    # Cost vs time efficiency scatter
    fig_efficiency = px.scatter(
        logistics_data,
        x='cost_efficiency',
        y='avg_transit_days',
        color='route_grade',
        size='shipment_count',
        hover_data=['transport_mode', 'origin_region', 'destination_region'],
        title='Cost Efficiency vs Transit Time Analysis',
        color_discrete_map={
            'Optimal': '#16a34a',
            'Good': '#65a30d',
            'Fair': '#d97706',
            'Needs Optimization': '#dc2626'
        }
    )
    fig_efficiency.update_layout(
        xaxis_title='Cost per KG',
        yaxis_title='Average Transit Days'
    )
    st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Optimization recommendations
    st.subheader("🔧 Optimization Recommendations")
    needs_optimization = logistics_data[
        logistics_data['route_grade'] == 'Needs Optimization'
    ].head(10)
    
    if not needs_optimization.empty:
        for _, route in needs_optimization.iterrows():
            st.markdown(f"""
            **{route['transport_mode']}**: {route['origin_region']} → {route['destination_region']}
            - Current Score: {route['optimization_score']:.2f}
            - Recommendation: {route['optimization_recommendation']}
            - Shipments: {route['shipment_count']}
            """)
    else:
        st.success("All routes are performing optimally!")

def main():
    """Main dashboard application"""
    st.markdown('<div class="main-header">🔗 Supply Chain Intelligence Platform</div>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Dashboard",
        ["Executive Overview", "Supplier Performance", "Risk Assessment", 
         "Demand Forecasting", "Logistics Optimization", "Raw Data Explorer"]
    )
    
    # Load data
    with st.spinner("Loading analytics data..."):
        analytics_data = load_analytics_data()
        raw_data = load_raw_data()
    
    if analytics_data is None:
        st.error("Failed to load analytics data. Please ensure the database is set up correctly.")
        return
    
    # Page routing
    if page == "Executive Overview":
        st.markdown("### 📊 Executive Dashboard")
        
        # KPI Cards
        if 'dashboard_data' in analytics_data:
            create_kpi_cards(analytics_data['dashboard_data'])
        
        # Quick insights
        col1, col2 = st.columns(2)
        
        with col1:
            if 'supplier_performance' in analytics_data:
                supplier_data = analytics_data['supplier_performance']
                strategic_partners = len(supplier_data[supplier_data['supplier_category'] == 'Strategic Partner'])
                st.metric("Strategic Partners", strategic_partners)
                
                avg_performance = supplier_data['performance_score'].mean()
                st.metric("Avg Supplier Performance", f"{avg_performance:.3f}")
        
        with col2:
            if 'risk_assessment' in analytics_data:
                risk_data = analytics_data['risk_assessment']
                critical_risks = len(risk_data[risk_data['risk_category'] == 'Critical Risk'])
                st.metric("Critical Risk Suppliers", critical_risks)
                
                if 'demand_accuracy' in analytics_data:
                    avg_accuracy = analytics_data['demand_accuracy']['forecast_accuracy'].mean()
                    st.metric("Forecast Accuracy", f"{avg_accuracy:.3f}")
        
        # Monthly trends
        if 'dashboard_data' in analytics_data and 'monthly_trends' in analytics_data['dashboard_data']:
            trends = analytics_data['dashboard_data']['monthly_trends']
            if not trends.empty:
                fig_trends = px.line(
                    trends,
                    x='month',
                    y='avg_accuracy',
                    title='Monthly Forecast Accuracy Trend'
                )
                st.plotly_chart(fig_trends, use_container_width=True)
    
    elif page == "Supplier Performance":
        create_supplier_performance_dashboard(analytics_data)
    
    elif page == "Risk Assessment":
        create_risk_assessment_dashboard(analytics_data)
    
    elif page == "Demand Forecasting":
        create_demand_forecasting_dashboard(analytics_data)
    
    elif page == "Logistics Optimization":
        create_logistics_optimization_dashboard(analytics_data)
    
    elif page == "Raw Data Explorer":
        st.markdown("### 🔍 Raw Data Explorer")
        
        data_type = st.selectbox(
            "Select Data Type",
            list(raw_data.keys())
        )
        
        if data_type in raw_data and not raw_data[data_type].empty:
            st.subheader(f"{data_type.title()} Data")
            st.dataframe(raw_data[data_type], use_container_width=True)
            
            # Basic statistics
            if st.checkbox("Show Statistics"):
                st.subheader("Data Statistics")
                st.write(raw_data[data_type].describe())
        else:
            st.warning(f"No data available for {data_type}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>
        Supply Chain Intelligence Platform | Built for Apple ADSP Portfolio
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()