"""
Advanced supply chain data generator with realistic business scenarios
"""
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import json
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

class SupplyChainDataGenerator:
    def __init__(self):
        self.suppliers = []
        self.products = []
        self.facilities = []
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
        # Apple-like product categories
        self.product_categories = {
            'iPhone': {'base_cost': 400, 'complexity': 0.9, 'demand_volatility': 0.3},
            'MacBook': {'base_cost': 800, 'complexity': 0.8, 'demand_volatility': 0.2},
            'iPad': {'base_cost': 300, 'complexity': 0.7, 'demand_volatility': 0.25},
            'Apple Watch': {'base_cost': 200, 'complexity': 0.6, 'demand_volatility': 0.4},
            'AirPods': {'base_cost': 100, 'complexity': 0.5, 'demand_volatility': 0.35},
            'Components': {'base_cost': 50, 'complexity': 0.4, 'demand_volatility': 0.5}
        }
        
        # Global regions with different risk profiles
        self.regions = {
            'North America': {'risk_score': 0.1, 'cost_multiplier': 1.2},
            'Europe': {'risk_score': 0.15, 'cost_multiplier': 1.1},
            'Asia Pacific': {'risk_score': 0.3, 'cost_multiplier': 0.8},
            'China': {'risk_score': 0.4, 'cost_multiplier': 0.7},
            'Southeast Asia': {'risk_score': 0.35, 'cost_multiplier': 0.75}
        }
    
    def generate_suppliers(self, n_suppliers=150):
        """Generate realistic supplier data"""
        suppliers = []
        
        for i in range(n_suppliers):
            region = random.choice(list(self.regions.keys()))
            region_data = self.regions[region]
            
            # Supplier performance varies by region and size
            tier = random.choices(['Tier 1', 'Tier 2', 'Tier 3'], weights=[0.2, 0.5, 0.3])[0]
            
            supplier = {
                'supplier_id': f'SUP_{i+1:04d}',
                'name': fake.company(),
                'region': region,
                'country': fake.country(),
                'tier': tier,
                'established_year': fake.random_int(min=1980, max=2020),
                'capacity': fake.random_int(min=1000, max=100000),
                'quality_score': max(0.5, np.random.normal(0.85, 0.1)),
                'delivery_score': max(0.5, np.random.normal(0.88, 0.08)),
                'cost_competitiveness': max(0.3, np.random.normal(0.75, 0.15)),
                'risk_score': region_data['risk_score'] + np.random.normal(0, 0.1),
                'sustainability_score': max(0.2, np.random.normal(0.7, 0.2)),
                'financial_stability': max(0.3, np.random.normal(0.8, 0.15)),
                'lead_time_days': fake.random_int(min=7, max=90),
                'minimum_order_qty': fake.random_int(min=100, max=10000),
                'certifications': random.sample(['ISO9001', 'ISO14001', 'OHSAS18001', 'TS16949'], 
                                              k=random.randint(1, 4))
            }
            suppliers.append(supplier)
        
        self.suppliers = suppliers
        return pd.DataFrame(suppliers)
    
    def generate_products(self, n_products=80):
        """Generate product catalog with realistic attributes"""
        products = []
        
        for i in range(n_products):
            category = random.choice(list(self.product_categories.keys()))
            category_data = self.product_categories[category]
            
            product = {
                'product_id': f'PROD_{i+1:04d}',
                'name': f"{category} {fake.word().title()} {fake.random_int(min=1, max=20)}",
                'category': category,
                'base_cost': category_data['base_cost'] * (1 + np.random.normal(0, 0.2)),
                'complexity_score': category_data['complexity'] + np.random.normal(0, 0.1),
                'weight_kg': fake.random_int(min=50, max=2000) / 1000,  # Convert to kg
                'dimensions_cm': f"{fake.random_int(min=5, max=50)}x{fake.random_int(min=5, max=50)}x{fake.random_int(min=1, max=10)}",
                'lifecycle_stage': random.choice(['Introduction', 'Growth', 'Maturity', 'Decline']),
                'environmental_impact': max(0.1, np.random.normal(0.5, 0.2)),
                'required_certifications': random.sample(['CE', 'FCC', 'RoHS', 'ENERGY_STAR'], 
                                                        k=random.randint(1, 3))
            }
            products.append(product)
        
        self.products = products
        return pd.DataFrame(products)
    
    def generate_facilities(self, n_facilities=25):
        """Generate manufacturing and distribution facilities"""
        facilities = []
        
        facility_types = ['Manufacturing', 'Distribution', 'R&D', 'Assembly']
        
        for i in range(n_facilities):
            region = random.choice(list(self.regions.keys()))
            facility_type = random.choice(facility_types)
            
            facility = {
                'facility_id': f'FAC_{i+1:04d}',
                'name': f"{fake.city()} {facility_type} Center",
                'type': facility_type,
                'region': region,
                'country': fake.country(),
                'latitude': fake.latitude(),
                'longitude': fake.longitude(),
                'capacity': fake.random_int(min=10000, max=500000),
                'utilization_rate': max(0.3, np.random.normal(0.75, 0.15)),
                'operational_cost_per_unit': fake.random_int(min=5, max=50),
                'energy_efficiency': max(0.3, np.random.normal(0.7, 0.2)),
                'automation_level': max(0.1, np.random.normal(0.6, 0.25)),
                'quality_certification': random.choice(['ISO9001', 'Six Sigma', 'Lean', 'TQM'])
            }
            facilities.append(facility)
        
        self.facilities = facilities
        return pd.DataFrame(facilities)
    
    def generate_demand_forecast(self, n_records=10000):
        """Generate realistic demand patterns with seasonality"""
        demand_data = []
        
        for _ in range(n_records):
            product = random.choice(self.products)
            facility = random.choice(self.facilities)
            
            # Generate date with realistic business patterns
            date = fake.date_between(start_date=self.start_date, end_date=self.end_date)
            
            # Add seasonality (higher demand in Q4 for consumer electronics)
            seasonal_multiplier = 1.0
            if date.month in [10, 11, 12]:  # Q4
                seasonal_multiplier = 1.4
            elif date.month in [1, 2]:  # Post-holiday dip
                seasonal_multiplier = 0.7
            
            # Base demand with product-specific volatility
            base_demand = fake.random_int(min=100, max=10000)
            volatility = self.product_categories[product['category']]['demand_volatility']
            actual_demand = max(0, base_demand * seasonal_multiplier * (1 + np.random.normal(0, volatility)))
            
            demand_record = {
                'demand_id': f'DEM_{len(demand_data)+1:06d}',
                'product_id': product['product_id'],
                'facility_id': facility['facility_id'],
                'date': date,
                'forecasted_demand': int(actual_demand * (1 + np.random.normal(0, 0.1))),
                'actual_demand': int(actual_demand),
                'forecast_accuracy': max(0.5, np.random.normal(0.85, 0.1)),
                'demand_driver': random.choice(['Seasonal', 'Promotional', 'New Launch', 'Regular', 'Emergency']),
                'confidence_interval': np.random.normal(0.15, 0.05)
            }
            demand_data.append(demand_record)
        
        return pd.DataFrame(demand_data)
    
    def generate_supplier_performance(self, n_records=5000):
        """Generate supplier performance metrics over time"""
        performance_data = []
        
        for _ in range(n_records):
            supplier = random.choice(self.suppliers)
            product = random.choice(self.products)
            
            # Performance varies by supplier tier and region
            base_performance = 0.8
            if supplier['tier'] == 'Tier 1':
                base_performance = 0.9
            elif supplier['tier'] == 'Tier 3':
                base_performance = 0.7
            
            performance_record = {
                'performance_id': f'PERF_{len(performance_data)+1:06d}',
                'supplier_id': supplier['supplier_id'],
                'product_id': product['product_id'],
                'evaluation_date': fake.date_between(start_date=self.start_date, end_date=self.end_date),
                'quality_score': max(0.3, base_performance + np.random.normal(0, 0.1)),
                'delivery_performance': max(0.3, base_performance + np.random.normal(0, 0.08)),
                'cost_performance': max(0.3, np.random.normal(0.75, 0.15)),
                'responsiveness': max(0.3, np.random.normal(0.8, 0.12)),
                'innovation_score': max(0.1, np.random.normal(0.6, 0.2)),
                'defect_rate': max(0.001, np.random.exponential(0.02)),
                'on_time_delivery_rate': max(0.5, np.random.normal(0.88, 0.1)),
                'cost_savings_achieved': fake.random_int(min=-5000, max=50000),
                'sustainability_compliance': max(0.3, np.random.normal(0.75, 0.15))
            }
            performance_data.append(performance_record)
        
        return pd.DataFrame(performance_data)
    
    def generate_logistics_data(self, n_shipments=8000):
        """Generate logistics and transportation data"""
        logistics_data = []
        
        transport_modes = ['Air', 'Sea', 'Road', 'Rail', 'Multimodal']
        
        for _ in range(n_shipments):
            supplier = random.choice(self.suppliers)
            facility = random.choice(self.facilities)
            product = random.choice(self.products)
            
            transport_mode = random.choice(transport_modes)
            
            # Transport costs and times vary by mode and distance
            base_cost = fake.random_int(min=500, max=15000)
            base_time = fake.random_int(min=1, max=30)
            
            if transport_mode == 'Air':
                cost_multiplier = 3.0
                time_multiplier = 0.3
            elif transport_mode == 'Sea':
                cost_multiplier = 0.5
                time_multiplier = 2.0
            else:
                cost_multiplier = 1.0
                time_multiplier = 1.0
            
            shipment = {
                'shipment_id': f'SHIP_{len(logistics_data)+1:06d}',
                'supplier_id': supplier['supplier_id'],
                'destination_facility_id': facility['facility_id'],
                'product_id': product['product_id'],
                'shipment_date': fake.date_between(start_date=self.start_date, end_date=self.end_date),
                'transport_mode': transport_mode,
                'quantity': fake.random_int(min=100, max=5000),
                'weight_kg': fake.random_int(min=500, max=20000),
                'volume_m3': fake.random_int(min=10, max=200),
                'transport_cost': base_cost * cost_multiplier,
                'estimated_transit_days': int(base_time * time_multiplier),
                'actual_transit_days': int(base_time * time_multiplier * (1 + np.random.normal(0, 0.2))),
                'carbon_footprint_kg': fake.random_int(min=100, max=5000),
                'customs_clearance_time': fake.random_int(min=0, max=5),
                'insurance_cost': fake.random_int(min=50, max=1000),
                'delivery_status': random.choice(['On Time', 'Delayed', 'Early', 'Damaged', 'Lost'])
            }
            logistics_data.append(shipment)
        
        return pd.DataFrame(logistics_data)
    
    def generate_all_data(self):
        """Generate complete supply chain dataset"""
        print("Generating suppliers...")
        suppliers_df = self.generate_suppliers()
        
        print("Generating products...")
        products_df = self.generate_products()
        
        print("Generating facilities...")
        facilities_df = self.generate_facilities()
        
        print("Generating demand forecast...")
        demand_df = self.generate_demand_forecast()
        
        print("Generating supplier performance...")
        performance_df = self.generate_supplier_performance()
        
        print("Generating logistics data...")
        logistics_df = self.generate_logistics_data()
        
        return {
            'suppliers': suppliers_df,
            'products': products_df,
            'facilities': facilities_df,
            'demand_forecast': demand_df,
            'supplier_performance': performance_df,
            'logistics': logistics_df
        }

if __name__ == "__main__":
    generator = SupplyChainDataGenerator()
    data = generator.generate_all_data()
    
    # Save to CSV files
    os.makedirs('data', exist_ok=True)
    for table_name, df in data.items():
        df.to_csv(f'data/{table_name}.csv', index=False)
        print(f"Generated {len(df)} records for {table_name}")