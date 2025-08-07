-- Supply Chain Optimization Database Schema
-- Designed for high-performance analytics and machine learning

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Suppliers table with comprehensive attributes
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    country TEXT NOT NULL,
    tier TEXT CHECK (tier IN ('Tier 1', 'Tier 2', 'Tier 3')),
    established_year INTEGER,
    capacity INTEGER,
    quality_score REAL CHECK (quality_score >= 0 AND quality_score <= 1),
    delivery_score REAL CHECK (delivery_score >= 0 AND delivery_score <= 1),
    cost_competitiveness REAL CHECK (cost_competitiveness >= 0 AND cost_competitiveness <= 1),
    risk_score REAL CHECK (risk_score >= 0 AND risk_score <= 1),
    sustainability_score REAL CHECK (sustainability_score >= 0 AND sustainability_score <= 1),
    financial_stability REAL CHECK (financial_stability >= 0 AND financial_stability <= 1),
    lead_time_days INTEGER,
    minimum_order_qty INTEGER,
    certifications TEXT, -- JSON array of certifications
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table with detailed specifications
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    base_cost REAL NOT NULL,
    complexity_score REAL CHECK (complexity_score >= 0 AND complexity_score <= 1),
    weight_kg REAL,
    dimensions_cm TEXT,
    lifecycle_stage TEXT CHECK (lifecycle_stage IN ('Introduction', 'Growth', 'Maturity', 'Decline')),
    environmental_impact REAL CHECK (environmental_impact >= 0 AND environmental_impact <= 1),
    required_certifications TEXT, -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Facilities table for manufacturing and distribution centers
CREATE TABLE IF NOT EXISTS facilities (
    facility_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT CHECK (type IN ('Manufacturing', 'Distribution', 'R&D', 'Assembly')),
    region TEXT NOT NULL,
    country TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    capacity INTEGER,
    utilization_rate REAL CHECK (utilization_rate >= 0 AND utilization_rate <= 1),
    operational_cost_per_unit REAL,
    energy_efficiency REAL CHECK (energy_efficiency >= 0 AND energy_efficiency <= 1),
    automation_level REAL CHECK (automation_level >= 0 AND automation_level <= 1),
    quality_certification TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Demand forecast table with ML predictions
CREATE TABLE IF NOT EXISTS demand_forecast (
    demand_id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL,
    facility_id TEXT NOT NULL,
    date DATE NOT NULL,
    forecasted_demand INTEGER,
    actual_demand INTEGER,
    forecast_accuracy REAL CHECK (forecast_accuracy >= 0 AND forecast_accuracy <= 1),
    demand_driver TEXT,
    confidence_interval REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (facility_id) REFERENCES facilities(facility_id)
);

-- Supplier performance metrics
CREATE TABLE IF NOT EXISTS supplier_performance (
    performance_id TEXT PRIMARY KEY,
    supplier_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    evaluation_date DATE NOT NULL,
    quality_score REAL CHECK (quality_score >= 0 AND quality_score <= 1),
    delivery_performance REAL CHECK (delivery_performance >= 0 AND delivery_performance <= 1),
    cost_performance REAL CHECK (cost_performance >= 0 AND cost_performance <= 1),
    responsiveness REAL CHECK (responsiveness >= 0 AND responsiveness <= 1),
    innovation_score REAL CHECK (innovation_score >= 0 AND innovation_score <= 1),
    defect_rate REAL CHECK (defect_rate >= 0),
    on_time_delivery_rate REAL CHECK (on_time_delivery_rate >= 0 AND on_time_delivery_rate <= 1),
    cost_savings_achieved REAL,
    sustainability_compliance REAL CHECK (sustainability_compliance >= 0 AND sustainability_compliance <= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Logistics and transportation data
CREATE TABLE IF NOT EXISTS logistics (
    shipment_id TEXT PRIMARY KEY,
    supplier_id TEXT NOT NULL,
    destination_facility_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    shipment_date DATE NOT NULL,
    transport_mode TEXT CHECK (transport_mode IN ('Air', 'Sea', 'Road', 'Rail', 'Multimodal')),
    quantity INTEGER,
    weight_kg REAL,
    volume_m3 REAL,
    transport_cost REAL,
    estimated_transit_days INTEGER,
    actual_transit_days INTEGER,
    carbon_footprint_kg REAL,
    customs_clearance_time INTEGER,
    insurance_cost REAL,
    delivery_status TEXT CHECK (delivery_status IN ('On Time', 'Delayed', 'Early', 'Damaged', 'Lost')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    FOREIGN KEY (destination_facility_id) REFERENCES facilities(facility_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_suppliers_region ON suppliers(region);
CREATE INDEX IF NOT EXISTS idx_suppliers_tier ON suppliers(tier);
CREATE INDEX IF NOT EXISTS idx_suppliers_risk_score ON suppliers(risk_score);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_lifecycle ON products(lifecycle_stage);

CREATE INDEX IF NOT EXISTS idx_facilities_type ON facilities(type);
CREATE INDEX IF NOT EXISTS idx_facilities_region ON facilities(region);

CREATE INDEX IF NOT EXISTS idx_demand_date ON demand_forecast(date);
CREATE INDEX IF NOT EXISTS idx_demand_product ON demand_forecast(product_id);
CREATE INDEX IF NOT EXISTS idx_demand_facility ON demand_forecast(facility_id);

CREATE INDEX IF NOT EXISTS idx_performance_supplier ON supplier_performance(supplier_id);
CREATE INDEX IF NOT EXISTS idx_performance_date ON supplier_performance(evaluation_date);

CREATE INDEX IF NOT EXISTS idx_logistics_date ON logistics(shipment_date);
CREATE INDEX IF NOT EXISTS idx_logistics_supplier ON logistics(supplier_id);
CREATE INDEX IF NOT EXISTS idx_logistics_status ON logistics(delivery_status);

-- Views for common analytics queries
CREATE VIEW IF NOT EXISTS supplier_scorecard AS
SELECT 
    s.supplier_id,
    s.name,
    s.region,
    s.tier,
    AVG(sp.quality_score) as avg_quality,
    AVG(sp.delivery_performance) as avg_delivery,
    AVG(sp.cost_performance) as avg_cost,
    AVG(sp.defect_rate) as avg_defect_rate,
    COUNT(sp.performance_id) as evaluation_count,
    s.risk_score,
    s.sustainability_score
FROM suppliers s
LEFT JOIN supplier_performance sp ON s.supplier_id = sp.supplier_id
GROUP BY s.supplier_id, s.name, s.region, s.tier, s.risk_score, s.sustainability_score;

CREATE VIEW IF NOT EXISTS demand_accuracy_analysis AS
SELECT 
    df.product_id,
    p.category,
    df.facility_id,
    f.region,
    AVG(df.forecast_accuracy) as avg_accuracy,
    AVG(ABS(df.forecasted_demand - df.actual_demand)) as avg_absolute_error,
    AVG(df.forecasted_demand) as avg_forecasted,
    AVG(df.actual_demand) as avg_actual,
    COUNT(*) as forecast_count
FROM demand_forecast df
JOIN products p ON df.product_id = p.product_id
JOIN facilities f ON df.facility_id = f.facility_id
GROUP BY df.product_id, p.category, df.facility_id, f.region;

CREATE VIEW IF NOT EXISTS logistics_performance AS
SELECT 
    l.transport_mode,
    l.supplier_id,
    s.region as supplier_region,
    AVG(l.transport_cost) as avg_cost,
    AVG(l.actual_transit_days) as avg_transit_days,
    AVG(l.carbon_footprint_kg) as avg_carbon_footprint,
    COUNT(CASE WHEN l.delivery_status = 'On Time' THEN 1 END) * 100.0 / COUNT(*) as on_time_percentage,
    COUNT(*) as shipment_count
FROM logistics l
JOIN suppliers s ON l.supplier_id = s.supplier_id
GROUP BY l.transport_mode, l.supplier_id, s.region;