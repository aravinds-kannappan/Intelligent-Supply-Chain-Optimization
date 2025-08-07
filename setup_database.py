"""
Database setup and data loading script
"""
import os
import pandas as pd
from config.database_config import db_manager
from data_generation.supply_chain_generator import SupplyChainDataGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Set up database schema and load data"""
    logger.info("Setting up supply chain database...")
    
    # Create database schema
    with open('database/schema.sql', 'r') as f:
        schema_sql = f.read()
    
    # Execute schema creation
    with db_manager.get_connection() as conn:
        conn.executescript(schema_sql)
        logger.info("Database schema created successfully")
    
    # Generate data if not exists
    data_dir = 'data'
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        logger.info("Generating supply chain data...")
        generator = SupplyChainDataGenerator()
        data = generator.generate_all_data()
        
        # Save to CSV files
        os.makedirs(data_dir, exist_ok=True)
        for table_name, df in data.items():
            df.to_csv(f'{data_dir}/{table_name}.csv', index=False)
            logger.info(f"Generated {len(df)} records for {table_name}")
    
    # Load data into database
    logger.info("Loading data into database...")
    
    tables = ['suppliers', 'products', 'facilities', 'demand_forecast', 
              'supplier_performance', 'logistics']
    
    for table in tables:
        csv_file = f'{data_dir}/{table}.csv'
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.to_sql(table, db_manager.engine, if_exists='replace', index=False)
            logger.info(f"Loaded {len(df)} records into {table} table")
        else:
            logger.warning(f"CSV file {csv_file} not found")
    
    logger.info("Database setup completed successfully!")

if __name__ == "__main__":
    setup_database()