"""
SQLAlchemy Models cho Symbol Directory

Tables:
1. stocks - Danh sách stocks
2. cryptocurrencies - Danh sách cryptos  
3. symbol_changes - Symbol đổi tên
4. delisted_companies - Symbol bị delisted
"""

from sqlalchemy import (
    Column, String, DateTime, Integer, Date, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

from src.database.models.base import Base


# ============================================================================
# 1. STOCKS TABLE
# ============================================================================
class Stock(Base):
    """
    Stock symbol directory
    
    Indexes:
    - PRIMARY KEY: symbol (unique identifier)
    - GIN index on fmp_data for JSONB queries
    - Index on exchange for filtering
    """
    __tablename__ = "stocks"
    
    # Primary key
    symbol = Column(String(20), primary_key=True, index=True, comment="Stock ticker symbol")
    
    # Basic info
    name = Column(String(255), nullable=False, comment="Company name")
    exchange = Column(String(50), nullable=True, index=True, comment="Exchange (NYSE, NASDAQ, etc)")
    exchange_short_name = Column(String(20), nullable=True, comment="Short exchange name")
    price = Column(String(50), nullable=True, comment="Last price as string")
    
    # FMP data (changed from 'metadata' to 'fmp_data' to avoid SQLAlchemy reserved name)
    fmp_data = Column(JSONB, nullable=True, comment="Additional FMP data")
    
    # Tracking fields
    synced_at = Column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        comment="Last sync timestamp"
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_stocks_symbol', 'symbol'),
        Index('idx_stocks_exchange', 'exchange'),
        Index('idx_stocks_fmp_data', 'fmp_data', postgresql_using='gin'),  # Changed
        Index('idx_stocks_synced_at', 'synced_at'),
    )
    
    def __repr__(self):
        return f"<Stock(symbol={self.symbol}, name={self.name}, exchange={self.exchange})>"


# ============================================================================
# 2. CRYPTOCURRENCIES TABLE
# ============================================================================
class Cryptocurrency(Base):
    """
    Cryptocurrency symbol directory
    
    Indexes:
    - PRIMARY KEY: symbol (unique identifier)
    - GIN index on fmp_data
    """
    __tablename__ = "cryptocurrencies"
    
    # Primary key
    symbol = Column(String(20), primary_key=True, index=True, comment="Crypto symbol (e.g., BTCUSD)")
    
    # Basic info
    name = Column(String(255), nullable=False, comment="Crypto name")
    currency = Column(String(20), nullable=True, comment="Base currency")
    stock_exchange = Column(String(50), nullable=True, comment="Exchange")
    exchange_short_name = Column(String(20), nullable=True, comment="Short exchange name")
    
    # FMP data (changed from 'metadata' to 'fmp_data')
    fmp_data = Column(JSONB, nullable=True, comment="Additional FMP data")
    
    # Tracking
    synced_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_crypto_symbol', 'symbol'),
        Index('idx_crypto_fmp_data', 'fmp_data', postgresql_using='gin'),  # Changed
        Index('idx_crypto_synced_at', 'synced_at'),
    )
    
    def __repr__(self):
        return f"<Cryptocurrency(symbol={self.symbol}, name={self.name})>"


# ============================================================================
# 3. SYMBOL CHANGES TABLE
# ============================================================================
class SymbolChange(Base):
    """
    Track symbol changes (renames, ticker changes)
    
    Indexes:
    - Composite primary key (old_symbol, date)
    - Index on old_symbol for redirect lookup
    - Index on new_symbol for reverse lookup
    - Index on date for recent changes
    """
    __tablename__ = "symbol_changes"
    
    # Composite primary key
    old_symbol = Column(String(20), primary_key=True, comment="Original symbol")
    date = Column(Date, primary_key=True, comment="Change date")
    
    # Change details
    new_symbol = Column(String(20), nullable=False, index=True, comment="New symbol")
    name = Column(String(255), nullable=True, comment="Company name")
    
    # Tracking
    synced_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_symbol_changes_old', 'old_symbol'),
        Index('idx_symbol_changes_new', 'new_symbol'),
        Index('idx_symbol_changes_date', 'date'),
        Index('idx_symbol_changes_synced', 'synced_at'),
    )
    
    def __repr__(self):
        return f"<SymbolChange(old={self.old_symbol}, new={self.new_symbol}, date={self.date})>"


# ============================================================================
# 4. DELISTED COMPANIES TABLE
# ============================================================================
class DelistedCompany(Base):
    """
    Track delisted companies
    
    Indexes:
    - PRIMARY KEY: symbol + delisted_date (composite)
    - Index on symbol for lookup
    - Index on delisted_date for recent delistings
    """
    __tablename__ = "delisted_companies"
    
    # Composite primary key
    symbol = Column(String(20), primary_key=True, comment="Delisted symbol")
    delisted_date = Column(Date, primary_key=True, nullable=True, comment="Delisting date")
    
    # Company details
    company_name = Column(String(255), nullable=True, comment="Company name")
    exchange = Column(String(50), nullable=True, comment="Exchange")
    ipo_date = Column(Date, nullable=True, comment="IPO date")
    
    # Tracking
    synced_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_delisted_symbol', 'symbol'),
        Index('idx_delisted_date', 'delisted_date'),
        Index('idx_delisted_exchange', 'exchange'),
        Index('idx_delisted_synced', 'synced_at'),
    )
    
    def __repr__(self):
        return f"<DelistedCompany(symbol={self.symbol}, date={self.delisted_date})>"


# ============================================================================
# 5. SYNC METADATA TABLE (Track sync jobs)
# ============================================================================
class SymbolDirectorySyncMetadata(Base):
    """
    Track sync job metadata
    
    Single row table - stores last sync info for each entity type
    """
    __tablename__ = "symbol_directory_sync_metadata"
    
    # Primary key
    entity_type = Column(
        String(50), 
        primary_key=True, 
        comment="Entity type: stocks, crypto, symbol_changes, delisted"
    )
    
    # Sync info
    last_sync_at = Column(DateTime, nullable=False, comment="Last successful sync")
    total_records = Column(Integer, nullable=False, default=0, comment="Total records synced")
    sync_duration_seconds = Column(Integer, nullable=True, comment="Sync duration")
    sync_status = Column(String(20), nullable=False, default="success", comment="success/failed")
    error_message = Column(String(500), nullable=True, comment="Error if failed")
    
    # Tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_sync_metadata_entity', 'entity_type'),
        Index('idx_sync_metadata_last_sync', 'last_sync_at'),
    )
    
    def __repr__(self):
        return f"<SyncMetadata(type={self.entity_type}, last_sync={self.last_sync_at}, status={self.sync_status})>"