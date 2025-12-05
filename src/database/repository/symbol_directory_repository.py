from typing import List, Optional, Dict, Any
from datetime import datetime, date
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.utils.logger.custom_logging import LoggerMixin

from src.database.models.symbol_directory import (
    Stock, Cryptocurrency, SymbolChange, DelistedCompany, SymbolDirectorySyncMetadata
)

logger = LoggerMixin().logger


class SymbolDirectoryRepository:
    """
    Repository cho Symbol Directory operations
    
    Features:
    - Bulk upsert (ON CONFLICT DO UPDATE)
    - Fast lookup với indexes
    - Transaction support
    - Error handling
    """
    
    def __init__(self, session: Session):
        """
        Initialize repository với SQLAlchemy session
        
        Args:
            session: SQLAlchemy Session
        """
        self.session = session
    
    # ========================================================================
    # STOCKS OPERATIONS
    # ========================================================================
    
    def upsert_stocks_bulk(self, stocks_data: List[Dict[str, Any]]) -> int:
        """
        Bulk upsert stocks sử dụng PostgreSQL ON CONFLICT
        
        Args:
            stocks_data: List of stock dicts với keys: symbol, name, exchange, etc.
            
        Returns:
            Number of rows affected
            
        Example:
            stocks = [
                {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", ...},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", ...}
            ]
            count = repo.upsert_stocks_bulk(stocks)
        """
        if not stocks_data:
            return 0
        
        try:
            # Build INSERT statement with ON CONFLICT DO UPDATE
            stmt = insert(Stock).values(stocks_data)
            
            # ON CONFLICT DO UPDATE
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol'],
                set_={
                    'name': stmt.excluded.name,
                    'exchange': stmt.excluded.exchange,
                    'exchange_short_name': stmt.excluded.exchange_short_name,
                    'price': stmt.excluded.price,
                    'fmp_data': stmt.excluded.fmp_data,  # Changed from metadata
                    'synced_at': stmt.excluded.synced_at,
                    'updated_at': datetime.utcnow()
                }
            )
            
            result = self.session.execute(stmt)
            self.session.commit()
            
            logger.info(f"Bulk upserted {len(stocks_data)} stocks")
            return len(stocks_data)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error bulk upserting stocks: {str(e)}", exc_info=True)
            raise
    
    def get_stock_by_symbol(self, symbol: str) -> Optional[Any]:
        """
        Get stock by symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock model hoặc None
        """
        stmt = select(Stock).where(Stock.symbol == symbol.upper())
        result = self.session.execute(stmt).scalar_one_or_none()
        return result
    
    def get_all_stocks(self, limit: int = None, offset: int = 0) -> List[Any]:
        """
        Get all stocks với pagination
        
        Args:
            limit: Max records to return
            offset: Offset for pagination
            
        Returns:
            List of Stock models
        """
        stmt = select(Stock).offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        
        result = self.session.execute(stmt).scalars().all()
        return list(result)
    
    def count_stocks(self) -> int:
        """Count total stocks"""
        stmt = select(func.count(Stock.symbol))
        return self.session.execute(stmt).scalar() or 0
    
    def stock_exists(self, symbol: str) -> bool:
        """
        Fast check if stock exists
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if exists
        """
        stmt = select(func.count(Stock.symbol)).where(Stock.symbol == symbol.upper())
        count = self.session.execute(stmt).scalar()
        return count > 0
    
    # ========================================================================
    # CRYPTOCURRENCIES OPERATIONS
    # ========================================================================
    
    def upsert_cryptos_bulk(self, cryptos_data: List[Dict[str, Any]]) -> int:
        """Bulk upsert cryptocurrencies"""
        if not cryptos_data:
            return 0
        
        try:
            stmt = insert(Cryptocurrency).values(cryptos_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol'],
                set_={
                    'name': stmt.excluded.name,
                    'currency': stmt.excluded.currency,
                    'stock_exchange': stmt.excluded.stock_exchange,
                    'exchange_short_name': stmt.excluded.exchange_short_name,
                    'fmp_data': stmt.excluded.fmp_data,  # Changed from metadata
                    'synced_at': stmt.excluded.synced_at,
                    'updated_at': datetime.utcnow()
                }
            )
            
            result = self.session.execute(stmt)
            self.session.commit()
            
            logger.info(f"Bulk upserted {len(cryptos_data)} cryptos")
            return len(cryptos_data)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error bulk upserting cryptos: {str(e)}", exc_info=True)
            raise
    
    def get_crypto_by_symbol(self, symbol: str) -> Optional[Any]:
        """Get crypto by symbol"""
        
        stmt = select(Cryptocurrency).where(Cryptocurrency.symbol == symbol.upper())
        return self.session.execute(stmt).scalar_one_or_none()
    
    def count_cryptos(self) -> int:
        """Count total cryptocurrencies"""
        stmt = select(func.count(Cryptocurrency.symbol))
        return self.session.execute(stmt).scalar() or 0
    
    def crypto_exists(self, symbol: str) -> bool:
        """Fast check if crypto exists"""
        stmt = select(func.count(Cryptocurrency.symbol)).where(Cryptocurrency.symbol == symbol.upper())
        count = self.session.execute(stmt).scalar()
        return count > 0
    
    # ========================================================================
    # SYMBOL CHANGES OPERATIONS
    # ========================================================================
    
    def upsert_symbol_changes_bulk(self, changes_data: List[Dict[str, Any]]) -> int:
        """Bulk upsert symbol changes"""
        if not changes_data:
            return 0
        
        try:
            stmt = insert(SymbolChange).values(changes_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['old_symbol', 'date'],
                set_={
                    'new_symbol': stmt.excluded.new_symbol,
                    'name': stmt.excluded.name,
                    'synced_at': stmt.excluded.synced_at
                }
            )
            
            result = self.session.execute(stmt)
            self.session.commit()
            
            logger.info(f"Bulk upserted {len(changes_data)} symbol changes")
            return len(changes_data)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error bulk upserting symbol changes: {str(e)}", exc_info=True)
            raise
    
    def get_symbol_redirect(self, old_symbol: str) -> Optional[str]:
        """
        Get new symbol for renamed symbol (latest change only)
        
        Args:
            old_symbol: Original symbol
            
        Returns:
            New symbol hoặc None
        """
        
        stmt = (
            select(SymbolChange.new_symbol)
            .where(SymbolChange.old_symbol == old_symbol.upper())
            .order_by(SymbolChange.date.desc())
            .limit(1)
        )
        result = self.session.execute(stmt).scalar_one_or_none()
        return result
    
    def count_symbol_changes(self) -> int:
        """Count total symbol changes"""
        stmt = select(func.count()).select_from(SymbolChange)
        return self.session.execute(stmt).scalar() or 0
    
    # ========================================================================
    # DELISTED COMPANIES OPERATIONS
    # ========================================================================
    
    def upsert_delisted_bulk(self, delisted_data: List[Dict[str, Any]]) -> int:
        """Bulk upsert delisted companies"""
        if not delisted_data:
            return 0
        
        try:
            
            stmt = insert(DelistedCompany).values(delisted_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'delisted_date'],
                set_={
                    'company_name': stmt.excluded.company_name,
                    'exchange': stmt.excluded.exchange,
                    'ipo_date': stmt.excluded.ipo_date,
                    'synced_at': stmt.excluded.synced_at
                }
            )
            
            result = self.session.execute(stmt)
            self.session.commit()
            
            logger.info(f"Bulk upserted {len(delisted_data)} delisted companies")
            return len(delisted_data)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error bulk upserting delisted: {str(e)}", exc_info=True)
            raise
    
    def is_delisted(self, symbol: str) -> bool:
        """Check if symbol is delisted"""
        
        stmt = select(func.count()).where(DelistedCompany.symbol == symbol.upper())
        count = self.session.execute(stmt).scalar()
        return count > 0
    
    def count_delisted(self) -> int:
        """Count total delisted companies"""
        stmt = select(func.count()).select_from(DelistedCompany)
        return self.session.execute(stmt).scalar() or 0
    
    # ========================================================================
    # SYNC METADATA OPERATIONS
    # ========================================================================
    
    def upsert_sync_metadata(
        self,
        entity_type: str,
        total_records: int,
        sync_duration_seconds: int,
        sync_status: str = "success",
        error_message: Optional[str] = None
    ) -> None:
        """
        Update sync metadata cho entity type
        
        Args:
            entity_type: stocks/crypto/symbol_changes/delisted
            total_records: Total records synced
            sync_duration_seconds: Duration in seconds
            sync_status: success/failed
            error_message: Error message if failed
        """
        try:
            
            stmt = insert(SymbolDirectorySyncMetadata).values(
                entity_type=entity_type,
                last_sync_at=datetime.utcnow(),
                total_records=total_records,
                sync_duration_seconds=sync_duration_seconds,
                sync_status=sync_status,
                error_message=error_message,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['entity_type'],
                set_={
                    'last_sync_at': datetime.utcnow(),
                    'total_records': total_records,
                    'sync_duration_seconds': sync_duration_seconds,
                    'sync_status': sync_status,
                    'error_message': error_message,
                    'updated_at': datetime.utcnow()
                }
            )
            
            self.session.execute(stmt)
            self.session.commit()
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating sync metadata: {str(e)}", exc_info=True)
            raise
    
    def get_sync_metadata(self, entity_type: str) -> Optional[Any]:
        """Get sync metadata for entity type"""
        
        stmt = select(SymbolDirectorySyncMetadata).where(
            SymbolDirectorySyncMetadata.entity_type == entity_type
        )
        return self.session.execute(stmt).scalar_one_or_none()
    
    def get_all_sync_metadata(self) -> List[Any]:
        """Get all sync metadata"""
        
        stmt = select(SymbolDirectorySyncMetadata)
        return list(self.session.execute(stmt).scalars().all())
    
    # ========================================================================
    # VALIDATION & SEARCH OPERATIONS
    # ========================================================================
    
    def validate_symbol(
        self,
        symbol: str,
        asset_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate symbol existence và determine asset class
        
        Args:
            symbol: Symbol to validate
            asset_class: Optional filter (stock/crypto)
            
        Returns:
            Dict với keys: is_valid, asset_class, exists
        """
        symbol = symbol.upper()
        
        # Check stocks
        is_stock = self.stock_exists(symbol)
        
        # Check crypto
        is_crypto = self.crypto_exists(symbol)
        
        # Determine asset class
        detected_asset_class = None
        if is_stock:
            detected_asset_class = "stock"
        elif is_crypto:
            detected_asset_class = "crypto"
        
        exists = is_stock or is_crypto
        
        # Validate against requested asset_class
        if asset_class:
            is_valid = (
                (asset_class == "stock" and is_stock) or
                (asset_class == "crypto" and is_crypto)
            )
        else:
            is_valid = exists
        
        return {
            "symbol": symbol,
            "is_valid": is_valid,
            "asset_class": detected_asset_class,
            "exists": exists
        }
    
    def search_symbols(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search symbols by prefix (autocomplete)
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of matching symbols
        """
        query = query.upper()
        results = []
        
        # Search stocks
        stock_stmt = (
            select(Stock.symbol, Stock.name, Stock.exchange)
            .where(Stock.symbol.like(f"{query}%"))
            .limit(limit)
        )
        stocks = self.session.execute(stock_stmt).all()
        
        for symbol, name, exchange in stocks:
            results.append({
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "asset_class": "stock"
            })
        
        # Search crypto
        crypto_stmt = (
            select(Cryptocurrency.symbol, Cryptocurrency.name)
            .where(Cryptocurrency.symbol.like(f"{query}%"))
            .limit(limit - len(results))
        )
        cryptos = self.session.execute(crypto_stmt).all()
        
        for symbol, name in cryptos:
            results.append({
                "symbol": symbol,
                "name": name,
                "asset_class": "crypto"
            })
        
        return results[:limit]