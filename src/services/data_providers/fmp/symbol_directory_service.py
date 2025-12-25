import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from sqlalchemy.orm import Session

from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.database import get_postgres_db

from src.models.symbol_directory import (
    StockListItem, StockListResponse,
    CryptoListItem, CryptoListResponse,
    SymbolChangeItem, SymbolChangesResponse,
    DelistedCompanyItem, DelistedCompaniesResponse,
    SymbolStatusDetail, SymbolStatus, AssetClass,
    SymbolValidationResult, CacheStatistics,
    SyncJobResult
)

from src.database.repository.symbol_directory_repository import SymbolDirectoryRepository

logger = LoggerMixin().logger


class SymbolDirectoryService:
    """
    Symbol Directory Service - PostgreSQL Version
    
    Features:
    - Fetch từ FMP API
    - Bulk upsert vào PostgreSQL
    - Fast validation với DB indexes
    - Symbol redirect support
    """
    
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize service
        
        Args:
            session: Optional SQLAlchemy session (nếu không có sẽ tạo mới)
        """
        self.base_url = settings.FMP_URL_STABLE or "https://financialmodelingprep.com/stable"
        self.api_key = settings.FMP_API_KEY
        self.timeout = httpx.Timeout(60.0, connect=15.0)
        
        # Session và repository
        if session:
            self.session = session
            self.repo = SymbolDirectoryRepository(session)
            self._owns_session = False
        else:
            # Tạo session mới
            self.session = get_postgres_db().get_session()
            self.repo = SymbolDirectoryRepository(self.session)
            self._owns_session = True
    
    def __del__(self):
        """Cleanup session nếu service tự tạo"""
        if hasattr(self, '_owns_session') and self._owns_session and hasattr(self, 'session'):
            self.session.close()
    
    # ========================================================================
    # FMP API CALLS 


    
    # ========================================================================
    
    async def _make_fmp_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Optional[List[Dict]]:
        """Call FMP API"""
        if not self.api_key or self.api_key == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error("FMP API key is not configured")
            return None
        
        url = f"{self.base_url}/{endpoint}"
        
        if params is None:
            params = {}
        params["apikey"] = self.api_key
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"FMP Request: {endpoint}")
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if isinstance(data, dict) and "Error Message" in data:
                    logger.error(f"FMP API Error: {data['Error Message']}")
                    return None
                
                if not isinstance(data, list):
                    logger.warning(f"Unexpected FMP response format: {type(data)}")
                    return None
                
                logger.info(f"FMP Response: {len(data)} records fetched from {endpoint}")
                return data
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling FMP: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.TimeoutException:
            logger.error(f"Timeout calling FMP endpoint: {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling FMP: {str(e)}", exc_info=True)
            return None
    
    # ========================================================================
    # STOCK LIST OPERATIONS
    # ========================================================================
    
    async def fetch_and_sync_stock_list(self) -> SyncJobResult:
        """
        Fetch stock list từ FMP và sync vào DB
        
        Returns:
            SyncJobResult với status và metrics
        """
        start_time = datetime.utcnow()
        
        try:
            # Fetch từ FMP
            data = await self._make_fmp_request("stock-list")
            
            if data is None:
                raise Exception("Failed to fetch stock list from FMP")
            
            # Prepare data for bulk upsert
            stocks_data = []
            for item in data:
                stock_dict = {
                    "symbol": item.get("symbol", "").upper(),
                    "name": item.get("name", ""),
                    "exchange": item.get("exchange"),
                    "exchange_short_name": item.get("exchangeShortName"),
                    "price": str(item.get("price", "")),
                    "fmp_data": item,  # Changed from metadata
                    "synced_at": datetime.utcnow(),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                stocks_data.append(stock_dict)
            
            # Bulk upsert
            count = self.repo.upsert_stocks_bulk(stocks_data)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Update sync metadata
            self.repo.upsert_sync_metadata(
                entity_type="stocks",
                total_records=count,
                sync_duration_seconds=int(duration),
                sync_status="success"
            )
            
            logger.info(f"Stock list sync completed: {count} stocks in {duration:.2f}s")
            
            return SyncJobResult(
                job_type="stock_list",
                success=True,
                records_fetched=len(data),
                records_cached=count,
                duration_seconds=duration
            )
        
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Stock list sync failed: {str(e)}", exc_info=True)
            
            # Update metadata with error
            try:
                self.repo.upsert_sync_metadata(
                    entity_type="stocks",
                    total_records=0,
                    sync_duration_seconds=int(duration),
                    sync_status="failed",
                    error_message=str(e)[:500]
                )
            except:
                pass
            
            return SyncJobResult(
                job_type="stock_list",
                success=False,
                records_fetched=0,
                records_cached=0,
                duration_seconds=duration,
                error_message=str(e)
            )
    
    # ========================================================================
    # CRYPTO LIST OPERATIONS
    # ========================================================================
    
    async def fetch_and_sync_crypto_list(self) -> SyncJobResult:
        """Fetch và sync crypto list"""
        start_time = datetime.utcnow()
        
        try:
            data = await self._make_fmp_request("cryptocurrency-list")
            
            if data is None:
                raise Exception("Failed to fetch crypto list from FMP")
            
            cryptos_data = []
            for item in data:
                crypto_dict = {
                    "symbol": item.get("symbol", "").upper(),
                    "name": item.get("name", ""),
                    "currency": item.get("currency"),
                    "stock_exchange": item.get("stockExchange"),
                    "exchange_short_name": item.get("exchangeShortName"),
                    "fmp_data": item,  # Changed from metadata
                    "synced_at": datetime.utcnow(),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                cryptos_data.append(crypto_dict)
            
            count = self.repo.upsert_cryptos_bulk(cryptos_data)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            self.repo.upsert_sync_metadata(
                entity_type="crypto",
                total_records=count,
                sync_duration_seconds=int(duration),
                sync_status="success"
            )
            
            logger.info(f"Crypto list sync completed: {count} cryptos in {duration:.2f}s")
            
            return SyncJobResult(
                job_type="crypto_list",
                success=True,
                records_fetched=len(data),
                records_cached=count,
                duration_seconds=duration
            )
        
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Crypto list sync failed: {str(e)}", exc_info=True)
            
            try:
                self.repo.upsert_sync_metadata(
                    entity_type="crypto",
                    total_records=0,
                    sync_duration_seconds=int(duration),
                    sync_status="failed",
                    error_message=str(e)[:500]
                )
            except:
                pass
            
            return SyncJobResult(
                job_type="crypto_list",
                success=False,
                records_fetched=0,
                records_cached=0,
                duration_seconds=duration,
                error_message=str(e)
            )
    
    # ========================================================================
    # SYMBOL CHANGES OPERATIONS
    # ========================================================================
    
    async def fetch_and_sync_symbol_changes(self) -> SyncJobResult:
        """Fetch và sync symbol changes"""
        start_time = datetime.utcnow()
        
        try:
            data = await self._make_fmp_request("symbol-change")
            
            if data is None:
                raise Exception("Failed to fetch symbol changes from FMP")
            
            changes_data = []
            for item in data:
                # Parse date
                date_str = item.get("date")
                change_date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else date.today()
                
                change_dict = {
                    "old_symbol": item.get("oldSymbol", "").upper(),
                    "new_symbol": item.get("newSymbol", "").upper(),
                    "date": change_date,
                    "name": item.get("name"),
                    "synced_at": datetime.utcnow(),
                    "created_at": datetime.utcnow()
                }
                changes_data.append(change_dict)
            
            count = self.repo.upsert_symbol_changes_bulk(changes_data)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            self.repo.upsert_sync_metadata(
                entity_type="symbol_changes",
                total_records=count,
                sync_duration_seconds=int(duration),
                sync_status="success"
            )
            
            logger.info(f"Symbol changes sync completed: {count} changes in {duration:.2f}s")
            
            return SyncJobResult(
                job_type="symbol_changes",
                success=True,
                records_fetched=len(data),
                records_cached=count,
                duration_seconds=duration
            )
        
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Symbol changes sync failed: {str(e)}", exc_info=True)
            
            try:
                self.repo.upsert_sync_metadata(
                    entity_type="symbol_changes",
                    total_records=0,
                    sync_duration_seconds=int(duration),
                    sync_status="failed",
                    error_message=str(e)[:500]
                )
            except:
                pass
            
            return SyncJobResult(
                job_type="symbol_changes",
                success=False,
                records_fetched=0,
                records_cached=0,
                duration_seconds=duration,
                error_message=str(e)
            )
    
    # ========================================================================
    # DELISTED COMPANIES OPERATIONS
    # ========================================================================
    
    async def fetch_and_sync_delisted_companies(self) -> SyncJobResult:
        """Fetch và sync delisted companies"""
        start_time = datetime.utcnow()
        
        try:
            data = await self._make_fmp_request("delisted-companies", params={"page": 0, "limit": 10000})
            
            if data is None:
                raise Exception("Failed to fetch delisted companies from FMP")
            
            delisted_data = []
            for item in data:
                # Parse dates
                delisted_date_str = item.get("delistedDate")
                ipo_date_str = item.get("ipoDate")
                
                delisted_date = None
                if delisted_date_str:
                    try:
                        delisted_date = datetime.strptime(delisted_date_str, "%Y-%m-%d").date()
                    except:
                        pass
                
                ipo_date = None
                if ipo_date_str:
                    try:
                        ipo_date = datetime.strptime(ipo_date_str, "%Y-%m-%d").date()
                    except:
                        pass
                
                delisted_dict = {
                    "symbol": item.get("symbol", "").upper(),
                    "delisted_date": delisted_date or date.today(),
                    "company_name": item.get("companyName"),
                    "exchange": item.get("exchange"),
                    "ipo_date": ipo_date,
                    "synced_at": datetime.utcnow(),
                    "created_at": datetime.utcnow()
                }
                delisted_data.append(delisted_dict)
            
            count = self.repo.upsert_delisted_bulk(delisted_data)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            self.repo.upsert_sync_metadata(
                entity_type="delisted",
                total_records=count,
                sync_duration_seconds=int(duration),
                sync_status="success"
            )
            
            logger.info(f"Delisted companies sync completed: {count} companies in {duration:.2f}s")
            
            return SyncJobResult(
                job_type="delisted",
                success=True,
                records_fetched=len(data),
                records_cached=count,
                duration_seconds=duration
            )
        
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Delisted sync failed: {str(e)}", exc_info=True)
            
            try:
                self.repo.upsert_sync_metadata(
                    entity_type="delisted",
                    total_records=0,
                    sync_duration_seconds=int(duration),
                    sync_status="failed",
                    error_message=str(e)[:500]
                )
            except:
                pass
            
            return SyncJobResult(
                job_type="delisted",
                success=False,
                records_fetched=0,
                records_cached=0,
                duration_seconds=duration,
                error_message=str(e)
            )
    
    # ========================================================================
    # FULL SYNC
    # ========================================================================
    
    async def sync_all(self) -> List[SyncJobResult]:
        """
        Sync tất cả data từ FMP vào PostgreSQL
        
        Total FMP API calls: 4
        
        Returns:
            List of SyncJobResult
        """
        results = []
        
        logger.info("=" * 80)
        logger.info("FULL SYMBOL DIRECTORY SYNC STARTED")
        logger.info("=" * 80)
        
        # 1. Sync stocks
        result = await self.fetch_and_sync_stock_list()
        results.append(result)
        
        # 2. Sync crypto
        result = await self.fetch_and_sync_crypto_list()
        results.append(result)
        
        # 3. Sync symbol changes
        result = await self.fetch_and_sync_symbol_changes()
        results.append(result)
        
        # 4. Sync delisted
        result = await self.fetch_and_sync_delisted_companies()
        results.append(result)
        
        # Summary
        total_records = sum(r.records_cached for r in results)
        successful = sum(1 for r in results if r.success)
        
        logger.info("-" * 80)
        logger.info(f"SYNC SUMMARY: {successful}/{len(results)} jobs successful, {total_records} total records")
        logger.info("=" * 80)
        
        return results
    
    # ========================================================================
    # VALIDATION & QUERY OPERATIONS
    # ========================================================================
    
    def validate_symbols_bulk(
        self, 
        symbols: List[str], 
        asset_class_filter: Optional[AssetClass] = None
    ) -> tuple[List[SymbolValidationResult], List[str], List[str]]:
        """
        Validate multiple symbols efficiently.
        """
        # Convert Enum to string value for Repository (e.g., AssetClass.STOCK -> "stock")
        asset_class_str = asset_class_filter.value if asset_class_filter else None
        
        # Batch lookup with optimized filter
        existing_map = self.repo.get_existing_symbols_batch(symbols, asset_class_str)
        
        results = []
        valid_symbols = []
        invalid_symbols = []

        for symbol in symbols:
            upper_symbol = symbol.upper()
            found_asset_class = existing_map.get(upper_symbol)
            
            is_valid = found_asset_class is not None
            
            # Create result object
            result_item = SymbolValidationResult(
                symbol=upper_symbol,
                is_valid=is_valid,
                asset_class=AssetClass(found_asset_class) if found_asset_class else None,
                exists_in_cache=is_valid,
                message=f"Valid {found_asset_class}" if is_valid else "Symbol not found"
            )
            
            results.append(result_item)
            
            if is_valid:
                valid_symbols.append(upper_symbol)
            else:
                invalid_symbols.append(upper_symbol)

        return results, valid_symbols, invalid_symbols
    
    def validate_symbol(
        self,
        symbol: str,
        asset_class: Optional[AssetClass] = None
    ) -> SymbolValidationResult:
        """
        Validate symbol với DB lookup
        
        Performance: < 5ms với proper indexing
        """
        result = self.repo.validate_symbol(
            symbol,
            asset_class.value if asset_class else None
        )
        
        return SymbolValidationResult(
            symbol=result["symbol"],
            is_valid=result["is_valid"],
            asset_class=AssetClass(result["asset_class"]) if result["asset_class"] else None,
            exists_in_cache=result["exists"],
            message=f"Valid {result['asset_class']}" if result["is_valid"] else "Symbol not found"
        )
    
    def get_symbol_status(self, symbol: str) -> SymbolStatusDetail:
        """Get comprehensive symbol status"""
        symbol = symbol.upper()
        
        # Validate existence
        validation = self.validate_symbol(symbol)
        
        # Check if renamed
        new_symbol = self.repo.get_symbol_redirect(symbol)
        if new_symbol:
            return SymbolStatusDetail(
                input_symbol=symbol,
                status=SymbolStatus.RENAMED,
                asset_class=AssetClass.STOCK,
                exists=True,
                is_actively_trading=False,
                current_symbol=new_symbol,
                redirect_symbol=new_symbol,
                message=f"Symbol renamed from {symbol} to {new_symbol}"
            )
        
        # Check if delisted
        if self.repo.is_delisted(symbol):
            return SymbolStatusDetail(
                input_symbol=symbol,
                status=SymbolStatus.DELISTED,
                asset_class=validation.asset_class,
                exists=True,
                is_actively_trading=False,
                current_symbol=symbol,
                message=f"Symbol {symbol} has been delisted"
            )
        
        # Active symbol
        if validation.is_valid:
            return SymbolStatusDetail(
                input_symbol=symbol,
                status=SymbolStatus.ACTIVE,
                asset_class=validation.asset_class,
                exists=True,
                is_actively_trading=True,
                current_symbol=symbol,
                message=f"Active {validation.asset_class.value} symbol"
            )
        
        # Not found
        return SymbolStatusDetail(
            input_symbol=symbol,
            status=SymbolStatus.INACTIVE,
            exists=False,
            is_actively_trading=False,
            message="Symbol not found"
        )
    
    def get_statistics(self) -> CacheStatistics:
        """Get sync statistics từ DB"""
        # Count symbols
        stocks_count = self.repo.count_stocks()
        crypto_count = self.repo.count_cryptos()
        symbol_changes_count = self.repo.count_symbol_changes()
        delisted_count = self.repo.count_delisted()
        
        # Get sync metadata
        stock_meta = self.repo.get_sync_metadata("stocks")
        crypto_meta = self.repo.get_sync_metadata("crypto")
        changes_meta = self.repo.get_sync_metadata("symbol_changes")
        delisted_meta = self.repo.get_sync_metadata("delisted")
        
        return CacheStatistics(
            stocks_count=stocks_count,
            crypto_count=crypto_count,
            symbol_changes_count=symbol_changes_count,
            delisted_count=delisted_count,
            last_stock_sync=stock_meta.last_sync_at if stock_meta else None,
            last_crypto_sync=crypto_meta.last_sync_at if crypto_meta else None,
            last_changes_sync=changes_meta.last_sync_at if changes_meta else None,
            last_delisted_sync=delisted_meta.last_sync_at if delisted_meta else None,
            cache_ttl_seconds=0,  # No TTL for DB
            next_sync_in=None  # Monthly sync managed by job
        )


# ============================================================================
# SERVICE FACTORY
# ============================================================================
def get_symbol_directory_service(session: Optional[Session] = None) -> SymbolDirectoryService:
    """
    Factory function để tạo service instance
    
    Args:
        session: Optional SQLAlchemy session
        
    Returns:
        SymbolDirectoryService instance
    """
    return SymbolDirectoryService(session)