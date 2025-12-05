from pydantic import BaseModel, Field
from typing import Optional, Union
from datetime import date


# Schema FMP API /v3/sp500_constituent
class SP500ConstituentItem(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    subSector: Optional[str] = Field(None, alias="subSector")
    headQuarter: Optional[str] = Field(None, alias="headQuarter")
    dateFirstAdded: Optional[Union[date, str]] = Field(None, alias="dateFirstAdded")
    cik: Optional[str] = None
    founded: Optional[str] = None

    class Config:
        populate_by_name = True

# Schema FMP API /v3/quote/
class SP500QuoteDataItem(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    changesPercentage: Optional[float] = None
    change: Optional[float] = None
    dayLow: Optional[float] = None
    dayHigh: Optional[float] = None
    yearHigh: Optional[float] = None
    yearLow: Optional[float] = None
    marketCap: Optional[float] = None
    priceAvg50: Optional[float] = None
    priceAvg200: Optional[float] = None
    exchange: Optional[str] = None
    volume: Optional[float] = None
    avgVolume: Optional[float] = None 
    open_price: Optional[float] = Field(None, alias="open")
    previousClose: Optional[float] = None
    eps: Optional[float] = None
    pe: Optional[float] = None

    class Config:
        populate_by_name = True

# Schema response
class SP500QuoteWithSectorItem(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    subSector: Optional[str] = None
    price: Optional[float] = None
    changesPercentage: Optional[float] = None
    change: Optional[float] = None
    dayLow: Optional[float] = None
    dayHigh: Optional[float] = None
    yearHigh: Optional[float] = None
    yearLow: Optional[float] = None
    marketCap: Optional[float] = None
    priceAvg50: Optional[float] = None
    priceAvg200: Optional[float] = None
    exchange: Optional[str] = None
    volume: Optional[float] = None
    avgVolume: Optional[float] = None
    open_price: Optional[float] = Field(None, alias="open")
    previousClose: Optional[float] = None
    eps: Optional[float] = None
    pe: Optional[float] = None

    class Config:
        populate_by_name = True