import logging
from pydantic import BaseModel, ConfigDict, Field, conint, field_validator, validator
from typing import Any, Dict, Literal, Optional, List, Union, TypeVar, Generic
from datetime import date, datetime
from enum import Enum

from src.utils.logger.set_up_log_dataFMP import setup_logger 
logger = setup_logger(__name__, log_level=logging.INFO) 

DataType = TypeVar('DataType')

class ScrapeRequest(BaseModel):
    usernames: List[str] = Field(..., min_length=1, description="Một danh sách các username trên Twitter cần cào dữ liệu.")

class TwitterAuthorSchema(BaseModel):

    id: int
    
    author_id: str
    
    username: str
    name: str
    is_verified: bool
    profile_picture_url: Optional[str] = None
    followers_count: Optional[int] = None # Dùng int cho BigInteger là đủ trong Pydantic
    following_count: Optional[int] = None
    statuses_count: Optional[int] = None
    
    class Config:
        from_attributes = True

class TweetSchema(BaseModel):
    # ID này là khóa chính tự tăng trong DB, kiểu số
    id: int
    
    # ID gốc của Tweet từ Twitter, kiểu chuỗi
    tweet_id: str
    
    # Khóa ngoại đến ID của bảng author, là kiểu số
    author_id: int
    
    text: str
    url: str
    created_at: datetime
    retweet_count: int
    reply_count: int
    like_count: int
    quote_count: int
    view_count: Optional[int] = None
    hashtags: Optional[List[str]] = []
    
    author: TwitterAuthorSchema

    class Config:
        from_attributes = True


class BtcInputOutput(BaseModel):
    address: Optional[str] = Field(None, alias="addr")
    value_in_satoshi: int = Field(..., alias="value")
    
    value_in_btc: float

class ParsedBtcTransaction(BaseModel):
    hash: str
    time: int 
    size: int
    fee: int 
    
    fee_in_btc: float
    input_value_btc: float
    output_value_btc: float
    
    inputs: List[BtcInputOutput]
    outputs: List[BtcInputOutput] = Field(..., alias="out")

class OnchainSubscriptionParams(BaseModel):
    addresses: List[str] = Field(..., description="Danh sách các địa chỉ ví Ethereum cần theo dõi.", min_length=1)

class FullOnchainTransaction(BaseModel):
    # Các trường chính
    hash: str
    block_hash: str = Field(..., alias="blockHash")
    block_number: int = Field(..., alias="blockNumber")
    from_address: str = Field(..., alias="from")
    to_address: Optional[str] = Field(None, alias="to") # 'to' có thể là null trong giao dịch tạo contract

    # Dữ liệu đã được chuyển đổi
    value_in_ether: float
    gas: int
    gas_price_in_gwei: float # Chuyển gasPrice sang Gwei cho dễ đọc
    nonce: int
    transaction_index: int = Field(..., alias="transactionIndex")
    
    # Các trường phụ
    type: str # Sẽ chuyển 0x2 thành "EIP-1559"
    input: str
    decoded_input: Optional[Dict[str, Any]] = Field(None, description="Dữ liệu input đã được giải mã nếu có.")

    class Config:
        populate_by_name = True

class PaginatedData(BaseModel, Generic[DataType]):
    totalRows: int
    data: List[DataType]

class MarketRegionEnum(str, Enum):
    ALL = "ALL"  
    US = "US"
    EUROPE = "EUROPE"
    ASIA = "ASIA"
    AMERICAS_EX_US = "AMERICA"
    AFRICA_MIDDLE_EAST = "AFRICA"
    COMMODITIES = "COMMODITIES"
    CRYPTOCURRENCIES = "CRYPTOCURRENCIES"

class CustomListWsParams(BaseModel):
    symbols: List[str] = Field(..., description="Danh sách các mã cổ phiếu/crypto/etf tùy chỉnh để lấy dữ liệu.")
    asset_type: str = Field("stock", description="Loại tài sản: 'stock', 'etf', hoặc 'crypto'")

class WebSocketRequest(BaseModel):
    event: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class PatternPoint(BaseModel):
    time: str
    value: float

class Trendline(BaseModel):
    line_type: Literal["resistance", "support"]
    points: List[PatternPoint]

class DetectedPattern(BaseModel):
    pattern_name: str
    path_points: Optional[List[PatternPoint]] = None
    trendlines: Optional[List[Trendline]] = None

class MACDOutput(BaseModel):
    macd_line: Optional[float] = Field(None, description="Đường MACD (EMA12 - EMA26)")
    signal_line: Optional[float] = Field(None, description="Đường tín hiệu (EMA9 của đường MACD)")
    histogram: Optional[float] = Field(None, description="MACD Histogram (MACD Line - Signal Line)")

class ChartDataItem(BaseModel):
    time: str = Field(..., description="Chuỗi datetime định dạng YYYY-MM-DD HH:MM:SS")
    value: float = Field(..., description="Giá trị (close price) tại thời điểm tương ứng")
    ma5: Optional[float] = Field(None, description="Giá trị đường trung bình động 5 kỳ (MA5)")
    ma20: Optional[float] = Field(None, description="Giá trị đường trung bình động 20 kỳ (MA20)")
    rsi: Optional[float] = Field(None, description="Chỉ số sức mạnh tương đối (RSI) 14 kỳ")
    macd: Optional[MACDOutput] = Field(None, description="Dữ liệu chỉ báo MACD (12, 26, 9)")

class PageStreamParams(BaseModel):
    page: int = Field(1, description="Số trang, bắt đầu từ 1.")
    limit: int = Field(20, description="Số lượng kết quả mỗi trang (1-100).")

class DiscoveryStreamParams(BaseModel):
    limit: int = Field(10, description="Số lượng kết quả trả về (tối đa 30).")

class MarketRegionParams(BaseModel):
    region: Optional[MarketRegionEnum] = None

class TickerTapeWSParams(BaseModel):
    symbols: str = Field(..., description="Danh sách các mã, cách nhau bởi dấu phẩy. Ví dụ: 'AAPL,MSFT,GOOG'")

class KeyStatsOutput(BaseModel):
    win_rate_all_positive_earnings: float = Field(..., alias="winRateAllPositiveEarnings", description="Tỷ lệ % các mã có lợi nhuận dương và giá tăng sau 20 ngày.")
    win_rate_chosen_picks: float = Field(..., alias="winRateChosenPicks", description="Tỷ lệ % các mã được chọn có giá tăng sau 20 ngày.")
    avg_return_chosen_picks: float = Field(..., alias="avgReturnChosenPicks", description="Lợi nhuận trung bình % sau 20 ngày của các mã được chọn.")
    class Config:
        populate_by_name = True
        
class PriceTargetItem(BaseModel):
    symbol: str
    published_date: datetime = Field(..., alias="publishedDate")
    news_url: str = Field(..., alias="newsURL")
    news_title: str = Field(..., alias="newsTitle")
    analyst_name: Optional[str] = Field(None, alias="analystName")
    price_target: float = Field(..., alias="priceTarget")
    adj_price_target: float = Field(..., alias="adjPriceTarget")
    price_when_posted: float = Field(..., alias="priceWhenPosted")
    news_publisher: str = Field(..., alias="newsPublisher")
    news_base_url: str = Field(..., alias="newsBaseURL")
    analyst_company: Optional[str] = Field(None, alias="analystCompany")
    
    class Config:
        populate_by_name = True

class PriceTargetWithChartOutput(PriceTargetItem):
    # Kế thừa từ PriceTargetItem và thêm trường chartData
    chart_data: List[ChartDataItem] = Field([], alias="chartData")

class PressReleaseItem(BaseModel):
    symbol: Optional[str] = None
    published_date: datetime = Field(..., alias="publishedDate")
    publisher: str
    title: str
    image: Optional[str] = None
    site: str
    text: str
    url: str

    class Config:
        populate_by_name = True

class AnalystEstimateItem(BaseModel):
    symbol: Optional[str] = Field(default=None)
    # date: Optional[date] = Field(default=None, description="Ngày của ước tính, thường là cuối kỳ")

    estimatedRevenueLow: Optional[float] = Field(default=None)
    estimatedRevenueHigh: Optional[float] = Field(default=None)
    estimatedRevenueAvg: Optional[float] = Field(default=None)

    estimatedEbitdaLow: Optional[float] = Field(default=None)
    estimatedEbitdaHigh: Optional[float] = Field(default=None)
    estimatedEbitdaAvg: Optional[float] = Field(default=None)

    estimatedEbitLow: Optional[float] = Field(default=None)
    estimatedEbitHigh: Optional[float] = Field(default=None)
    estimatedEbitAvg: Optional[float] = Field(default=None)

    estimatedNetIncomeLow: Optional[float] = Field(default=None)
    estimatedNetIncomeHigh: Optional[float] = Field(default=None)
    estimatedNetIncomeAvg: Optional[float] = Field(default=None)

    estimatedSgaExpenseLow: Optional[float] = Field(default=None)
    estimatedSgaExpenseHigh: Optional[float] = Field(default=None)
    estimatedSgaExpenseAvg: Optional[float] = Field(default=None)

    estimatedEpsAvg: Optional[float] = Field(default=None)
    estimatedEpsHigh: Optional[float] = Field(default=None)
    estimatedEpsLow: Optional[float] = Field(default=None)

    numberAnalystEstimatedRevenue: Optional[int] = Field(default=None, alias="numberAnalystsEstimatedRevenue")
    numberAnalystsEstimatedEps: Optional[int] = Field(default=None)


class KeyMetricsTTMItem(BaseModel):
    revenuePerShareTTM: Optional[float] = Field(default=None)
    netIncomePerShareTTM: Optional[float] = Field(default=None)
    operatingCashFlowPerShareTTM: Optional[float] = Field(default=None)
    freeCashFlowPerShareTTM: Optional[float] = Field(default=None)
    cashPerShareTTM: Optional[float] = Field(default=None)
    bookValuePerShareTTM: Optional[float] = Field(default=None)
    tangibleBookValuePerShareTTM: Optional[float] = Field(default=None)
    shareholdersEquityPerShareTTM: Optional[float] = Field(default=None)
    interestDebtPerShareTTM: Optional[float] = Field(default=None)
    marketCapTTM: Optional[float] = Field(default=None)
    enterpriseValueTTM: Optional[float] = Field(default=None)
    peRatioTTM: Optional[float] = Field(default=None)
    priceToSalesRatioTTM: Optional[float] = Field(default=None)
    pocfratioTTM: Optional[float] = Field(default=None, alias="pfcfRatioTTM") # Price to Operating Cash Flow Ratio
    pfcfRatioTTM: Optional[float] = Field(default=None) # Price to Free Cash Flow Ratio
    pbRatioTTM: Optional[float] = Field(default=None)
    ptbRatioTTM: Optional[float] = Field(default=None) # Price to Tangible Book Ratio
    evToSalesTTM: Optional[float] = Field(default=None)
    enterpriseValueOverEBITDATTM: Optional[float] = Field(default=None)
    evToOperatingCashFlowTTM: Optional[float] = Field(default=None)
    evToFreeCashFlowTTM: Optional[float] = Field(default=None)
    earningsYieldTTM: Optional[float] = Field(default=None)
    freeCashFlowYieldTTM: Optional[float] = Field(default=None)
    debtToEquityTTM: Optional[float] = Field(default=None)
    debtToAssetsTTM: Optional[float] = Field(default=None)
    netDebtToEBITDATTM: Optional[float] = Field(default=None)
    currentRatioTTM: Optional[float] = Field(default=None)
    interestCoverageTTM: Optional[float] = Field(default=None)
    incomeQualityTTM: Optional[float] = Field(default=None)
    dividendYieldTTM: Optional[float] = Field(default=None)
    dividendYieldPercentageTTM: Optional[float] = Field(default=None) # FMP có thể có nhiều tên
    payoutRatioTTM: Optional[float] = Field(default=None)
    salesGeneralAndAdministrativeToRevenueTTM: Optional[float] = Field(default=None)
    researchAndDevelopementToRevenueTTM: Optional[float] = Field(default=None)
    intangiblesToTotalAssetsTTM: Optional[float] = Field(default=None)
    capexToOperatingCashFlowTTM: Optional[float] = Field(default=None)
    capexToRevenueTTM: Optional[float] = Field(default=None)
    capexToDepreciationTTM: Optional[float] = Field(default=None)
    stockBasedCompensationToRevenueTTM: Optional[float] = Field(default=None)
    grahamNumberTTM: Optional[float] = Field(default=None)
    roicTTM: Optional[float] = Field(default=None) # Return on Invested Capital
    returnOnTangibleAssetsTTM: Optional[float] = Field(default=None)
    grahamNetNetTTM: Optional[float] = Field(default=None)
    workingCapitalTTM: Optional[float] = Field(default=None)
    tangibleAssetValueTTM: Optional[float] = Field(default=None)
    netCurrentAssetValueTTM: Optional[float] = Field(default=None)
    investedCapitalTTM: Optional[float] = Field(default=None)
    averageReceivablesTTM: Optional[float] = Field(default=None)
    averagePayablesTTM: Optional[float] = Field(default=None)
    averageInventoryTTM: Optional[float] = Field(default=None)
    daysSalesOutstandingTTM: Optional[float] = Field(default=None)
    daysPayablesOutstandingTTM: Optional[float] = Field(default=None)
    daysOfInventoryOnHandTTM: Optional[float] = Field(default=None)
    receivablesTurnoverTTM: Optional[float] = Field(default=None)
    payablesTurnoverTTM: Optional[float] = Field(default=None)
    inventoryTurnoverTTM: Optional[float] = Field(default=None)
    roeTTM: Optional[float] = Field(default=None) # Return on Equity
    capexPerShareTTM: Optional[float] = Field(default=None)

class FinancialRatiosItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    date: Optional[str] = None
    fiscalYear: Optional[str] = None
    period: Optional[str] = None
    reportedCurrency: Optional[str] = None

    grossProfitMargin: Optional[float] = None
    ebitMargin: Optional[float] = None
    ebitdaMargin: Optional[float] = None
    operatingProfitMargin: Optional[float] = None
    netProfitMargin: Optional[float] = None
    assetTurnover: Optional[float] = None
    currentRatio: Optional[float] = None
    quickRatio: Optional[float] = None
    cashRatio: Optional[float] = None
    priceToEarningsRatio: Optional[float] = None
    priceToBookRatio: Optional[float] = None
    priceToSalesRatio: Optional[float] = None
    priceToFreeCashFlowRatio: Optional[float] = None
    debtToAssetsRatio: Optional[float] = None
    debtToEquityRatio: Optional[float] = None
    financialLeverageRatio: Optional[float] = None
    operatingCashFlowRatio: Optional[float] = None
    dividendPayoutRatio: Optional[float] = None
    dividendYield: Optional[float] = None
    enterpriseValueMultiple: Optional[float] = None

class FinancialStatementGrowthItem(BaseModel):
    # date: Optional[date] = None
    symbol: Optional[str] = Field(None)
    calendarYear: Optional[str] = Field(None)
    period: Optional[str] = Field(None, description="Kỳ báo cáo (FY, Q1, Q2, Q3, Q4)")
    revenueGrowth: Optional[float] = None
    grossProfitGrowth: Optional[float] = None
    ebitgrowth: Optional[float] = None 
    operatingIncomeGrowth: Optional[float] = None
    netIncomeGrowth: Optional[float] = None
    epsgrowth: Optional[float] = Field(None, alias="epsGrowth") 
    epsdilutedGrowth: Optional[float] = Field(None, alias="epsDilutedGrowth")
    weightedAverageSharesGrowth: Optional[float] = None
    weightedAverageSharesDilutedGrowth: Optional[float] = None
    dividendsperShareGrowth: Optional[float] = None
    operatingCashFlowGrowth: Optional[float] = None
    freeCashFlowGrowth: Optional[float] = None
    # tenKfillingDate: Optional[date] = Field(default = None, alias="10KFillingDate") 
    receivablesGrowth: Optional[float] = None
    inventoryGrowth: Optional[float] = None
    assetGrowth: Optional[float] = None
    bookValueperShareGrowth: Optional[float] = None
    debtGrowth: Optional[float] = None
    rdexpenseGrowth: Optional[float] = Field(None, alias="rdExpenseGrowth") 
    sgaexpensesGrowth: Optional[float] = Field(None, alias="sgaExpenseGrowth") 

# Model chung cho một mục trong báo cáo tài chính
class FinancialStatementItem(BaseModel):
    date: Union[date, str]
    reportedCurrency: Optional[str] = None
    cik: Optional[str] = None
    fillingDate: Optional[Union[date, str]] = None 
    acceptedDate: Optional[Union[datetime, str]] = None 
    calendarYear: Optional[str] = None
    period: Optional[str] = None 
    link: Optional[str] = None
    finalLink: Optional[str] = None

    class Config:
        extra = "allow"


# Model cho Báo cáo kết quả kinh doanh (Income Statement)
class IncomeStatement(FinancialStatementItem):
    revenue: Optional[float] = None
    costOfRevenue: Optional[float] = None
    grossProfit: Optional[float] = None
    grossProfitRatio: Optional[float] = None
    researchAndDevelopmentExpenses: Optional[float] = None
    generalAndAdministrativeExpenses: Optional[float] = None
    sellingAndMarketingExpenses: Optional[float] = None
    sellingGeneralAndAdministrativeExpenses: Optional[float] = None # Tổng của S&M và G&A
    otherExpenses: Optional[float] = None
    operatingExpenses: Optional[float] = None
    costAndExpenses: Optional[float] = None
    interestIncome: Optional[float] = None
    interestExpense: Optional[float] = None
    depreciationAndAmortization: Optional[float] = None
    ebitda: Optional[float] = None
    ebitdaratio: Optional[float] = None
    operatingIncome: Optional[float] = None # Còn gọi là EBIT
    operatingIncomeRatio: Optional[float] = None
    totalOtherIncomeExpensesNet: Optional[float] = None
    incomeBeforeTax: Optional[float] = None
    incomeBeforeTaxRatio: Optional[float] = None
    incomeTaxExpense: Optional[float] = None
    netIncome: Optional[float] = None
    netIncomeRatio: Optional[float] = None
    eps: Optional[float] = None
    epsdiluted: Optional[float] = Field(None, alias="epsDiluted") 
    weightedAverageShsOut: Optional[float] = None
    weightedAverageShsOutDil: Optional[float] = None

    class Config:
        populate_by_name = True 


# Model cho Bảng cân đối kế toán (Balance Sheet)
class BalanceSheetStatement(FinancialStatementItem):
    cashAndCashEquivalents: Optional[float] = None
    shortTermInvestments: Optional[float] = None
    cashAndShortTermInvestments: Optional[float] = None
    netReceivables: Optional[float] = None
    inventory: Optional[float] = None
    otherCurrentAssets: Optional[float] = None
    totalCurrentAssets: Optional[float] = None
    propertyPlantEquipmentNet: Optional[float] = None
    goodwill: Optional[float] = None
    intangibleAssets: Optional[float] = None
    goodwillAndIntangibleAssets: Optional[float] = None
    longTermInvestments: Optional[float] = None
    taxAssets: Optional[float] = None
    otherNonCurrentAssets: Optional[float] = None
    totalNonCurrentAssets: Optional[float] = None
    otherAssets: Optional[float] = None 
    totalAssets: Optional[float] = None
    accountPayables: Optional[float] = None
    shortTermDebt: Optional[float] = None
    taxPayables: Optional[float] = None
    deferredRevenue: Optional[float] = None
    otherCurrentLiabilities: Optional[float] = None
    totalCurrentLiabilities: Optional[float] = None
    longTermDebt: Optional[float] = None
    deferredRevenueNonCurrent: Optional[float] = None
    deferredTaxLiabilitiesNonCurrent: Optional[float] = None
    otherNonCurrentLiabilities: Optional[float] = None
    totalNonCurrentLiabilities: Optional[float] = None
    otherLiabilities: Optional[float] = None
    capitalLeaseObligations: Optional[float] = None 
    totalLiabilities: Optional[float] = None
    preferredStock: Optional[float] = None
    commonStock: Optional[float] = None
    retainedEarnings: Optional[float] = None
    accumulatedOtherComprehensiveIncomeLoss: Optional[float] = None
    othertotalStockholdersEquity: Optional[float] = None 
    totalStockholdersEquity: Optional[float] = None
    totalEquity: Optional[float] = None 
    totalLiabilitiesAndStockholdersEquity: Optional[float] = None
    minorityInterest: Optional[float] = None 
    totalLiabilitiesAndTotalEquity: Optional[float] = None 
    totalInvestments: Optional[float] = None
    totalDebt: Optional[float] = None
    netDebt: Optional[float] = None


# Model cho Báo cáo lưu chuyển tiền tệ (Cash Flow Statement)
class CashFlowStatement(FinancialStatementItem):
    netIncome: Optional[float] = None
    depreciationAndAmortization: Optional[float] = None
    deferredIncomeTax: Optional[float] = None
    stockBasedCompensation: Optional[float] = None
    changeInWorkingCapital: Optional[float] = None
    accountsReceivables: Optional[float] = None 
    inventory: Optional[float] = None
    accountsPayables: Optional[float] = None 
    otherWorkingCapital: Optional[float] = None
    otherNonCashItems: Optional[float] = None
    netCashProvidedByOperatingActivities: Optional[float] = None 
    investmentsInPropertyPlantAndEquipment: Optional[float] = None
    acquisitionsNet: Optional[float] = None
    purchasesOfInvestments: Optional[float] = None
    salesMaturitiesOfInvestments: Optional[float] = None
    otherInvestingActivities: Optional[float] = None
    netCashUsedForInvestingActivities: Optional[float] = None 
    debtRepayment: Optional[float] = None
    commonStockIssued: Optional[float] = None
    commonStockRepurchased: Optional[float] = None
    dividendsPaid: Optional[float] = None
    otherFinancingActivities: Optional[float] = None
    netCashUsedProvidedByFinancingActivities: Optional[float] = None 
    effectOfForexChangesOnCash: Optional[float] = None
    netChangeInCash: Optional[float] = None
    cashAtEndOfPeriod: Optional[float] = None
    cashAtBeginningOfPeriod: Optional[float] = None
    operatingCashFlow: Optional[float] = None 
    capitalExpenditure: Optional[float] = None 
    freeCashFlow: Optional[float] = None 


# Model tổng hợp cho tất cả các báo cáo tài chính của một công ty
class FinancialStatementsData(BaseModel):
    symbol: str
    income_statements_annual: List[IncomeStatement] = Field(default_factory=list)
    income_statements_quarterly: List[IncomeStatement] = Field(default_factory=list)
    balance_sheets_annual: List[BalanceSheetStatement] = Field(default_factory=list)
    balance_sheets_quarterly: List[BalanceSheetStatement] = Field(default_factory=list)
    cash_flow_statements_annual: List[CashFlowStatement] = Field(default_factory=list)
    cash_flow_statements_quarterly: List[CashFlowStatement] = Field(default_factory=list)



class LogoData(BaseModel):
    logo_url: str = Field(..., description="URL của logo")

class ClientSymbolRequest(BaseModel):
    symbols: str = Field(..., description="Danh sách các mã, cách nhau bởi dấu phẩy, ví dụ: AAPL,MSFT")

class EquityDetailWsParams(BaseModel):
    symbol: str
    timeframe: str
    from_date: str 
    to_date: str  

class ListPageWsParams(BaseModel):
    """
    Tham số phân trang cho các danh sách.
    """
    page: int = Field(1, description="Số trang, bắt đầu từ 1.")
    limit: int = Field(20, description="Số lượng kết quả mỗi trang (1-100).")

class DiscoveryWsParams(BaseModel):
    """
    Tham số cho các danh sách discovery (top gainers, losers, actives).
    """
    limit: int = Field(10, description="Số lượng kết quả trả về.")

class SocialSentimentItem(BaseModel):
    date: Optional[datetime] = Field(None, description="Ngày giờ của dữ liệu cảm xúc")
    symbol: Optional[str] = Field(None, description="Mã cổ phiếu")
    stocktwitsPosts: Optional[int] = Field(None, description="Số lượng bài đăng trên Stocktwits")
    twitterPosts: Optional[int] = Field(None, description="Số lượng bài đăng trên Twitter/X")
    stocktwitsComments: Optional[int] = Field(None, description="Số lượng bình luận trên Stocktwits")
    twitterComments: Optional[int] = Field(None, description="Số lượng bình luận trên Twitter/X")
    stocktwitsLikes: Optional[int] = Field(None, description="Số lượng lượt thích trên Stocktwits")
    twitterLikes: Optional[int] = Field(None, description="Số lượng lượt thích trên Twitter/X")
    stocktwitsImpressions: Optional[int] = Field(None, description="Số lượt hiển thị trên Stocktwits")
    twitterImpressions: Optional[int] = Field(None, description="Số lượt hiển thị trên Twitter/X")
    stocktwitsSentiment: Optional[float] = Field(None, description="Điểm cảm xúc trên Stocktwits")
    twitterSentiment: Optional[float] = Field(None, description="Điểm cảm xúc trên Twitter/X")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

    @field_validator('date', mode='before')
    @classmethod
    def parse_date(cls, value: Any) -> Optional[datetime]:
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    if value.endswith('Z'):
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return datetime.fromisoformat(value)
                except ValueError:
                    return None
        elif isinstance(value, datetime):
            return value
        return None

class StockNewsSentimentItem(BaseModel):
    symbol: Optional[str] = Field(None, description="Mã cổ phiếu liên quan đến tin tức")
    publishedDate: Optional[datetime] = Field(None, description="Ngày giờ tin tức được công bố")
    title: Optional[str] = Field(None, description="Tiêu đề của bài viết")
    image: Optional[str] = Field(None, description="URL hình ảnh minh họa cho bài viết")
    site: Optional[str] = Field(None, description="Tên website nguồn tin")
    text: Optional[str] = Field(None, description="Một đoạn trích hoặc nội dung chính của bài viết")
    url: Optional[str] = Field(None, description="URL đầy đủ của bài viết gốc")
    sentiment: Optional[str] = Field(None, description="Phân tích cảm xúc của tin tức (ví dụ: Positive, Negative, Neutral)")
    sentimentScore: Optional[float] = Field(None, description="Điểm số cảm xúc, thường từ -1 đến 1 hoặc 0 đến 1")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

    @field_validator('publishedDate', mode='before')
    @classmethod
    def parse_published_date(cls, value: Any) -> Optional[datetime]:
        if isinstance(value, str):
            try:
                if value.endswith('Z'):
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                return datetime.fromisoformat(value)
            except ValueError:
                logger.warning(f"Could not parse publishedDate string: {value}")
                return None
        elif isinstance(value, datetime):
            return value
        return None

#Model trung gian cho bước 1 Screener
class ScreenerStep1Data(BaseModel):
    symbol: str
    companyName: Optional[str] = None 
    sector: Optional[str] = None
    beta: Optional[float] = None
    lastAnnualDividend: Optional[float] = None

class ScreenerWSParams(BaseModel):
    limit: int = Field(100, ge=1, le=1000)
    market_cap_more_than: Optional[float] = Field(None, alias="marketCapMoreThan")
    market_cap_lower_than: Optional[float] = Field(None, alias="marketCapLowerThan")
    price_more_than: Optional[float] = Field(None, alias="priceMoreThan")
    price_lower_than: Optional[float] = Field(None, alias="priceLowerThan")
    beta_more_than: Optional[float] = Field(None, alias="betaMoreThan")
    beta_lower_than: Optional[float] = Field(None, alias="betaLowerThan")
    volume_more_than: Optional[float] = Field(None, alias="volumeMoreThan")
    volume_lower_than: Optional[float] = Field(None, alias="volumeLowerThan")
    dividend_more_than: Optional[float] = Field(None, alias="dividendMoreThan")
    dividend_lower_than: Optional[float] = Field(None, alias="dividendLowerThan")
    is_etf: Optional[bool] = Field(None, alias="isEtf")
    is_fund: Optional[bool] = Field(None, alias="isFund")
    is_actively_trading: Optional[bool] = Field(True, alias="isActivelyTrading")
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None

    class Config:
        populate_by_name = True

    def get_active_filters_for_python_call(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=False) 

class ScreenerOutput(BaseModel):
    symbol: str = Field(..., description="Mã cổ phiếu")
    name: Optional[str] = Field(None, description="Tên công ty")
    price: Optional[float] = Field(None, description="Giá hiện tại")
    changesPercentage: Optional[float] = Field(None, description="Phần trăm thay đổi giá trong ngày")
    change: Optional[float] = Field(None, description="Mức thay đổi giá trong ngày")
    dayLow: Optional[float] = Field(None, description="Giá thấp nhất trong ngày")
    dayHigh: Optional[float] = Field(None, description="Giá cao nhất trong ngày")
    yearHigh: Optional[float] = Field(None, description="Giá cao nhất trong 52 tuần")
    yearLow: Optional[float] = Field(None, description="Giá thấp nhất trong 52 tuần")
    marketCap: Optional[float] = Field(None, description="Vốn hóa thị trường")
    priceAvg50: Optional[float] = Field(None, description="Giá trung bình 50 ngày")
    priceAvg200: Optional[float] = Field(None, description="Giá trung bình 200 ngày")
    exchange: Optional[str] = Field(None, description="Sàn giao dịch")
    volume: Optional[float] = Field(None, description="Khối lượng giao dịch trong ngày")
    avgVolume: Optional[float] = Field(None, description="Khối lượng giao dịch trung bình")
    open_price: Optional[float] = Field(None, alias="open", description="Giá mở cửa")
    previousClose: Optional[float] = Field(None, description="Giá đóng cửa ngày hôm trước")
    eps: Optional[float] = Field(None, description="Thu nhập trên mỗi cổ phiếu (EPS)")
    pe: Optional[float] = Field(None, description="Chỉ số P/E")
    earningsAnnouncement: Optional[str] = Field(None, description="Ngày công bố báo cáo thu nhập (dưới dạng chuỗi)")
    sharesOutstanding: Optional[float] = Field(None, description="Số lượng cổ phiếu đang lưu hành")
    timestamp: Optional[int] = Field(None, description="Dấu thời gian Unix của dữ liệu")
    beta : Optional[float] = Field(None, description="Hệ số beta của cổ phiếu (đo lường rủi ro so với thị trường)")
    sector : Optional[str] = Field(None, description="Ngành của công ty")
    beta: Optional[float] = Field(None, description="Hệ số beta")
    sector: Optional[str] = Field(None, description="Ngành")
    dividend: Optional[float] = Field(None, alias="lastAnnualDividend", description="Cổ tức hàng năm (từ lastAnnualDividend của FMP)")
    industry: Optional[str] = Field(None, description="Ngành công nghiệp của công ty")

    class Config:
        populate_by_name = True 

class HistoricalDataItem(BaseModel):
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None 
    change: Optional[float] = Field(None, description="Tính toán: close - open của mỗi thanh nến")
    change_percent: Optional[float] = Field(None, description="Tính toán: (change / open) * 100 của mỗi thanh nến")
    ma5: Optional[float] = Field(None, description="Đường trung bình động 5 kỳ (MA5)")
    ma20: Optional[float] = Field(None, description="Đường trung bình động 20 kỳ (MA20)")
    rsi: Optional[float] = Field(None, description="Chỉ số sức mạnh tương đối (RSI) 14 kỳ")
    macd: Optional[MACDOutput] = Field(None, description="Dữ liệu chỉ báo MACD (12, 26, 9)")

class CryptoNewsItem(BaseModel):
    publishedDate: Optional[datetime] = None
    title: Optional[str] = None
    image: Optional[str] = None
    site: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None
    tickers: Optional[List[str]] = Field(default_factory=list)
    updatedAt: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    type: Optional[str] = None

class SenateTradingItem(BaseModel):
    symbol: Optional[str] = None
    disclosure_date: date = Field(..., alias="disclosureDate")
    transaction_date: date = Field(..., alias="transactionDate")
    first_name: str = Field(..., alias="firstName")
    last_name: str = Field(..., alias="lastName")
    office: str
    owner: str
    asset_description: str = Field(..., alias="assetDescription")
    asset_type: str = Field(..., alias="assetType")
    transaction_type: str = Field(..., alias="type")
    amount: str
    comment: str
    link: str

    class Config:
        populate_by_name = True

class StockDetailPayload(BaseModel):
    # --- Dữ liệu từ API Quote (v3/quote) ---
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
    earningsAnnouncement: Optional[str] = None
    sharesOutstanding: Optional[float] = None
    timestamp: Optional[int] = None

    # --- Dữ liệu từ API Historical Chart (v3/historical-chart) ---
    data: List[HistoricalDataItem] = Field(default_factory=list, description="Dữ liệu biểu đồ lịch sử")
    detected_patterns: Optional[List[DetectedPattern]] = Field(None, description="Danh sách các mẫu hình giá được phát hiện")

    # --- Dữ liệu từ API Key Metrics TTM (v3/key-metrics-ttm) ---
    revenue_per_share_ttm: Optional[float] = Field(None, alias="revenuePerShareTTM")
    net_income_per_share_ttm: Optional[float] = Field(None, alias="netIncomePerShareTTM")
    operating_cash_flow_per_share_ttm: Optional[float] = Field(None, alias="operatingCashFlowPerShareTTM")
    free_cash_flow_per_share_ttm: Optional[float] = Field(None, alias="freeCashFlowPerShareTTM")
    cash_per_share_ttm: Optional[float] = Field(None, alias="cashPerShareTTM")
    book_value_per_share_ttm: Optional[float] = Field(None, alias="bookValuePerShareTTM")
    enterprise_value_ttm: Optional[float] = Field(None, alias="enterpriseValueTTM")
    pe_ratio_ttm: Optional[float] = Field(None, alias="peRatioTTM")
    price_to_sales_ratio_ttm: Optional[float] = Field(None, alias="priceToSalesRatioTTM")
    pocfratio_ttm: Optional[float] = Field(None, alias="pocfratioTTM")
    pfcf_ratio_ttm: Optional[float] = Field(None, alias="pfcfRatioTTM")
    pb_ratio_ttm: Optional[float] = Field(None, alias="pbRatioTTM")
    ev_to_sales_ttm: Optional[float] = Field(None, alias="evToSalesTTM")
    enterprise_value_over_ebitda_ttm: Optional[float] = Field(None, alias="enterpriseValueOverEBITDATTM")
    debt_to_equity_ttm: Optional[float] = Field(None, alias="debtToEquityTTM")
    debt_to_assets_ttm: Optional[float] = Field(None, alias="debtToAssetsTTM")
    net_debt_to_ebitda_ttm: Optional[float] = Field(None, alias="netDebtToEBITDATTM")
    current_ratio_ttm: Optional[float] = Field(None, alias="currentRatioTTM")
    interest_coverage_ttm: Optional[float] = Field(None, alias="interestCoverageTTM")
    dividend_yield_ttm: Optional[float] = Field(None, alias="dividendYieldTTM")
    payout_ratio_ttm: Optional[float] = Field(None, alias="payoutRatioTTM")
    roic_ttm: Optional[float] = Field(None, alias="roicTTM")
    roe_ttm: Optional[float] = Field(None, alias="roeTTM")
    
    # --- Dữ liệu từ API Ratios TTM (v3/ratios-ttm) ---
    net_profit_margin_ttm: Optional[float] = Field(None, alias="netProfitMarginTTM")
    gross_profit_margin_ttm: Optional[float] = Field(None, alias="grossProfitMarginTTM")
    operating_profit_margin_ttm: Optional[float] = Field(None, alias="operatingProfitMarginTTM")
    return_on_assets_ttm: Optional[float] = Field(None, alias="returnOnAssetsTTM")
    cash_flow_to_debt_ratio_ttm: Optional[float] = Field(None, alias="cashFlowToDebtRatioTTM")
    asset_turnover_ttm: Optional[float] = Field(None, alias="assetTurnoverTTM")
    price_to_book_ratio_ttm: Optional[float] = Field(None, alias="priceToBookRatioTTM")
    price_to_sales_ratio_ttm_2: Optional[float] = Field(None, alias="priceSalesRatioTTM") 
    enterprise_value_multiple_ttm: Optional[float] = Field(None, alias="enterpriseValueMultipleTTM")
    detected_patterns: List[DetectedPattern] = Field([], description="Danh sách các mẫu hình giá được phát hiện")
    class Config:
        populate_by_name = True

class CompanyProfile(BaseModel):
    # core
    symbol: str
    price: Optional[float] = None
    marketCap: Optional[int] = None
    beta: Optional[float] = None
    lastDividend: Optional[float] = None
    range: Optional[str] = None
    change: Optional[float] = None
    changePercentage: Optional[float] = Field(None, description="Number (e.g., 2.1008 = 2.1008%)")

    # volumes
    volume: Optional[int] = None
    averageVolume: Optional[int] = Field(None, alias="averageVolume")

    # company meta
    companyName: Optional[str] = None
    currency: Optional[str] = None
    cik: Optional[str] = None
    isin: Optional[str] = None
    cusip: Optional[str] = None

    # exchanges
    exchangeFullName: Optional[str] = None
    exchange: Optional[str] = None

    # biz info
    industry: Optional[str] = None
    sector: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    ceo: Optional[str] = None
    country: Optional[str] = None
    fullTimeEmployees: Optional[Union[int, str]] = None  # FMP đôi khi trả "164000" dạng string

    # contact / address
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None

    # media & dates
    image: Optional[str] = None
    ipoDate: Optional[str] = None     # FMP trả "YYYY-MM-DD" (string). Để string để giữ đúng format

    # flags
    defaultImage: Optional[bool] = None
    isEtf: Optional[bool] = None
    isActivelyTrading: Optional[bool] = None
    isAdr: Optional[bool] = None
    isFund: Optional[bool] = None

    class Config:
        populate_by_name = True
        str_strip_whitespace = True
        

class FMPSearchResultItem(BaseModel):
    symbol: str
    name: Optional[str] = None
    currency: Optional[str] = None
    stockExchange: Optional[str] = None  
    exchangeShortName: Optional[str] = None 

class CryptoSpotlightItem(BaseModel):
    trend: Optional[str] = Field(None, description="Xu hướng giá (up, down, neutral)")
    symbol: str = Field(..., description="Mã tiền điện tử (ví dụ: BTCUSD)")
    name: Optional[str] = Field(None, description="Tên tiền điện tử")
    logo_url: Optional[str] = Field(None, description="URL logo của tiền điện tử")
    price: Optional[float] = Field(None, description="Giá hiện tại")
    change: Optional[float] = Field(None, description="Mức thay đổi giá")
    percent_change: Optional[float] = Field(None, description="Phần trăm thay đổi giá")
    volume: Optional[float] = Field(None, description="Khối lượng giao dịch")
    market_cap: Optional[float] = Field(None, description="Vốn hóa thị trường")

    chartData: List[ChartDataItem] = Field(default_factory=list, description="Dữ liệu biểu đồ lịch sử")

class FMPCompanyOutlookProfile(BaseModel):
    symbol: Optional[str] = Field(None, description="Mã cổ phiếu")
    symbol_name: Optional[str] = Field(None, description="Tên mã cổ phiếu")
    
    name: Optional[str] = Field(None, description="Tên công ty")
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    enterprise_to_ebitda: Optional[float] = None
    shares_outstanding: Optional[float] = None
    employees: Optional[int] = None
    shareholders: Optional[float] = Field(None, description="FMP không cung cấp trực tiếp")
    
    pe_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    
    roa: Optional[float] = None
    roe: Optional[float] = None
    return_invested_capital: Optional[float] = None
    revenue_per_employee: Optional[float] = None
    
    quick_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    net_debt: Optional[float] = None
    total_debt: Optional[float] = None 
    total_assets: Optional[float] = None 
    
    volume_average_10d: Optional[float] = None
    beta: Optional[float] = None
    year_high: Optional[float] = None
    year_low: Optional[float] = None
    
    dividends_paid: Optional[float] = Field(None, description="Tổng cổ tức đã trả TTM (tính toán)")
    dividend_yield: Optional[float] = None 
    dividends_per_share: Optional[float] = None 
    
    net_margin: Optional[float] = None 
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None 
    pretax_margin: Optional[float] = None 
    
    basic_eps_fy: Optional[float] = None 
    eps: Optional[float] = None 
    eps_diluted: Optional[float] = None 
    net_income: Optional[float] = None 
    ebitda: Optional[float] = None 
    gross_profit_mrq: Optional[float] = None 
    gross_profit: Optional[float] = None 
    last_year_revenue: Optional[float] = None 
    revenue: Optional[float] = None 
    free_cash_flow: Optional[float] = None 

    class Config:
        populate_by_name = True 

class ProfileResponseWrapper(BaseModel, Generic[DataType]): 
    symbol: str
    symbol_name: Optional[str] = None
    data: List[DataType] 

class TickerTapeData(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    change: Optional[float] = None
    percent_change: Optional[float] = None

    chartData: List[ChartDataItem] = Field(default_factory=list, description="Dữ liệu biểu đồ lịch sử")

class DiscoveryItemOutput(BaseModel): 
    symbol: str
    name: Optional[str] = None
    url_logo: Optional[str] = None
    event_catalyst: Optional[str] = Field(None, description="Nguồn dữ liệu cho trường này chưa xác định, mặc định là None")
    price: Optional[float] = None
    change: Optional[float] = None
    percent_change: Optional[float] = None
    volume: Optional[float] = None

    high_24h: Optional[float] = Field(None, description="Giá cao nhất trong phiên (dayHigh)")
    low_24h: Optional[float] = Field(None, description="Giá thấp nhất trong phiên (dayLow)")
    volume_24h: Optional[float] = Field(None, description="Khối lượng giao dịch phiên hiện tại")
    market_cap: Optional[float] = Field(None, description="Vốn hóa thị trường hiện tại")
    volume_change_24h: Optional[float] = Field(None, description="Thay đổi volume so với phiên trước")
    market_cap_change_24h: Optional[float] = Field(None, description="Thay đổi market cap so với ngày trước")
    
    chartData: List[ChartDataItem] = Field(default_factory=list, description="Dữ liệu biểu đồ lịch sử cho mã")
    detected_patterns: Optional[List[DetectedPattern]] = Field(None, description="Danh sách các mẫu hình giá được phát hiện")
    strongBuy: Optional[int] = Field(default=None, description="Số lượng khuyến nghị Strong Buy")
    buy: Optional[int] = Field(default=None, description="Số lượng khuyến nghị Buy")
    hold: Optional[int] = Field(default=None, description="Số lượng khuyến nghị Hold")
    sell: Optional[int] = Field(default=None, description="Số lượng khuyến nghị Sell")
    strongSell: Optional[int] = Field(default=None, description="Số lượng khuyến nghị Strong Sell")
    consensus: Optional[str] = Field(default=None, description="Khuyến nghị đồng thuận (Buy, Hold, Sell, ...)")
    @validator('price', 'change', 'percent_change', 'volume', 'high_24h', 'low_24h', 'volume_24h', 'market_cap', 
               'volume_change_24h', 'market_cap_change_24h', pre=True)
    def validate_optional_floats(cls, v):
        if v == "":
            return None
        return v

class HistoricalPriceItem(BaseModel):
    date: Union[datetime, str] 
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[Union[int, float]] = None
    vwap: Optional[float] = None 
    change: Optional[float] = None
    percent_change: Optional[float] = None

class CompanyProfileData(BaseModel):
    symbol: str
    name: Optional[str] = None
    description: Optional[str] = None
    country: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    price_to_sales: Optional[float] = None
    price_to_book: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    total_revenue_fy: Optional[float] = None
    net_income_fy: Optional[float] = None
    total_assets_mrq: Optional[float] = None
    total_debt_mrq: Optional[float] = None

class CompanyDescription(BaseModel):
    """Model để chỉ chứa mô tả của công ty."""
    symbol: str
    description: Optional[str] = Field(None, description="Mô tả chi tiết về công ty")

class RegionalScreenerItem(BaseModel):
    symbol: str
    companyName: Optional[str] = Field(default=None)
    marketCap: Optional[float] = Field(default=None)
    sector: Optional[str] = Field(default=None)
    industry: Optional[str] = Field(default=None)
    beta: Optional[float] = Field(default=None)
    price: Optional[float] = Field(default=None)
    lastAnnualDividend: Optional[float] = Field(default=None)
    volume: Optional[float] = Field(default=None)
    exchange: Optional[str] = Field(default=None)
    historyPeriodChangePercent: Optional[float] = Field(default=None, description="Phần trăm thay đổi giá giữa ngày đầu và ngày cuối trong dữ liệu lịch sử (5 ngày)")

    history: List[HistoricalPriceItem] = Field(default_factory=list, description="Lịch sử giá trong 5 ngày gần nhất")

    model_config = ConfigDict(
        populate_by_name=True,
    )

class MarketOverviewData(BaseModel): 
    symbol: str
    name: Optional[str] = Field(None, description="Tên đầy đủ của tài sản")
    short_name: Optional[str] = Field(None, description="Tên viết tắt/ngắn gọn của tài sản")
    asset_type: Optional[str] = Field(None, description="Loại tài sản (ví dụ: STOCK, ETF, INDEX, CRYPTO)")
    
    exchange: Optional[str] = Field(None, description="Sàn giao dịch")
    currency: Optional[str] = Field(None, description="Đơn vị tiền tệ")

    last_price: Optional[float] = Field(None, description="Giá khớp lệnh cuối cùng")
    open_price: Optional[float] = Field(None, alias="open", description="Giá mở cửa trong ngày")
    day_high: Optional[float] = Field(None, alias="high", description="Giá cao nhất trong ngày")
    day_low: Optional[float] = Field(None, alias="low", description="Giá thấp nhất trong ngày")
    previous_close_price: Optional[float] = Field(None, alias="prev_close", description="Giá đóng cửa ngày hôm trước")
    
    change: Optional[float] = Field(None, description="Mức thay đổi giá so với giá đóng cửa hôm trước")
    change_percent: Optional[float] = Field(None, description="Phần trăm thay đổi giá")
    
    bid: Optional[float] = Field(None, description="Giá chào mua tốt nhất")
    bid_size: Optional[int] = Field(None, description="Khối lượng chào mua tốt nhất")
    ask: Optional[float] = Field(None, description="Giá chào bán tốt nhất")
    ask_size: Optional[int] = Field(None, description="Khối lượng chào bán tốt nhất")
    
    volume: Optional[Union[int, float]] = Field(None, description="Khối lượng giao dịch trong ngày")
    volume_average: Optional[Union[int, float]] = Field(None, description="Khối lượng giao dịch trung bình (thường là 30 ngày)")
    volume_average_10d: Optional[Union[int, float]] = Field(None, description="Khối lượng giao dịch trung bình 10 ngày")

    year_high: Optional[float] = Field(None, description="Giá cao nhất trong 52 tuần")
    year_low: Optional[float] = Field(None, description="Giá thấp nhất trong 52 tuần")
    
    ma_50d: Optional[float] = Field(None, description="Đường trung bình động 50 ngày")
    ma_200d: Optional[float] = Field(None, description="Đường trung bình động 200 ngày")

    class Config:
        populate_by_name = True 

class SymbolReferenceItem(BaseModel): 
    
    trend: Optional[str] = Field(None, description="Xu hướng cổ phiếu: 'up', 'down', hoặc 'neutral'")
    symbol: str = Field(..., description="Mã cổ phiếu")
    name: Optional[str] = Field(None, description="Tên công ty")
    symbol_url: Optional[str] = Field(None, description="URL tới trang chi tiết về mã")
    price: Optional[float] = Field(None, description="Giá hiện tại hoặc mới nhất")
    change: Optional[float] = Field(None, description="Giá trị thay đổi")
    percent_change: Optional[float] = Field(None, alias="percentChange", description="Phần trăm thay đổi giá")
    volume: Optional[float] = Field(None, description="Khối lượng giao dịch")
    chartData: List[ChartDataItem] = Field(default_factory=list, description="Dữ liệu biểu đồ lịch sử cho mã")

    class Config:
        populate_by_name = True 
        

class FMPGainerItem(BaseModel): 
    symbol: str
    name: Optional[str] = None
    change: Optional[float] = None
    price: Optional[float] = None
    changesPercentage: Optional[float] = None

class NewsItemOutput(BaseModel):
    type: Optional[str] = Field(None, description="Loại tin tức (ví dụ: LATEST_GENERAL, LATEST_COMPANY, TRENDING)")
    category: Optional[str] = Field(None, description="Danh mục tin tức (ví dụ: GENERAL, COMPANY_SPECIFIC, MARKET, TECHNOLOGY)")
    title: str = Field(..., description="Tiêu đề tin tức")
    description: Optional[str] = Field(None, description="Mô tả/nội dung tóm tắt")
    news_url: str = Field(..., description="URL của bài viết")
    image_url: Optional[str] = Field(None, description="URL hình ảnh minh họa")
    is_importance: Optional[str] = Field(None, description="Cờ đánh dấu mức độ quan trọng (FMP không cung cấp, mặc định là None)")
    date: str = Field(..., description="Ngày đăng tin (chuỗi ISO hoặc định dạng tương tự từ FMP)")
    source_site: Optional[str] = Field(None, description="Tên website nguồn tin (ví dụ: Zacks, Reuters)")


# ---  Model API Response Generic ---

class APIResponseData(BaseModel, Generic[DataType]):
    data: List[DataType] = Field(..., description="Danh sách các mục dữ liệu theo kiểu đã chỉ định.")

class APIResponse(BaseModel, Generic[DataType]): 
    message: str = Field("OK", description="Thông báo chung cho biết kết quả của yêu cầu.")
    status: str = Field("200", description="Mã trạng thái kiểu HTTP dưới dạng chuỗi.")
    provider_used: Optional[str] = Field(None, description="Nhà cung cấp dữ liệu cuối cùng được sử dụng cho yêu cầu.")
    totalRows: Optional[int] = Field(None, description="Tổng số lượng item có sẵn (dành cho phân trang)")

    data: Optional[APIResponseData[DataType]] = Field(None, description="Payload dữ liệu thực tế, được bao bọc trong một đối tượng APIResponseData.")

    class Config:
        populate_by_name = True 

