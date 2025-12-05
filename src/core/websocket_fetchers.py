# File: src/core/websocket_fetchers.py

import asyncio
import httpx
import logging
from typing import Any, Optional

from src.services.bitcoin_ws_service import BitcoinWsService
from src.services.transaction_decoder_service import TransactionDecoderService
from src.services.alchemy_service import AlchemyWsService
from src.services.regional_screener_service import MarketIndicesService
from src.services.discovery_service import DiscoveryService
from src.services.list_service import ListService
from src.services.equity_detail_service import EquityDetailService
from src.services.ticker_tape_service import TickerTapeService
from src.core.websocket_manager import global_connection_manager
from src.models.equity import APIResponse, APIResponseData, BtcInputOutput, CustomListWsParams, FullOnchainTransaction, MarketRegionEnum, MarketRegionParams, OnchainSubscriptionParams, ParsedBtcTransaction, TickerTapeWSParams, TickerTapeData, StockDetailPayload, DiscoveryItemOutput, EquityDetailWsParams, ListPageWsParams, DiscoveryWsParams

from src.utils.config import settings
import aioredis

logging.getLogger('websockets.protocol').setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
SATOSHIS_IN_BTC = 100_000_000

stock_list = [
        # US Stocks
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", "BRK-A", "BRK-B",
        "V", "JPM", "JNJ", "WMT", "PG", "MA", "HD", "BAC", "KO", "PEP",
        "XOM", "CVX", "LLY", "AVGO", "ORCL", "CRM", "ADBE", "NFLX", "COST", "MCD",
        "DIS", "NKE", "INTC", "CSCO", "PFE", "MRK", "ABBV", "TMO", "ACN", "AMD",
        "TXN", "HON", "UNH", "QCOM", "CAT", "SBUX", "GS", "LOW", "BA", "IBM",
        "GE", "RTX", "BLK", "AXP", "AMGN", "PYPL", "UBER", "ZM", "SQ", "SNAP",
        "SHOP", "SNOW", "RBLX", "SPOT", "COIN", "HOOD", "PLTR", "AI", "SOFI", "RIVN",
        "LCID", "DKNG", "PTON", "GME", "AMC", "BB", "NOK", "SPCE", "CLOV", "WISH",
        # Adding 100 more as requested (first batch)
        "F", "GM", "MAR", "HLT", "BKNG", "ABNB", "DAL", "UAL", "AAL", "LUV", "TGT", "BBY", "KSS", "M", "JWN", "GPS", "EBAY", "ETSY", "CHWY", "W", "FDX", "UPS", "ADP", "PAYX", "FIS", "FISV", "GPN", "FLT", "VRSK", "MSCI", "SPGI", "MCO", "CME", "ICE", "NDAQ", "TWTR", "PINS", "DOCU", "DDOG", "NET", "CRWD", "ZS", "OKTA", "MDB", "TEAM", "WDAY", "NOW", "INTU", "ADSK", "PANW", "FTNT", "VMW", "DELL", "HPQ", "HPE", "STX", "WDC", "MU", "AMAT", "LRCX", "KLAC", "TER", "ADI", "MCHP", "SWKS", "QRVO", "ON", "STM", "NXPI", "INFINEON.DE", "SAP.DE", "SIE.DE", "BAYN.DE", "BMW.DE", "VOW3.DE", "MBG.DE", "AIR.PA", "SAF.PA", "RMS.PA", "KER.PA", "SAN.PA", "BNP.PA", "TTE.PA", "ACA.PA", "ENGI.PA", "STM.PA", "CAP.PA", "UNA.AS", "PHIA.AS", "INGA.AS", "AD.AS", "DSM.AS", "HEIA.AS", "RACE.MI", "ENEL.MI", "ISP.MI", "NESN.VX", "NOVN.VX", "UBSG.VX", "ZURN.VX", "CSGN.VX",
        # Adding another 100 (second batch)
        "RIO.L", "GLEN.L", "AAL.L", "BATS.L", "AZN.L", "GSK.L", "ULVR.L", "DGE.L", "REL.L", "BP.L", "BARC.L", "LLOY.L", "STAN.L", "PRU.L", "NG.L", "VOD.L", "BT-A.L", "TSCO.L", "SBRY.L", "MKS.L", "WPP.L", "ITV.L", "WTB.L", "IHG.L", "AHT.L", "CCL.L", "EZJ.L", "IAG.L", "RR.L", "BA.L", "FERG.L", "CRH.L", "EXPN.L", "SMT.L", "3IN.L", "BDEV.L", "PSN.L", "TW.L", "LAND.L", "BLND.L", "SGRO.L", "LGEN.L", "AV.L", "SLA.L", "HSX.L", "ITRK.L", "KGF.L", "NXT.L", "JD.L", "RKT.L", "ADM.L", "FLTR.L", "ENT.L", "OCDO.L", "DCC.L", "FRES.L", "RMG.L", "SSE.L", "UU.L", "SVT.L", "AUTO.L", "HL.L", "PSON.L", "INF.L", "CCH.L", "SN.L", "SMIN.L", "SPX.L", "ABDN.L", "MNG.L", "PHNX.L", "RSA.L", "STJ.L", "RSA", "STJ", "WEIR.L", "VIO.L", "BNZL.L", "DPH.L", "GVC.L", "SKG.L", "ICP.L", "III.L", "BKG.L", "CNP.AX", "WES.AX", "MQG.AX", "ANZ.AX", "NAB.AX", "WBC.AX", "FMG.AX", "NCM.AX", "ORG.AX", "STO.AX", "WPL.AX", "SCG.AX", "TLS.AX", "RIO.AX", "BHP.AX", "S32.AX", "CSL.AX", "RMD.AX", "COH.AX",
        # Adding another 100 (third batch) - Focusing on Asian and other European/Canadian markets
        "6752.T", "8058.T", "9984.T", "4063.T", "7974.T", "6501.T", "8306.T", "8316.T", "9432.T", "9433.T", "000660.KS", "005380.KS", "035420.KS", "051910.KS", "068270.KS", "005490.KS", "012330.KS", "028050.KS", "006400.KS", "096770.KS", "000880.KS", "032830.KS", "105560.KS", "000720.KS", "017670.KS", "000020.KS", "035720.KS", "018260.KS", "003490.KS", "011170.KS", "0700.HK", "1299.HK", "9988.HK", "3690.HK", "2318.HK", "0941.HK", "1398.HK", "3988.HK", "2628.HK", "0883.HK", "D05.SI", "O39.SI", "U11.SI", "Z74.SI", "C6L.SI", "G13.SI", "BS6.SI", "S68.SI", "Y92.SI", "C09.SI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "MARUTI.NS", "WIPRO.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNQ.TO", "BMO.TO", "TRP.TO", "CP.TO", "CNR.TO", "SU.TO", "MFC.TO", "ATD.TO", "BCE.TO", "SLF.TO", "TRI.TO", "IFC.F", "ADS.DE", "ALV.DE", "BEI.DE", "CON.DE", "DBK.DE", "DPW.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HEI.DE", "HEN3.DE", "LIN.DE", "MRK.DE", "MUV2.DE", "RWE.DE", "VNA.DE",
        # Adding another 100 (fourth batch) - More European, some US mid/small cap, other international
        "BBVA.MC", "SAN.MC", "ITX.MC", "TEF.MC", "REP.MC", "IBE.MC", "AMS.MC", "FER.MC", "AENA.MC", "GRF.MC", "MAP.MC", "NTGY.MC", "ACS.MC", "ELE.MC", "MTS.MC", "SAB.MC", "REE.MC", "CLNX.MC", "MEL.MC", "ANA.MC", "STLAM.MI", "G.MI", "HER.MI", "SRG.MI", "MB.MI", "PIRC.MI", "SPM.MI", "BGN.MI", "PRY.MI", "FBK.MI", "TIT.MI", "BAMI.MI", "NEXI.MI", "MONC.MI", "TEN.MI", "REC.MI", "DIA.MI", "A2A.MI", "AZM.MI", "TRN.MI", "LDO.MI", "JUVE.MI", "ATL.MI", "SFER.MI", "BPE.MI", "ENI.MI", "LKOH.ME", "SBER.ME", "GAZP.ME", "ROSN.ME", "GMKN.ME", "NVTK.ME", "PLZL.ME", "TATN.ME", "MTSS.ME", "YNDX", "POLY.L", "ERIC-B.ST", "VOLV-B.ST", "ESSITY-B.ST", "HM-B.ST", "SEB-A.ST", "SWED-A.ST", "TELIA.ST", "SKA-B.ST", "ALFA.ST", "SAND.ST", "SCV-B.ST", "ATCO-A.ST", "Investor-B.ST", "SHB-A.ST", "NDA-SE.ST", "ABB.ST", "AZN.ST", "BOL.ST", "GETI-B.ST", "KINV-B.ST", "LATO-B.ST", "LUMI.ST", "MILL-SDB.ST", "NIBE-B.ST", "PEAB-B.ST", "SAAB-B.ST", "SKF-B.ST", "SSAB-A.ST", "SWMA.ST", "TREL-B.ST", "WALL-B.ST", "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2881.TW", "2882.TW", "1301.TW", "1303.TW", "1326.TW", "1216.TW", "2002.TW", "2412.TW", "3008.TW", "4904.TW",
        # Adding 200 more (fifth batch) - Broadening to more markets and US smaller caps
        "AFL", "ALL", "AXS", "BEN", "BLK", "CINF", "CMA", "COF", "DFS", "FITB", "GL", "HIG", "IVZ", "KEY", "L", "LNC", "MET", "MKL", "MTB", "NAVI", "NTRS", "PFG", "PGR", "PRU", "RF", "SCHW", "SIVB", "STT", "SYF", "TROW", "TRV", "UNM", "WFC", "WRB", "XEL", "XYL", "YUM", "ZBH", "ZTS", "A", "AAN", "AAP", "ABCB", "ABG", "ABM", "ABT", "ADNT", "ADM", "AEE", "AEP", "AES", "AGCO", "AGNC", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALLE", "ALSN", "ALXN", "AMCR", "AME", "AMG", "AMH", "AMP", "ANET", "ANSS", "ANTM", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ARGX.BR", "ASAI.AS", "ATE.AT", "ATLC.L", "BAS.DE", "BB.TO", "BBD-B.TO", "BC9.IR", "BEKB.BR", "BFIT.MI", "BIM.AS", "BION.SW", "BLDP.TO", "BN.PA", "BNR.DE", "BPOST.BR", "BRE.DE", "BVB.DE", "CARL-B.CO", "CBK.DE", "CCO.TO", "CEZ.PR", "CG.PA", "CHR.CO", "CIE.DE", "COLO-B.CO", "COLR.MC", "CPG.TO", "CRG.IR", "CS.PA", "DB1.DE", "DCC.IR", "DELIV.DE", "DEMANT.CO", "DG.PA", "DHL.DE", "DIM.PA", "DKL.DE", "DSV.CO", "EDEN.PA", "EDF.PA", "EDP.LS", "EDPR.LS", "EBS.VI", "EL.PA", "ELEKTRA.ST", "ELISA.HE", "ELUX-B.ST", "ENR.DE", "ENR.VI", "EO.PA", "EPIA.MI", "ERF.PA", "ERST.VI", "ESI.PA", "EUT.PA", "EVK.DE", "EVN.VI", "EVO.ST", "EXO.PA", "EXS1.DE", "FACC.VI", "FCG.DE", "FGR.PA", "FLTR.IR", "FNAC.PA", "FOR.ST", "FORTUM.HE", "FR.PA", "FRA.DE", "FREENET.DE", "GALP.LS", "GET.PA", "GEV.BE", "GFG.DE", "GIB.A.TO", "GILD", "GIS", "GLPG.AS", "GMAB.CO", "GMX.DE", "GPEG.MI", "GRF.MC", "GSZ.DE", "GTO.TO", "GTT.PA", "HAG.DE", "HAN.DE", "HAPP.DE", "HDL.DE", "HGR.MI", "HLE.DE", "HLG.DE", "HO.PA", "HOLN.SW", "HTC.DE", "HUH1V.HE", "HUSQ-B.ST", "IAP.DE", "IBE.MC", "IBPO.PA", "ICA.ST", "IFX.DE", "IGG.L", "IMB.L", "IMCD.AS", "IMMO.VI", "IMT.MI", "IPO.DE", "IPS.PA", "IPN.DE", "ISN.DE", "ISR.DE", "ITX.MC", "IVE.MI", "IVO.DE", "IXI.DE", "JDEP.AS", "JER.LS", "JMT.LS", "JUN3.DE", "KBC.BR", "KCO.DE", "KESKO-B.HE", "KGF.L", "KHC", "KION.DE", "KKR", "KNEBV.HE", "KPN.AS", "KSP.DE", "LATOUR-B.ST", "LB.PA", "LEG.DE",
        # Adding final 320 (sixth batch) - Reaching 1000, very diverse global mix
        "LEN.DE", "LHA.DE", "LIBA.DE", "LINDA.DE", "LXS.DE", "LYXA.DE", "MAOA.DE", "MBG.VI", "MDO.DE", "METRO.DE", "MLP.DE", "MOR.DE", "MTX.DE", "NEM.DE", "NOEJ.DE", "NRN.DE", "O2D.DE", "OSS.DE", "P911.DE", "PAH3.DE", "PBB.DE", "PFV.DE", "PNE3.DE", "PUM.DE", "QIA.DE", "QSC.DE", "RAA.DE", "RHK.DE", "RIB.DE", "RSTA.DE", "RWEG.DE", "SAX.DE", "SBS.DE", "SDF.DE", "SGL.DE", "SHA.DE", "SHOP.DE", "SIEGY", "SIX2.DE", "SOW.DE", "SRT3.DE", "STAG.DE", "STL.DE", "SUSE.DE", "SY1.DE", "SYK", "SZG.DE", "SZU.DE", "TKA.DE", "TLX.DE", "TRN.DE", "TTK.DE", "TUI1.DE", "UHR.VX", "UKI.DE", "UL.AS", "UNI01.DE", "UTDI.DE", "VIB3.DE", "VIE.VI", "VOS.DE", "VOW.DE", "WAC.DE", "WCH.DE", "WDI.DE", "WIE.DE", "WLN.AS", "WOS.AS", "YAR.OL", "ZAL.DE", "ZPR.DE", "ZUM.DE", "PKO.WA", "PEO.WA", "LPP.WA", "PZU.WA", "KGH.WA", "CDR.WA", "DNP.WA", "SAN.WA", "MBK.WA", "SPL.WA", "ORL.WA", "ACP.WA", "KTY.WA", "ALE.WA", "CCC.WA", "TPE.WA", "JSW.WA", "MAB.MC", "PHM.MC", "SGRE.MC", "BKT.MC", "SACYR.MC", "LOG.MC", "BKIA.MC", "ACX.MC", "VIS.MC", "IDR.MC", "ALM.MC", "CIEA.MC", "EKT.MC", "ROVI.MC", "GEST.MI", "SOL.MI", "IGD.MI", "DOV.MI", "CELL.MI", "AMP.MI", "CAI.MI", "ASC.MI", "MARR.MI", "BDT.MI", "SAI.MI", "SFL.MI", "TIP.MI", "AVIO.MI", "MFEA.MI", "MFEEB.MI", "DB.MI", "BBVA.WA", "ENG.MC", "IPG", "IRM", "ISRG", "IT", "ITW", "IVV", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEX", "KEY", "KEYS", "KIM", "KLAC", "KMB", "KMI", "KMX", "KOF", "KR", "LMT", "LNT", "LRCX", "LUMN", "LUV", "LYB", "LYV", "MAR", "MAS", "MCHP", "MDLZ", "MDT", "MMC", "MMM", "MO", "MOS", "MPC", "MPWR", "MRNA", "MS", "MSI", "MTCH", "MTD", "MUFG", "MXIM", "NDAQ", "NEE", "NEM", "NI", "NLOK", "NLSN", "NOC", "NOV", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVR", "NWL", "NWS", "NWSA", "O", "ODFL", "OKE", "OMC", "ORLY", "OTIS", "OXY", "PAYC", "PBCT", "PCAR", "PCG", "PEG", "PFE", "PFGC", "PG", "PH", "PKI", "PLD", "PM", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRGO", "PSA", "PSX", "PTC", "PVH", "PWR", "PXD", "QGEN", "REG", "REGN", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "SBAC", "SBNY", "SBUX", "SCCO", "SCHL", "SEE", "SHW", "SJM", "SLB", "SLG", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STZ", "SWK", "SWKS", "SYF", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRMB", "TRU", "TSCO", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UA", "UAA", "UAL", "UDR", "UHS", "ULTA", "UNP", "UPS", "URI", "USB", "VFC", "VICI", "VLO", "VMC", "VRSN", "VRTX", "VTRS", "VZ", "WAB", "WAT", "WBA", "WCN", "WEC", "WELL", "WHR", "WM", "WMB", "WST", "WU", "WY", "WYNN", "XRAY", "XRX", "YUMC", "ZBRA", "SOLJ.J", "IMPJ.J", "ANGJ.J", "SBKJ.J", "CPIJ.J", "NPN.JO", "BTI.JO", "GFI.JO", "SSW.JO", "EXX.JO", "APN.JO", "DSY.JO", "FSR.JO", "MNP.JO", "SHP.JO", "REM.JO", "MRP.JO", "VOD.JO", "MTN.JO", "NED.JO", "SLMJ.J", "ARIJ.J", "KIOJ.J", "OMUJ.J", "PPHJ.J", "PIKJ.J", "CLSJ.J", "QLTJ.J", "WHLJ.J", "TRUJ.J", "BAWJ.J", "SPPJ.J", "BIDJ.J", "SNHJ.J", "ABGJ.J", "AVI.JO", "SOLBE1.JO", "NTC.JO", "TFG.JO", "PPC.JO", "GRT.JO", "HCI.JO", "NPH.JO", "DRD.JO", "ITLJ.J", "LHC.JO", "AEL.JO", "LTE.JO", "RBX.JO", "DRC.JO"
]

crypto_list_usd = [
    "BTCUSD", "ETHUSD", "USDTUSD", "BNBUSD", "SOLUSD", "USDCUSD", "XRPUSD", "DOGEUSD", "TONUSD", "ADAUSD",
    "SHIBUSD", "AVAXUSD", "TRXUSD", "WBTCUSD", "DOTUSD", "LINKUSD", "MATICUSD", "BCHUSD", "ICPUSD", "LTCUSD",
    "NEARUSD", "LEOUSD", "DAIUSD", "UNIUSD", "ETCUSD", "STETHUSD", "XLMUSD", "OKBUSD", "INJUSD", "XMRUSD",
    "IMXUSD", "HBARUSD", "CROUSD", "ATOMUSD", "KASUSD", "FDUSDUSD", "FILUSD", "APTUSD", "MKRUSD", "RNDRUSD",
    "LDOUSD", "GRTUSD", "VETUSD", "OPUSD", "ARUSD", "TAOUSD", "TIAUSD", "SUIUSD", "MNTUSD", "BONKUSD",
    "BEAMUSD", "FETUSD", "AGIXUSD", "THETAUSD", "FLOKIUSD", "ENAUSD", "PYTHUSD", "BGBUSD", "ARBUSD", "AAVEUSD",
    "ALGOUSD", "JUPUSD", "WIFUSD", "EGLDUSD", "FLOWUSD", "FTMUSD", "SEIUSD", "XTZUSD", "AXSUSD", "EOSUSD",
    "BTTUSD", "SANDUSD", "CHZUSD", "MANAUSD", "QNTUSD", "SNXUSD", "NEOUSD", "GALAUSD", "AKTUSD", "CFXUSD",
    "MINAUSD", "KCSUSD", "IOTAUSD", "FXSUSD", "ZECUSD", "KLAYUSD", "GNOUSD", "DYDXUSD", "OCEANUSD", "TUSDUSD",
    "RUNEUSD", "FTTUSD", "ROSEUSD", "GMXUSD", "CVXUSD", "ANKRUSD", "PEPEUSD", "CAKEUSD", "USDDUSD", "ZILUSD",
    "KAVAUSD", "PAXGUSD", "TWTUSD", "CRVUSD", "MASKUSD", "WOOUSD", "AUDIOUSD", "ENSUSD", "CELOUSD", "ASTRUSD",
    "QTUMUSD", "SSVUSD", "ONEUSD", "RVNUSD", "WAVESUSD", "COMPUSD", "NEXOUSD", "HOTUSD", "DASHUSD", "BATUSD",
    "XECUSD", "ENJUSD", "XDCUSD", "SCUSD", "ZRXUSD", "STXUSD", "KSMUSD", "ARDRUSD", "UMAUSD", "POLYUSD",
    "STORJUSD", "YFIUSD", "ICXUSD", "SUSHIUSD", "IOTXUSD", "RLCUSD", "LSKUSD", "SKLUSD", "SFPUSD", "GLMRUSD",
    "HNTUSD", "OGNUSD", "BADGERUSD", "API3USD", "BANDUSD", "COTIUSD", "RENUSD", "AMPUSD", "ONTUSD", "POWRUSD",
    "PERPUSD", "CHRUSD", "LPTUSD", "ALPHAUSD", "ARPAUSD", "REQUSD", "BLZUSD", "TRUUSD", "CVCUSD", "ORNUSD",
    "FLUXUSD", "RLYUSD", "MDXUSD", "BNTUSD", "IDEXUSD", "NMRUSD", "AUCTIONUSD", "MLNUSD", "PLAUSD", "RADUSD",
    "QIUSD", "ALICEUSD", "DENTUSD", "ILVUSD", "ACHUSD", "SYSUSD", "SUPERUSD", "GHSTUSD", "PENDLEUSD", "HIGHUSD",
    "STEEMUSD", "CTSIUSD", "DODOUSD", "VGXUSD", "GTCUSD", "MBOXUSD", "OXTUSD", "PUNDIXUSD", "WAXPUSD", "UOSUSD",
    "FUNUSD", "CKBUSD", "BALUSD", "SLPUSD", "TLMUSD", "DGBUSD", "REEFUSD", "ELFUSD", "ANTUSD", "ZENUSD",
    "PHAUSD", "WRXUSD", "FETUSD", "SXPUSD", "CELRUSD", "LRCUSD", "DUSKUSD", "OCEANUSD", "COCOSUSD", "BELUSD",
    "NKNUSD", "OMGUSD", "BNXUSD", "WINUSD", "VTHOUSD", "TOMOUSD", "FISUSD", "KP3RUSD", "FRONTUSD", "OGUSD",
    "FORTHUSD", "UNFIUSD", "BURGERUSD", "BAKEUSD", "HARDUSD", "PONDUSD", "DERCUSD", "ALCXUSD", "TRBUSD", "FIDAUSD",
    "PROMUSD", "MTLUSD", "OAXUSD", "RAREUSD", "WINGUSD", "KEEPUSD", "NUUSD", "POLSUSD", "TVKUSD", "ERNUSD",
    "C98USD", "BETAUSD", "LITUSD", "DATAUSD", "STMXUSD", "ARKUSD", "LOOMUSD", "MBLUSD", "ARPAUSD", "RIFUSD",
    "TFUELUSD", "QKCUSD", "COSUSD", "KEYUSD", "WANUSD", "CTXCUSD", "IRISUSD", "PNTUSD", "MDTUSD", "AVAUSD",
    "VITEUSD", "BLZUSD", "AIONUSD", "DREPUSD", "PERLUSD", "PBTUSD", "AERGOUSD", "SOLVEUSD", "BTSUSD", "SRMUSD",
    "CREAMUSD", "AKROUSD", "VIDTUSD", "SWRVUSD", "COVERUSD", "HEGICUSD", "YFIIUSD", "WNXMUSD", "ALEPHUSD", "DFUSD",
    "GHSTUSD", "KP0RUSD", "BONDUSD", "PICKLEUSD", "ROOKUSD", "DPIUSD", "INDEXUSD", "FARMUSD", "NESTUSD", "FORUSD",
    "GOFUSD", "JRTUSD", "PNKUSD", "UBTUSD", "RSRUSD", "STAUSD", "STAKEUSD", "RINGUSD", "XRTUSD", "MAPSUSD",
    "OXYUSD", "BRZUSD", "DEGOUSD", "MATHUSD", "TRADEUSD", "HXROUSD", "RAMPUSD", "YLDUSD", "RAZORUSD", "OMUSD",
    "DEXEUSD", "XEDUSD", "XORUSD", "MTAUSD", "STCUSD", "DSLAUSD", "HAIUSD", "LAYERUSD", "DEPUSD", "PLOTUSD",
    "ROOMUSD", "DAOUSD", "HAPIUSD", "GLCHUSD", "ORAIUSD", "RFUELUSD", "TLOSUSD", "VELOUSD", "ZCNUSD", "ALPAUSD",
    "KLVUSD", "PMONUSD", "FEVRUSD", "DGUSD", "UNOUSD", "ZAPUSD", "MARSHUSD", "XAVAUSD", "DFYNUSD", "POLKUSD",
    "POOLZUSD", "RAZEUSD", "EQZUSD", "NAOSUSD", "FODLUSD", "MONIUSD", "QRDOUSD", "SOVUSD", "SUKUUSD", "CELLUSD",
    "XCADUSD", "BLPUSD", "CQTUSD", "EPIKUSD", "GAMEUSD", "GMRUSD", "HEROUSD", "LUFFYUSD", "MCUSD", "MISTUSD",
    "NTVRKUSD", "PYRUSD", "RACAUSD", "REVOUSD", "SFUNDUSD", "SLNV2USD", "STARLUSD", "UFOUSD", "VLXUSD", "WARUSD",
    "YOOSHIUSD", "ZCXUSD", "ZOOUSD", "ADSUSD", "AIOZUSD", "ARVUSD", "BABYUSD", "BDPUSD", "BIFIUSD", "BLANKUSD",
    "BLESUSD", "BOSONUSD", "BRKLUSD", "BSCUSD", "BSCPADUSD", "BSCSUSD", "CAKEPADUSD", "CARDUSD", "CARRUSD", "CGGUSD",
    "CHAINUSD", "CHESSUSD", "CLHUSD", "CNDUSD", "COOKUSD", "COREUSD", "CPOOLUSD", "CUDOSUSD", "DAPPUSD", "DDIMUSD",
    "DEHRUSD", "DFLUSD", "DHEGUSD", "DHTUSD", "DISUSD", "DMSUSD", "DOVUSD", "DPETUSD", "DREUSD", "DRSUSD",
    "DUCKUSD", "DUSTUSD", "DXLUSD", "ECELLUSD", "EFXUSD", "EHRTUSD", "ELMONUSD", "EMONUSD", "EPKUSD", "EPWUSD",
    "ETHAUSD", "FARAUSD", "FCONUSD", "FEARUSD", "FINAUSD", "FIREUSD", "FLAMEUSD", "FLOWUSD", "FLYUSD", "FORMUSD",
    "FREEUSD", "FRMUSD", "FRONUSD", "FUSEUSD", "FXFUSD", "FYZUSD", "GAFIUSD", "GENEUSD", "GENSUSD", "GETUSD",
    "GFUSD", "GGGUSD", "GGTKUSD", "GHDUSD", "GIVUSD", "GLQUSD", "GMEEUSD", "GOUSD", "GOLDUSD", "GONEUSD",
    "GOVIUSD", "GPOOLUSD", "GRIDUSD", "GSUSD", "GUMUSD", "HAKAUSD", "HCTUSD", "HDAOUSD", "HGETUSD", "HODUSD",
    "HORDUSD", "HQTUSD", "HTRUSD", "HYVEUSD", "IBSUSD", "ICEUSD", "IDEAUSD", "IDIAUSD", "IDVUSD", "IGGUSD",
    "IMTUSD", "INUSD", "INSURUSD", "INXTUSD", "IQNUSD", "ISLAUSD", "ISTUSD", "IXSUSD", "JGNUSD", "KALMUSD",
    "KAIUSD", "KEXUSD", "KEYFIUSD", "KPADUSD", "KRLUSD", "KSMUSD", "KSTUSD", "KTCUSD", "KUDUSD", "LABSUSD",
    "LACEUSD", "LAUNCHUSD", "LBPUSD", "LEASHUSD", "LEMDUSD", "LFGUSD", "LITHUSD", "LKRUSD", "LOKAUSD", "LOOKSUSD",
    "LOTUSD", "LPOOLUSD", "LSSUSD", "LTRBTUSD", "LUARTUSD", "LUAUSD", "LUDUSD", "LYXEUSD", "MAGEUSD", "MANUSD",
    "MANGOUSD", "MANYUSD", "MARSUSD", "MASQUSD", "MBXUSD", "MCHCUSD", "MCONTENTUSD", "MCRNUSD", "MCVUSD", "MDMSUSD",
    "MDXUSD", "MEANUSD", "MEDIAUSD", "MEPADUSD", "METUSD", "METISUSD", "MEVUSD", "MFIUSD", "MGGUSD", "MIDASUSD",
    "MIMUSD", "MINTUSD", "MISTUSD", "MIXUSD", "MMUSD", "MMFUSD", "MMSUSD", "MNGOUSD", "MNSTUSD", "MNSTRUSD",
    "MOBIUSD", "MOCHIUSD", "MODUSD", "MOFIUSD", "MOJOUSD", "MOMOUSD", "MOONUSD", "MORKUSD", "MORPHUSD", "MOTUSD",
    "MOVEZUSD", "MOWAUSD", "MPADUSD", "MPLUSD", "MQLUSD", "MRCHUSD", "MRFIUSD", "MRPHUSD", "MSBUSD", "MSCUSD",
    "MSSUSD", "MSUUSD", "MTIXUSD", "MTVTUSD", "MUDRAUSD", "MULTIUSD", "MVPUSD", "MVSUSD", "MXCUSD", "MYIDUSD",
    "MYMUSD", "MYOUSD", "MYRAUSD", "MYTHUSD", "N1USD", "NABOBUSD", "NAFTUSD", "NAKAUSD", "NALSUSD", "NANOUSD",
    "NASUSD", "NAVIUSD", "NAXUSD", "NBOTUSD", "NBPUSD", "NCDTUSD", "NCTUSD", "NDCUSD", "NDRUSD", "NDXUSD",
    "NEBLUSD", "NEBOUSD", "NEFTUSD", "NEOFIUSD", "NEONUSD", "NERDUSD", "NESTUSD", "NETUSD", "NETMUSD", "NETVRUSD",
    "NEXUSD", "NEXMUSD", "NFTUSD", "NFTBUSD", "NFTDUSD", "NFTLUSD", "NFTOUSD", "NFTPADUSD", "NFTSUSD", "NFTYUSD",
    "NFTZUSD", "NGLUSD", "NGMUSD", "NIFUSD", "NIIFIUSD", "NIMUSD", "NIOXUSD", "NKAUSD", "NKCUSD", "NLCUSD",
    "NLGUSD", "NMCUSD", "NMRKUSD", "NMSUSD", "NNIUSD", "NOAUSD", "NODEUSD", "NODLUSD", "NOIAUSD", "NOMUSD",
    "NOONUSD", "NORUSD", "NOSUSD", "NOTAUSD", "NOTEUSD", "NPASUSD", "NPXSUSD", "NROUSD", "NRSUSD", "NRVUSD",
    "NRVEUSD", "NSBTUSD", "NSDXUSD", "NSHAREUSD", "NSUREUSD", "NTUSD", "NTRUSD", "NTRNUSD", "NTSUSD", "NTXUSD",
    "NTVUSD", "NUARSUSD", "NULSUSD", "NUMUSD", "NUSDUSD", "NUTUSD", "NUVOUSD", "NWCUSD", "NXTUSD", "NXUSDUSD",
    "NYANUSD", "NYMUSD", "NYCUSD", "NYEUSD", "NYNUSD", "NYXUSD", "NYZOUSD", "O3USD", "OASUSD", "OAXUSD",
    "OBSRUSD", "OBTUSD", "OCCUSD", "OCEUSD", "OCTUSD", "OCTOUSD", "ODEUSD", "ODINUSD", "ODDZUSD", "ODSUSD",
    "OFIUSD", "OGXUSD", "OHUSD", "OHMUSD", "OINUSD", "OKUSD", "OKLGUSD", "OKSUSD", "OLASUSD", "OLEUSD",
    "OLTUSD", "OLYUSD", "OMAXUSD", "OMBUSD", "OMGUSD", "OMIUSD", "OMNIUSD", "OMTUSD", "ONCUSD", "ONDOUSD",
    "ONEUSD", "ONEMOONUSD", "ONEROOTUSD", "ONETUSD", "ONIONUSD", "ONITUSD", "ONLXUSD", "ONSTONUSD", "ONTUSD", "ONUSUSD",
    "ONXUSD", "OOEUSD", "OPCTUSD", "OPENUSD", "OPERAUSD", "OPIUMUSD", "OPLUSD", "OPNNUSD", "OPSUSD", "OPTUSD",
    "OPUUSD", "OPULUSD", "OPXUSD", "ORAUSD", "ORBUSD", "ORBITUSD", "ORBSUSD", "ORCAUSD", "OREUSD", "ORIUSD",
    "ORIGOUSD", "ORIONUSD", "ORKUSD", "ORMEUSD", "ORNUSD", "OROUSD", "ORSUSD", "ORSTUSD", "ORTUSD", "ORZUSD",
    "OSUSD", "OSCUSD", "OSKUSD", "OSLUSD", "OSMUSD", "OSMOUSD", "OSQUSD", "OSTUSD", "OSWAPUSD", "OTBUSD",
    "OTCUSD", "OTKUSD", "OTOUSD", "OTTUSD", "OUROUSD", "OUSUSD", "OVUSD", "OVAUSD", "OVERUSD", "OVIXUSD",
    "OVRUSD", "OVRLUSD", "OWCUSD", "OWLUSD", "OWNUSD", "OXUSD", "OXBUSD", "OXDUSD", "OXTUSD", "OXYUSD",
    "OXZUSD", "OYUSD", "OZCUSD", "P2PUSD", "P3DUSD", "PAAUSD", "PACAUSD", "PACEUSD", "PACKUSD", "PADUSD",
    "PAIDUSD", "PAINTUSD", "PALUSD", "PALLAUSD", "PAMPUSD", "PANUSD", "PANDAUSD", "PANDOUSD", "PANGUSD", "PANIAUSD",
    "PANKUSD", "PANTUSD", "PAOUSD", "PAPUSD", "PAPERUSD", "PAPPUSD", "PARUSD", "PARAUSD", "PARASUSD", "PARETOUSD",
    "PARKUSD", "PARSIQUSD", "PARTUSD", "PASCUSD", "PASSUSD", "PASTAUSD", "PATUSD", "PATEUSD", "PATHUSD", "PAWNUSD",
    "PAXUSD", "PAYUSD", "PAYBUSD", "PAYCONUSD", "PAYNETUSD", "PAYPUSD", "PAYSUSD", "PAZUSD", "PBASEUSD", "PBRUSD",
    "PBTCUSD", "PBXUSD", "PCHAINUSD", "PCMUSD", "PCNUSD", "PCXUSD", "PDAIUSD", "PDIUSD", "PDMUSD", "PDSHAREUSD",
    "PDSUSD", "PDXUSD", "PEAUSD", "PEACEUSD", "PEAKUSD", "PEARLUSD", "PEASUSD", "PEAXUSD", "PEDUSD", "PEEUSD",
    "PEEPSUSD", "PEGUSD", "PEGAUSD", "PENGUSD", "PENKUSD", "PENTOUSD", "PEONUSD", "PEPUSD", "PEPECUSD", "PEPEDUSD",
    "PEPEETFUSD", "PEPEFLOKIUSD", "PEPEGUSD", "PEPELUSD", "PEPESUSD", "PEPETUSD", "PEPEWUSD", "PERAUSD", "PERCUSD", "PERIUSD",
    "PERLUSD", "PERXUSD", "PESUSD", "PETAUSD", "PETSUSD", "PEXUSD", "PEXAUSD", "PFLUSD", "PFTUSD", "PFWUSD",
    "PGCUSD", "PGDUSD", "PGFUSD", "PGKUSD", "PGNUSD", "PGSUSD", "PGTUSD", "PGXUSD", "PHAUSD", "PHAEUSD",
    "PHALLAUSD", "PHANESUSD", "PHANTOMUSD", "PHATUSD", "PHBUSD", "PHEUSD", "PHNXUSD", "PHOUSD", "PHONUSD", "PHONONUSD",
    "PHOTONUSD", "PHPUSD", "PHRUSD", "PHXUSD", "PHYUSD", "PHYLUSD", "PIUSD", "PIAUSD", "PIASUSD", "PIBUSD",
    "PICAUSD", "PICNUSD", "PIDAOUSD", "PIEUSD", "PIFUSD", "PIGUSD", "PIGEUSD", "PIGGYUSD", "PIGINUUSD", "PIKAUSD",
    "PIKABONKUSD", "PIKACHUUSD", "PIKENUUSD", "PIKOUSD", "PILUSD", "PILEUSD", "PILLUSD", "PILOTUSD", "PINUSD", "PINAKAUSD",
    "PINCUSD", "PINEUSD", "PINKUSD", "PINKEUSD", "PINKMUSD", "PINKPUSD", "PINKSUSD", "PINKYUSD", "PIOUSD", "PIONUSD",
    "PIPADUSD", "PIPLUSD", "PIRATEUSD", "PISUSD", "PITUSD", "PITBUSD", "PITCHUSD", "PITHUSD", "PIVXUSD", "PIXUSD",
    "PIXELUSD", "PIXIAUSD", "PIXLSUSD", "PIZAUSD", "PKOINUSD", "PKRUSD", "PKTUSD", "PLAUSD", "PLACEUSD", "PLANUSD",
    "PLANETUSD", "PLASTIKUSD", "PLATUSD", "PLAYUSD", "PLAYAUSD", "PLAYBUSD", "PLAYCUSD", "PLAYDUSD", "PLAYERUSD", "PLAYKEYUSD",
    "PLAYNUSD", "PLBUSD", "PLBTUSD", "PLCUSD", "PLCUUSD", "PLCYUSD", "PLDUSD", "PLEUSD", "PLEBUSD", "PLEDGEUSD",
    "PLENAUSD", "PLENTUSD", "PLESUSD", "PLEXUSD", "PLGUSD", "PLGRUSD", "PLMUSD", "PLNUSD", "PLRUSD", "PLSUSD",
    "PLSAUSD", "PLSDUSD", "PLSRUSD", "PLTUSD", "PLTCUSD", "PLUUSD", "PLUSUSD", "PLUTUSD", "PLUTOUSD", "PLXUSD",
    "PLYUSD", "PLZUSD", "PMAUSD", "PMANUSD", "PMBUSD", "PMCUSD", "PMDUSD", "PMGTUSD", "PMLUSD", "PMMUSD",
    "PMNUSD", "PMONUSD", "PMPUSD", "PMPYUSD", "PMTUSD", "PMXUSD", "PNAUSD", "PNDUSD", "PNDCUSD", "PNFTUSD",
    "PNGUSD", "PNIXSUSD", "PNKUSD", "PNLUSD", "PNODEUSD", "PNSUSD", "PNTUSD", "PNXUSD", "PNYUSD", "POAUSD"
]

etf_list = [
    "SPY", "IVV", "VOO", "VTI", "QQQ", "VEA", "VUG", "IEFA", "AGG", "VTV",
    "GLD", "VWO", "BND", "IJR", "IJH", "IWM", "EFA", "VYM", "XLF", "SCHD",
    "DIA", "VB", "EEM", "XLK", "VXUS", "ITOT", "SCHX", "IWF", "TIP", "LQD",
    "SCHB", "IWD", "SDY", "USMV", "GLDM", "XLE", "QUAL", "XLV", "MDY", "VO",
    "VIG", "SCHF", "IAU", "XLY", "RSP", "SHV", "SHY", "SCHG", "SPLG", "BSV",
    "XLC", "IEF", "SCHA", "DGRO", "HYG", "IWR", "XLI", "VNQ", "SLV", "IVW",
    "TLT", "VOE", "EMLC", "VTIP", "BNDX", "MBB", "NOBL", "VBR", "ESGU", "SCHP",
    "XLB", "GDX", "IUSG", "VXF", "XLP", "IEMG", "SCHM", "IUSV", "IEI", "GDXJ",
    "SPHD", "DVY", "VT", "IWB", "ACWI", "VCSH", "SCHE", "IYR", "XLRE", "OEF",
    "VBK", "XLU", "ARKK", "JNK", "IGSB", "ESGV", "FNDA", "DGRW", "VGT", "SCHH",
    "DON", "FDN", "IGIB", "MUB", "SPAB", "XBI", "MOAT", "SPDW", "XLFS", "FXI",
    "SOXX", "IWO", "USHY", "FLOT", "IOO", "IWP", "IXUS", "KRE", "PFF", "SPLV",
    "VHT", "BIV", "VMBS", "BOTZ", "COWZ", "EWJ", "FVD", "IAUM", "IEUR", "IHI",
    "IJT", "ITA", "IYC", "IYE", "IYF", "IYH", "IYK", "IYLD", "IYM", "IYW",
    "IYZ", "JEPI", "KBE", "KIE", "MGK", "MTUM", "OIH", "PBW", "PEJ", "PGX",
    "PHO", "PSJ", "PTH", "REM", "REZ", "RFG", "RHS", "RTH", "RTM", "RWO",
    "RWX", "RZG", "RZV", "SCHC", "SCHV", "SCHZ", "SDOG", "SGOV", "SMH", "SOXL",
    "SPEM", "SPGM", "SPHQ", "SPMD", "SPMV", "SPTM", "SPTS", "SPUS", "SPVM", "SPVU",
    "SPXB", "SPXE", "SPXG", "SPXL", "SPXN", "SPXS", "SPXT", "SPXU", "SPXV", "SPYD",
    "SPYG", "SPYV", "SPYX", "SQQQ", "SRLN", "SSO", "SUB", "TAN", "TECL", "TECS",
    "TFI", "TQQQ", "UCO", "UDOW", "UGAZ", "UNG", "UPRO", "USFR", "USO", "UUP",
    "UVXY", "VCR", "VDC", "VDE", "VEGI", "VEU", "VFH", "VGLT", "VGSH", "VIS",
    "VLUE", "VNM", "VNQI", "VONG", "VONV", "VOOG", "VOOV", "VOX", "VPL", "VPUEF",
    "VPU", "VSS", "VTWO", "VTWV", "VTHR", "VTEB", "VV", "VYMI", "XAR", "XES",
    "XHB", "XHE", "XHS", "XITK", "XLG", "XLNX", "XME", "XNTK", "XOP", "XPH",
    "XRT", "XSD", "XSLV", "XSW", "XTL", "XTN", "XWEB", "YOLO", "ZROZ", "AAAU",
    "ACES", "ACSI", "ACWV", "ADRE", "AEWU", "AFK", "AFTY", "AGGPU", "AGGY", "AGQ",
    "AGZ", "AIA", "AIEQ", "AIIQ", "AIRR", "AJAN", "ALFA", "ALFY", "ALTS", "ALTY",
    "AMLP", "AMOM", "AMZA", "ANEW", "ANGL", "APPL", "AQWA", "ARGT", "ARKF", "ARKG",
    "ARKQ", "ARKW", "ARKX", "ARMR", "ASEA", "ASHR", "ASHX", "AUSF", "AVDE", "AVDV",
    "AVEM", "AVGE", "AVGOV", "AVHY", "AVIG", "AVLC", "AVLV", "AVMC", "AVMU", "AVNM",
    "AVRE", "AVSC", "AVSF", "AVUS", "AVUV", "AWAY", "AWTM", "AXJL", "AYF", "AZAA",
    "AZL", "BAB", "BALT", "BATT", "BBAX", "BBCA", "BBEU", "BBJP", "BBLB", "BBMC",
    "BBSA", "BBTG", "BBUS", "BCA", "BCD", "BCI", "BCIL", "BCIM", "BCNA", "BCOW",
    "BCS", "BCUS", "BDCX", "BDRY", "BEDZ", "BETZ", "BFIT", "BFOR", "BICK", "BIGZ",
    "BIL", "BIZD", "BJK", "BKCH", "BKEM", "BKF", "BKIE", "BKLN", "BLCN", "BLDG",
    "BLHY", "BLOK", "BLV", "BNDC", "BNDD", "BNE", "BNKU", "BNY", "BOIL", "BOKK",
    "BOND", "BOSS", "BOUT", "BOXX", "BRF", "BRZU", "BSAE", "BSBE", "BSCE", "BSCHE",
    "BSCI", "BSCJ", "BSCK", "BSCL", "BSCM", "BSCN", "BSCO", "BSCP", "BSCQ", "BSCR",
    "BSCS", "BSCT", "BSCU", "BSCV", "BSCW", "BSCX", "BSCZ", "BSET", "BSJA", "BSJB",
    "BSJC", "BSJD", "BSJE", "BSJF", "BSJG", "BSJH", "BSJI", "BSJJ", "BSJK", "BSJL",
    "BSJM", "BSJN", "BSJO", "BSJP", "BSJQ", "BSJR", "BSJT", "BSJU", "BSMO", "BSMR",
    "BSMW", "BSMX", "BSMY", "BSMZ", "BSTP", "BTAL", "BTEC", "BTEK", "BTF", "BUG",
    "BUFR", "BUL", "BULL", "BUZZ", "BVAL", "BVLU", "BVS", "BWX", "BWZ", "BYL",
    "BYLD", "BZFD", "BZQ", "CALF", "CANE", "CAPE", "CARD", "CARZ", "CATH", "CAV",
    "CBND", "CBON", "CCOR", "CCRV", "CCSO", "CCXJ", "CDAE", "CDEE", "CDL", "CDNA",
    "CEFA", "CEFD", "CEFS", "CEMB", "CEW", "CFA", "CFRA", "CGOV", "CHAU", "CHB",
    "CHEP", "CHGX", "CHIC", "CHIE", "CHII", "CHIH", "CHIK", "CHIL", "CHIM", "CHIN",
    "CHIQ", "CHIR", "CHIS", "CHIT", "CHIU", "CHIX", "CHNA", "CHNB", "CHNG", "CHNL",
    "CHOC", "CHRG", "CHRT", "CHSC", "CHSE", "CHSEP", "CHSG", "CHSP", "CHST", "CHSU",
    "CHSZ", "CHVT", "CHVX", "CHWE", "CHXF", "CHXJ", "CIBR", "CIGO", "CIK", "CIL",
    "CINF", "CIP", "CIRZ", "CIX", "CIZ", "CLIX", "CLOU", "CLRG", "CLSE", "CLSK",
    "CLST", "CLTL", "CMEZ", "CMF", "CMFY", "CMIG", "CNBS", "CNDAL", "CNRG", "CNYA",
    "COBO", "COLA", "COMT", "COND", "CONL", "CONSL", "COPX", "CORN", "CORP", "COST",
    "CPER", "CPI", "CQQQ", "CRAK", "CRBN", "CRDT", "CROP", "CRUD", "CRUZ", "CSB",
    "CSD", "CSF", "CSHI", "CSL", "CSLS", "CSM", "CSML", "CSOP", "CSPR", "CSTD",
    "CTEC", "CTEST", "CTRU", "CURE", "CUT", "CVAR", "CVY", "CWAI", "CWB", "CWI",
    "CWID", "CWX", "CXSE", "CYA", "CYB", "CYBR", "DALT", "DAPP", "DARP", "DATA",
    "DATE", "DAUG", "DAX", "DBA", "DBAU", "DBAW", "DBB", "DBBE", "DBC", "DBDR",
    "DBE", "DBEF", "DBEH", "DBEM", "DBEU", "DBEZ", "DBGR", "DBJA", "DBJP", "DBKO",
    "DBMF", "DBMX", "DBO", "DBP", "DBRE", "DBS", "DBTX", "DBV", "DBZ", "DCF",
    "DDBI", "DDG", "DDIV", "DDLS", "DDM", "DDMX", "DDS", "DDWM", "DEE", "DEED",
    "DEEF", "DEEP", "DEF", "DEFA", "DEFN", "DEHP", "DEM", "DEMZ", "DES", "DEU",
    "DEUS", "DEUT", "DEW", "DEWJ", "DFAX", "DFAI", "DFAU", "DFCF", "DFE", "DFEM",
    "DFEN", "DFEV", "DFF", "DFHY", "DFIC", "DFIH", "DFIP", "DFIS", "DFIV", "DFJ",
    "DFND", "DFNL", "DFNV", "DFRA", "DFREX", "DFSA", "DFSE", "DFSI", "DFSM", "DFSU",
    "DFUS", "DFUV", "DFVX", "DGAZ", "DGBP", "DGL", "DGLD", "DGRE", "DGS", "DGT",
    "DGTZ", "DGZ", "DHS", "DHTA", "DIAX", "DIBR", "DICE", "DIDV", "DIG", "DIHP",
    "DIM", "DIRT", "DISR", "DIST", "DIVA", "DIVB", "DIVC", "DIVD", "DIVE", "DIVG",
    "DIVI", "DIVL", "DIVS", "DIVT", "DIVZ", "DJCB", "DJCI", "DJCO", "DJD", "DJIA",
    "DJIFF", "DJIM", "DJIX", "DJMC", "DJMIX", "DJN", "DJONU", "DJRE", "DJSK", "DJTR",
    "DJUS", "DKA", "DKL", "DLA", "DLN", "DLS", "DMAG", "DMAY", "DMC", "DMBS",
    "DMDV", "DMER", "DMF", "DMRE", "DMRS", "DMS", "DMUD", "DMYD", "DNL", "DNNG",
    "DNOV", "DNY", "DOC", "DOD", "DOGS", "DOL", "DOO", "DOTCOM", "DOVL", "DPAY",
    "DPST", "DPU", "DPX", "DQFD", "DRGO", "DRIP", "DRIV", "DRN", "DRSK", "DRUGS",
    "DRV", "DRW", "DSDA", "DSI", "DSKE", "DSM", "DSOC", "DSTL", "DSU", "DSUM",
    "DTD", "DTEM", "DTH", "DTL", "DTN", "DTYL", "DUDE", "DUG", "DULL", "DURA",
    "DUSL", "DUST", "DUSA", "DVAN", "DVLU", "DVOL", "DVYA", "DVYE", "DWAQ", "DWAS",
    "DWAT", "DWCR", "DWLD", "DWM", "DWMC", "DWSH", "DWTR", "DWX", "DXGE", "DXJ",
    "DXJF", "DXJH", "DXJS", "DXJT", "DXJU", "DXPS", "DXYZ", "DYEQ", "DYLD", "DYLS",
    "DYNF", "DYPE", "DYSL", "DZK", "DZZ", "EAGG", "EASI", "EBND", "ECH", "ECLN",
    "ECNS", "ECON", "ECOW", "ECOZ", "EDC", "EDEN", "EDIV", "EDOC", "EDOG", "EDOW",
    "EDRV", "EDU", "EEFT", "EEH", "EELV", "EEMD", "EEMO", "EEMS", "EEMV", "EEMX",
    "EES", "EET", "EFAV", "EFAD", "EFAX", "EFC", "EFFE", "EFG", "EFIV", "EFNL",
    "EFO", "EFU", "EFV", "EFWD", "EFZ", "EGAN", "EGFIX", "EGIF", "EGPT", "EGRW",
    "EGS", "EGY", "EHAN", "EHC", "EHI", "EIDO", "EIFL", "EIGR", "EIML", "EINC",
    "EIPR", "EIRL", "EIS", "EITF", "EIVI", "EJAN", "EJH", "EJP", "EKSA", "ELDF",
    "ELDOU", "ELEF", "EMAG", "EMBH", "EMCB", "EMCD", "EMCG", "EMCR", "EMDB", "EMDV",
    "EMDY", "EMEG", "EMEQ", "EMES", "EMFM", "EMGF", "EMGOV", "EMHG", "EMHY", "EMIF",
    "EMIH", "EMIN", "EMITF", "EMJH", "EMKR", "EMLPF", "EMLPL", "EMMF", "EMNT", "EMOM",
    "EMQQ", "EMRE", "EMSA", "EMSD", "EMSH", "EMTL", "EMTY", "EMUD", "EMVL", "EMXC",
    "EMXF", "EMXG", "EMXI", "EMXN", "EMXP", "EMXS", "EMXX", "ENGH", "ENFR", "ENRG",
    "ENSD", "ENTR", "ENXF", "ENZL", "EOGO", "EPI", "EPM", "EPP", "EPS", "EPHE",
    "EPOL", "EPRF", "EPU", "EPV", "EQAL", "EQRR", "EQUL", "EQWL", "EQWS", "ERDM",
    "ERMX", "ERSX", "ERTH", "ERUS", "ERX", "ERY", "ESEB", "ESG", "ESGA", "ESGD",
    "ESGE", "ESGG", "ESGH", "ESGI", "ESGJ", "ESGK", "ESGL", "ESGM", "ESGN", "ESGO",
    "ESGP", "ESGQ", "ESGR", "ESGS", "ESGT", "ESGUU", "ESGVU", "ESGW", "ESGY", "ESHY",
    "ESML", "ESP", "ESPO", "ETFL", "ETFS", "ETG", "ETHO", "ETHOX", "ETJ", "ETN",
    "ETO", "ETR", "ETSY", "ETV", "ETY", "EUA", "EUDG", "EUFIN", "EUFL", "EUFX",
    "EUM", "EUMV", "EUO", "EURL", "EURN", "EURR", "EUSC", "EVAL", "EVGBC", "EVI",
    "EVLMC", "EVSTC", "EVX", "EWA", "EWZS"
]

equity_detail_service = EquityDetailService()
list_service = ListService()
discovery_service = DiscoveryService()
indices_service = MarketIndicesService()

def create_ws_message(event: str, topic: str, data: Any) -> dict:
    return {"event": event, "topic": topic, "payload": data}

# --- Fetcher Definitions ---

async def _bitcoin_transactions_fetcher(topic: str):
    btc_service = BitcoinWsService()
    try:
        await btc_service.connect()
        if not btc_service.websocket: 
            logger.warning("BitcoinWsService failed to establish connection.")
            return

        await btc_service.subscribe_to_new_transactions()

        async for raw_tx in btc_service.listen_for_transactions():
            async with global_connection_manager._lock:
                if topic not in global_connection_manager.topic_subscriptions:
                    logger.info(f"[Fetcher:{topic}] No subscribers. Stopping BTC listener.")
                    break
            
            try:
                # --- Xử lý và làm giàu dữ liệu Bitcoin ---
                
                # Tính tổng giá trị input và output
                total_input_satoshi = sum(inp.get('prev_out', {}).get('value', 0) for inp in raw_tx.get('inputs', []))
                total_output_satoshi = sum(out.get('value', 0) for out in raw_tx.get('out', []))

                # Chuyển đổi đơn vị
                fee_in_btc = raw_tx.get('fee', 0) / SATOSHIS_IN_BTC
                input_value_btc = total_input_satoshi / SATOSHIS_IN_BTC
                output_value_btc = total_output_satoshi / SATOSHIS_IN_BTC

                # Tạo các đối tượng input/output đã được xử lý
                parsed_inputs = []
                for inp in raw_tx.get('inputs', []):
                    prev_out = inp.get('prev_out', {})
                    if prev_out:
                        parsed_inputs.append(BtcInputOutput(
                            addr=prev_out.get('addr'),
                            value=prev_out.get('value', 0),
                            value_in_btc=prev_out.get('value', 0) / SATOSHIS_IN_BTC
                        ))

                parsed_outputs = []
                for out in raw_tx.get('out', []):
                     parsed_outputs.append(BtcInputOutput(
                         addr=out.get('addr'),
                         value=out.get('value', 0),
                         value_in_btc=out.get('value', 0) / SATOSHIS_IN_BTC
                     ))
                
                # Tạo đối tượng giao dịch cuối cùng
                parsed_tx_payload = ParsedBtcTransaction(
                    hash=raw_tx.get('hash'),
                    time=raw_tx.get('time'),
                    size=raw_tx.get('size'),
                    fee=raw_tx.get('fee', 0),
                    fee_in_btc=round(fee_in_btc, 8),
                    input_value_btc=round(input_value_btc, 8),
                    output_value_btc=round(output_value_btc, 8),
                    inputs=parsed_inputs,
                    out=parsed_outputs
                )

                response_data_payload = APIResponseData[ParsedBtcTransaction](data=[parsed_tx_payload])
                api_response = APIResponse[ParsedBtcTransaction](
                    message="New Bitcoin Transaction",
                    provider_used="blockchain_com_ws",
                    data=response_data_payload
                )

                message = {
                    "event": "onchain_btc_transaction_update",
                    "topic": topic,
                    "payload": api_response.model_dump(by_alias=True)
                }
                await global_connection_manager.broadcast_to_topic(topic, message)

            except Exception as parsing_error:
                logger.error(f"[Fetcher:{topic}] Error parsing BTC transaction data: {parsing_error}", exc_info=True)

    except asyncio.CancelledError:
        logger.info(f"[Fetcher:{topic}] BTC listener task was cancelled.")
    except Exception as e:
        logger.error(f"[Fetcher:{topic}] Unhandled exception in BTC listener: {e}", exc_info=True)

async def _alchemy_transactions_fetcher(topic: str, params: OnchainSubscriptionParams):
    alchemy_service = AlchemyWsService(api_key=settings.ALCHEMY_API_KEY)
    decoder_service = TransactionDecoderService() # Khởi tạo service giải mã
    
    try:
        await alchemy_service.connect()
        if not alchemy_service.websocket:
            logger.error(f"[Fetcher:{topic}] Failed to connect to Alchemy WebSocket.")
            return

        addresses_to_watch = []
        for addr in params.addresses:
            addresses_to_watch.append({"to": addr})
            addresses_to_watch.append({"from": addr})

        await alchemy_service.subscribe_to_mined_transactions(
            addresses=addresses_to_watch,
            hashes_only=False
        )
        
        if not alchemy_service.subscription_id:
             logger.warning(f"[Fetcher:{topic}] Failed to get subscription ID. Stopping task.")
             return

        async for tx_data in alchemy_service.listen_for_transactions():
            async with global_connection_manager._lock:
                if topic not in global_connection_manager.topic_subscriptions:
                    logger.info(f"[Fetcher:{topic}] No more subscribers. Stopping.")
                    break
            
            raw_tx = tx_data.get("transaction")
            if not raw_tx:
                continue

            try:
                # --- Bước 1: Xử lý dữ liệu gốc (không đổi) ---
                wei_value = int(raw_tx.get("value", "0x0"), 16)
                ether_value = wei_value / 10**18
                gas_price_wei = int(raw_tx.get("gasPrice", "0x0"), 16)
                gas_price_gwei = gas_price_wei / 10**9
                tx_type_hex = raw_tx.get("type", "0x0")
                tx_type_str = "Legacy"
                if tx_type_hex == "0x1": tx_type_str = "EIP-2930"
                elif tx_type_hex == "0x2": tx_type_str = "EIP-1559"
                
                # --- Bước 2: Giải mã dữ liệu input (logic mới) ---
                decoded_input_info = None
                to_addr = raw_tx.get("to")
                input_data = raw_tx.get("input")
                # Chỉ giải mã nếu có địa chỉ hợp đồng và input data không phải là rỗng ('0x')
                if to_addr and input_data and input_data != "0x":
                    decoded_result = await decoder_service.decode_transaction_input(to_addr, input_data)
                    if decoded_result:
                        func_obj, func_params = decoded_result
                        decoded_input_info = {
                            "functionName": func_obj.fn_name,
                            "params": {k: str(v) for k, v in func_params.items()}
                        }
                        logger.info(f"Successfully decoded input for tx {raw_tx.get('hash')}: {func_obj.fn_name}")

                # --- Bước 3: Tạo payload cuối cùng ---
                processed_data = {
                    "hash": raw_tx.get("hash"),
                    "blockHash": raw_tx.get("blockHash"),
                    "blockNumber": int(raw_tx.get("blockNumber", "0x0"), 16),
                    "from": raw_tx.get("from"),
                    "to": to_addr,
                    "gas": int(raw_tx.get("gas", "0x0"), 16),
                    "gas_price_in_gwei": round(gas_price_gwei, 2),
                    "input": input_data,
                    "nonce": int(raw_tx.get("nonce", "0x0"), 16),
                    "transactionIndex": int(raw_tx.get("transactionIndex", "0x0"), 16),
                    "value_in_ether": round(ether_value, 8),
                    "type": tx_type_str,
                    "decoded_input": decoded_input_info # Thêm trường đã giải mã
                }
                
                parsed_tx_payload = FullOnchainTransaction(**processed_data)

                response_data_payload = APIResponseData[FullOnchainTransaction](data=[parsed_tx_payload])
                api_response = APIResponse[FullOnchainTransaction](
                    message="New Ethereum Transaction",
                    provider_used="alchemy_ws",
                    data=response_data_payload
                )

                message = {
                    "event": "onchain_transaction_update",
                    "topic": topic,
                    "payload": api_response.model_dump(by_alias=True, exclude_none=True)
                }
                await global_connection_manager.broadcast_to_topic(topic, message)

            except Exception as parsing_error:
                logger.error(f"[Fetcher:{topic}] Error parsing transaction data: {parsing_error}", exc_info=True)

    except asyncio.CancelledError:
        logger.info(f"[Fetcher:{topic}] Alchemy listener task was cancelled.")
    except Exception as e:
        logger.error(f"[Fetcher:{topic}] Unhandled exception in Alchemy listener: {e}", exc_info=True)

async def _ticker_tape_fetcher(topic: str, params: TickerTapeWSParams):
    symbols = sorted([s.strip().upper() for s in params.symbols.split(',') if s.strip()])
    while True:
        try:
            data = await TickerTapeService.get_ticker_tape_batch(symbols)
            api_response = APIResponse[TickerTapeData](message="OK", status="200", data=APIResponseData(data=data or [])).model_dump(exclude_none=True)
            message = create_ws_message("ticker_tape_update", topic, api_response)
            await global_connection_manager.broadcast_to_topic(topic, message)
        except Exception as e:
            logger.error(f"Error in _ticker_tape_fetcher for topic {topic}: {e}", exc_info=True)
        await asyncio.sleep(7)

async def _equity_detail_fetcher(topic: str, params: EquityDetailWsParams):
    while True:
        try:
            async with httpx.AsyncClient() as client:
                data = await equity_detail_service.get_equity_detail(
                    symbol=params.symbol, timeframe=params.timeframe, from_date_str=params.from_date, to_date_str=params.to_date, client=client
                )
            api_response = APIResponse[StockDetailPayload](message="OK", status="200", provider_used="fmp", data=APIResponseData(data=[data] if data else [])).model_dump(exclude_none=True)
            message = create_ws_message("equity_detail_update", topic, api_response)
            await global_connection_manager.broadcast_to_topic(topic, message)
        except Exception as e:
            logger.error(f"Error in _equity_detail_fetcher for topic {topic}: {e}", exc_info=True)
        await asyncio.sleep(7)

async def _list_fetcher(topic: str, params: ListPageWsParams, asset_type: str, redis_client: Optional[aioredis.Redis]):
    symbol_map = {"stocks": stock_list, "etfs": etf_list, "crypto": crypto_list_usd}
    process_func_map = {"stocks": list_service.process_stocks_etfs_batch, "etfs": list_service.process_stocks_etfs_batch, "crypto": list_service.process_crypto_batch}
    
    symbol_list_source = symbol_map.get(asset_type, [])
    process_function = process_func_map.get(asset_type)
    if not process_function: return

    symbols_for_page = symbol_list_source[(params.page - 1) * params.limit : params.page * params.limit]
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                data = await process_function(symbols_for_page, client, redis_client)
                api_response = APIResponse[DiscoveryItemOutput](message="OK", status="200", provider_used=f"fmp_live_{asset_type}", data=APIResponseData(data=data or [])).model_dump(exclude_none=True)
                message = create_ws_message(f"list_{asset_type}_update", topic, api_response)
                await global_connection_manager.broadcast_to_topic(topic, message)
            except Exception as e:
                logger.error(f"Error in _list_fetcher for topic {topic}: {e}", exc_info=True)
            await asyncio.sleep(7)

async def _discovery_fetcher(topic: str, params: DiscoveryWsParams, mover_type: str, redis_client: Optional[aioredis.Redis]):
    service_func_map = {"gainers": discovery_service.get_gainers, "losers": discovery_service.get_losers, "actives": discovery_service.get_actives}
    service_function = service_func_map.get(mover_type)
    if not service_function: return
    
    while True:
        try:
            data = await service_function(limit=params.limit, redis_client=redis_client)
            api_response = APIResponse[DiscoveryItemOutput](message="OK", status="200", provider_used=f"fmp_live_{mover_type}", data=APIResponseData(data=data or [])).model_dump(exclude_none=True)
            message = create_ws_message(f"discovery_{mover_type}_update", topic, api_response)
            await global_connection_manager.broadcast_to_topic(topic, message)
        except Exception as e:
            logger.error(f"Error in _discovery_fetcher for topic {topic}: {e}", exc_info=True)
        await asyncio.sleep(7)

async def _market_overview_fetcher(topic: str, params: MarketRegionParams):
    is_all = params.region == MarketRegionEnum.ALL
    while True:
        try:
            if is_all:
                data = await indices_service.get_and_fetch_all_indices()
            else:
                data = await indices_service.get_and_fetch_indices_by_region(region=params.region.value)
            
            api_response = APIResponse[DiscoveryItemOutput](message="OK", status="200", provider_used="fmp_live_indices", data=APIResponseData(data=data or [])).model_dump(exclude_none=True)
            message = create_ws_message("market_overview_update", topic, api_response)
            await global_connection_manager.broadcast_to_topic(topic, message)
        except Exception as e:
            logger.error(f"Error in _market_overview_fetcher for topic {topic}: {e}", exc_info=True)
        await asyncio.sleep(7)

async def _custom_list_fetcher(topic: str, params: CustomListWsParams, redis_client: Optional[aioredis.Redis]):
    """
    Fetcher để lấy và broadcast dữ liệu cho một danh sách symbols tùy chỉnh từ client.
    """
    process_func_map = {
        "stock": list_service.process_stocks_etfs_batch,
        "etf": list_service.process_stocks_etfs_batch,
        "crypto": list_service.process_crypto_batch
    }
    process_function = process_func_map.get(params.asset_type, list_service.process_stocks_etfs_batch)
    
    update_interval_seconds = 3

    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                logger.info(f"Fetching data for custom list topic '{topic}' ({len(params.symbols)} symbols)...")
                
                data = await process_function(
                    symbols_to_process=params.symbols,
                    client=client,
                    redis_client=redis_client
                )
                api_response = APIResponse[DiscoveryItemOutput](
                    message="OK (live update)",
                    status="200",
                    provider_used=f"fmp_live_custom_{params.asset_type}",
                    data=APIResponseData(data=data or [])
                ).model_dump(exclude_none=True)
                
                message = create_ws_message("custom_list_update", topic, api_response)
                await global_connection_manager.broadcast_to_topic(topic, message)

            except Exception as e:
                logger.error(f"Error in _custom_list_fetcher for topic {topic}: {e}", exc_info=True)
            
            await asyncio.sleep(update_interval_seconds)

async def _custom_chart_fetcher(topic: str, params: CustomListWsParams, redis_client: Optional[aioredis.Redis]):
    """
    Fetcher chỉ để lấy và broadcast dữ liệu chart cho một danh sách symbols tùy chỉnh.
    """

    update_interval_seconds = 60

    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                logger.info(f"Fetching CHART data for custom list topic '{topic}' ({len(params.symbols)} symbols)...")

                charts_map = await list_service._get_batch_charts(
                    symbols_to_process=params.symbols,
                    client=client,
                    redis_client=redis_client,
                    asset_type=params.asset_type
                )
                api_response = {
                    "message": "OK (live chart update)",
                    "status": "200",
                    "provider_used": "fmp_live_custom_chart",
                    "data": charts_map
                }
                
                message = create_ws_message("custom_chart_update", topic, api_response)
                await global_connection_manager.broadcast_to_topic(topic, message)

            except Exception as e:
                logger.error(f"Error in _custom_chart_fetcher for topic {topic}: {e}", exc_info=True)
            
            await asyncio.sleep(update_interval_seconds)