import httpx
import logging
from datetime import datetime

from src.models.equity import PaginatedData
from src.services.list_service import ListService
from src.services.discovery_service import DiscoveryService 
from src.helpers.redis_cache import get_redis_client_for_scheduler, set_paginated_cache 
from src.utils.config import settings
from src.utils.logger.set_up_log_dataFMP import setup_logger

PREFETCH_LIMIT = 100
STOCKS_CACHE_KEY = "prefetched_stocks_data_v2"
ETFS_CACHE_KEY = "prefetched_etfs_data_v2"
CRYPTO_CACHE_KEY = "prefetched_crypto_data_v2"
CACHE_TTL = 60 * 60

PREFETCH_DISCOVERY_LIMIT = 100
GAINERS_CACHE_KEY = "prefetched_discovery_gainers"
LOSERS_CACHE_KEY = "prefetched_discovery_losers"
ACTIVES_CACHE_KEY = "prefetched_discovery_actives"
CACHE_DISCOVERY_TTL = 60 * 60

DEFAULT_SCREENER_CACHE_KEY = "prefetched_default_screener"
PREFETCH_SCREENER_LIMIT = 100
CACHE_SCREENER_TTL = 60 * 60

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

logging.getLogger("apscheduler").setLevel(logging.WARNING)
logger = setup_logger(__name__, log_level=logging.INFO)


class JobMonitor:
    """Monitor job execution"""
    
    @staticmethod
    def log_start(job_name: str):
        # logger.info(f"[{job_name}] Starting")
        return datetime.now()
    
    @staticmethod
    def log_end(job_name: str, start_time: datetime, success: bool = True):
        duration = (datetime.now() - start_time).total_seconds()
        # status = "✅" if success else "❌"
        # logger.info(f"{status} [{job_name}] {duration:.2f}s")
        if duration > 50:
            logger.warning(f"⚠️ [{job_name}] Slow: {duration:.2f}s")


async def prefetch_data_job(job_name, symbol_source, service_func, cache_key, cache_ttl):
    """Job with monitoring and better timeouts"""
    start = JobMonitor.log_start(job_name)
    
    async with get_redis_client_for_scheduler() as redis_client:
        if not redis_client:
            logger.error(f"[{job_name}] No Redis")
            JobMonitor.log_end(job_name, start, False)
            return

        try:
            symbols = symbol_source[:PREFETCH_LIMIT]
            if not symbols:
                JobMonitor.log_end(job_name, start, False)
                return

            timeout = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)
            limits = httpx.Limits(max_keepalive_connections=30, max_connections=100, keepalive_expiry=20.0)

            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                data = await service_func(symbols_to_process=symbols, client=client, redis_client=redis_client)

            if data:
                result = PaginatedData(totalRows=PREFETCH_LIMIT, data=data)
                await set_paginated_cache(redis_client, cache_key, result, expiry=cache_ttl)
                logger.info(f"[{job_name}] Cached {len(data)} items")
                JobMonitor.log_end(job_name, start, True)
            else:
                JobMonitor.log_end(job_name, start, False)
                
        except Exception as e:
            logger.error(f"[{job_name}] Error: {e}")
            JobMonitor.log_end(job_name, start, False)


async def prefetch_stock_data():
    list_service = ListService()
    await prefetch_data_job(
        job_name=f"Fetch {PREFETCH_LIMIT} STOCK symbol",
        symbol_source=stock_list,
        service_func=list_service.process_stocks_etfs_batch,
        cache_key=STOCKS_CACHE_KEY,
        cache_ttl=CACHE_TTL
    )

async def prefetch_etf_data():
    list_service = ListService()
    await prefetch_data_job(
        job_name=f"Fetch {PREFETCH_LIMIT} ETF symbol",
        symbol_source=etf_list,
        service_func=list_service.process_stocks_etfs_batch,
        cache_key=ETFS_CACHE_KEY,
        cache_ttl=CACHE_TTL
    )
    
async def prefetch_crypto_data():
    list_service = ListService()
    await prefetch_data_job(
        job_name=f"Fetch {PREFETCH_LIMIT} CRYPTO symbol",
        symbol_source=crypto_list_usd,
        service_func=list_service.process_crypto_batch_scheduler,
        cache_key=CRYPTO_CACHE_KEY,
        cache_ttl=CACHE_TTL
    )


async def prefetch_discovery_job(discovery_type: str):
    """
    Prefetch discovery data for a given type (gainers, losers, actives).
    
    Parameters:
    discovery_type (str): The type of discovery data to prefetch.
    
    Returns:
    None
    """
    
    job_name = f"Discovery-{discovery_type.upper()}"
    start = JobMonitor.log_start(job_name)
    
    service = DiscoveryService()
    service_map = {"gainers": service.get_gainers, "losers": service.get_losers, "actives": service.get_actives}
    cache_map = {"gainers": GAINERS_CACHE_KEY, "losers": LOSERS_CACHE_KEY, "actives": ACTIVES_CACHE_KEY}
    
    async with get_redis_client_for_scheduler() as redis_client:
        if not redis_client:
            JobMonitor.log_end(job_name, start, False)
            return
        
        try:
            data = await service_map[discovery_type](limit=PREFETCH_DISCOVERY_LIMIT, redis_client=redis_client)
            
            if data:
                result = PaginatedData(totalRows=PREFETCH_DISCOVERY_LIMIT, data=data)
                await set_paginated_cache(redis_client, cache_map[discovery_type], result, expiry=CACHE_DISCOVERY_TTL)
                JobMonitor.log_end(job_name, start, True)
            else:
                JobMonitor.log_end(job_name, start, False)
        except Exception as e:
            logger.error(f"[{job_name}] Error: {e}")
            JobMonitor.log_end(job_name, start, False)


# Discovery gainers, losers, actives data scheduler jobs 
async def prefetch_gainers_data(): await prefetch_discovery_job("gainers")
async def prefetch_losers_data(): await prefetch_discovery_job("losers")
async def prefetch_actives_data(): await prefetch_discovery_job("actives")


async def prefetch_default_screener_data():
    job_name = "SCREENER"
    start = JobMonitor.log_start(job_name)
    
    async with get_redis_client_for_scheduler() as redis_client:
        if not redis_client:
            JobMonitor.log_end(job_name, start, False)
            return

        try:
            data = await DiscoveryService().get_screener_http_compatible(limit=PREFETCH_SCREENER_LIMIT)
            if data:
                result = PaginatedData(totalRows=PREFETCH_SCREENER_LIMIT, data=data)
                await set_paginated_cache(redis_client, DEFAULT_SCREENER_CACHE_KEY, result, expiry=CACHE_SCREENER_TTL)
                logger.info(f"[{job_name}] Cached {len(data)} items")
                JobMonitor.log_end(job_name, start, True)
            else:
                JobMonitor.log_end(job_name, start, False)
        except Exception as e:
            logger.error(f"[{job_name}] Error: {e}")
            JobMonitor.log_end(job_name, start, False)