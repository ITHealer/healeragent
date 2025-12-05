import pandas as pd
from requests_html import HTMLSession

class GetDate:
    @staticmethod
    def force_float(elt):
        
        try:
            return float(elt)
        except:
            return elt
        
    @staticmethod
    def _convert_to_numeric(s):

        if "M" in s:
            s = s.strip("M")
            return GetDate.force_float(s) * 1_000_000
        
        if "B" in s:
            s = s.strip("B")
            return GetDate.force_float(s) * 1_000_000_000
        
        return GetDate.force_float(s)
    
    @staticmethod
    def _raw_get_daily_info(site):
        
        session = HTMLSession()
        
        resp = session.get(site)
        
        tables = pd.read_html(resp.html.raw_html)  
        
        df = tables[0].copy()
        
        df.columns = tables[0].columns
        if "52 Wk Range" in df.columns:
            del df["52 Wk Range"]
            del df["Unnamed: 2"]
            del df["52 Wk Change %"]

        df["Change %"] = df["Change %"].astype(str)
        cleaned_series = df["Change %"].str.strip('%+ ')
        cleaned_series = cleaned_series.str.replace(',', '', regex=False)
        df["Change %"] = pd.to_numeric(cleaned_series, errors='coerce')
        
        fields_to_change = [x for x in df.columns.tolist() if "Vol" in x \
                            or x == "Market Cap"]
        for field in fields_to_change:
            if type(df[field][0]) == str:
                df.dropna(subset=[field], inplace=True)
                df[field] = df[field].map(GetDate._convert_to_numeric)
        session.close()
        return df

    def get_day_most_active(count: int = 100):

        return GetDate._raw_get_daily_info(f"https://finance.yahoo.com/most-active?offset=0&count={count}")


    def get_day_gainers(count: int = 100):

        return GetDate._raw_get_daily_info(f"https://finance.yahoo.com/gainers?offset=0&count={count}")


    def get_day_losers(count: int = 100):

        return GetDate._raw_get_daily_info(f"https://finance.yahoo.com/losers?offset=0&count={count}")