from src.models.equity import NewsItemOutput 
from typing import Any, Dict, Optional

class NewsMapper:
    @staticmethod
    def fmp_item_to_news_output(
        fmp_item: Dict[str, Any], 
        news_type: Optional[str] = "UNKNOWN", 
        category: Optional[str] = "GENERAL"
    ) -> Optional[NewsItemOutput]:
        """
        Ánh xạ một item tin tức từ FMP (general hoặc stock_news) sang NewsItemOutput.
        """
        if not fmp_item or not isinstance(fmp_item, dict):
            return None

        try:
            title = fmp_item.get("title")
            news_url = fmp_item.get("url")
            published_date_str = fmp_item.get("publishedDate")

            if not title or not news_url or not published_date_str:
                return None

            return NewsItemOutput(
                type=news_type,
                category=category,
                title=title,
                description=fmp_item.get("text"),
                news_url=news_url,
                image_url=fmp_item.get("image"),
                source_site=fmp_item.get("site"), 
                is_importance=None,
                date=str(published_date_str)
            )
        except Exception as e:
            return None