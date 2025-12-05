import csv
from src.database.models.stock_schemas import NewsSchema
from datetime import datetime
import yfinance as yf

def is_duplicate_venue(venue_name: str, seen_names: set) -> bool:
    return venue_name in seen_names

def is_complete_venue(venue: dict, required_keys: list) -> bool:
    return all(key in venue for key in required_keys)

def save_venues_to_csv(venues: list, filename: str):
    if not venues:
        print("No venues to save.")
        return

    # Use field names from the Venue model
    fieldnames = NewsSchema.model_fields.keys()

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(venues)
    print(f"Saved {len(venues)} venues to '{filename}'.")

def get_current_timestamp_string():
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp_str

def get_asset_type(symbol: str):
    if symbol.startswith('^'):
        return 'INDEX'
    elif "=X" in symbol:
        return "CURRENCY"
    elif "=F" in symbol:
        return "FUTURES"
    else:
        return "UNKNOWN"
    
def get_asset_name(symbol: str):
    if "=" in symbol and symbol.endswith("=F"):
        return "Futures Contract" 
    
# Lấy tin tức từ Yahoo Finance
async def get_news(symbol: str, limit: int = 5):
    # Tạo đối tượng Ticker từ yfinance
    ticker = yf.Ticker(symbol)
    try:
        # Lấy tin tức từ Yahoo Finance
        news_items = ticker.get_news()
        results = []

        # Giới hạn số lượng bài viết
        news_items = news_items[:limit]

        for d in news_items:
            new_content: dict = {}
            content = d.get("content")

            if not content:
                continue

            # Lấy ảnh từ thumbnail nếu có
            if thumbnail := content.get("thumbnail"):
                images = thumbnail.get("resolutions")
                if images:
                    new_content["images"] = [
                        {k: str(v) for k, v in img.items()} for img in images
                    ]

            new_content["date"] = content.get("pubDate")
            new_content["title"] = content.get("title")
            new_content["source"] = content.get("provider", {}).get("displayName")
            
            new_content["url"] = content.get("canonicalUrl", {}).get("url")
            description = content.get("description")
            summary = content.get("summary")

            if description:
                new_content["text"] = description
            elif summary:
                new_content["text"] = summary

            results.append(new_content)

        return results

    except Exception as e:
        return {"message": f"Error retrieving news: {str(e)}", "status": "500", "data": {}}
