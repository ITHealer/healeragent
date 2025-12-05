# File: src/services/twitter_scraping_service.py

import asyncio
import re
import httpx
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
import aioredis

from src.models.equity import TweetSchema
from src.utils.config import settings
from src.database.models.x_news import TwitterAuthor, Tweet, TwitterCommunity

logger = logging.getLogger(__name__)

class TwitterScrapingService:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis_client = redis_client
        self.api_key = settings.TWITTER_API_IO_KEY
        self.base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"

    async def scrape_and_save_tweets(
        self, 
        username: str, 
        db: Session
    ) -> Optional[List[TweetSchema]]:
        """
        Cào các tweet mới của một user, lưu vào DB, và trả về danh sách các tweet mới.
        """
        if not self.api_key or self.api_key == "Your API Key":
            logger.error("TwitterAPI.io key is not configured.")
            return None

        username_lower = username.lower()
        redis_key_last_check = f"twitter_last_checked:{username_lower}"

        # 1. Lấy thời gian kiểm tra cuối cùng từ Redis
        last_checked_iso = await self.redis_client.get(redis_key_last_check)
        if last_checked_iso:
            last_checked_time = datetime.fromisoformat(last_checked_iso.decode('utf-8'))
        else:
            # Nếu chưa có, bắt đầu từ 1 giờ trước
            last_checked_time = datetime.now(timezone.utc) - timedelta(hours=3)
        
        until_time = datetime.now(timezone.utc)

        # 2. Xây dựng và gọi API
        since_str = last_checked_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
        until_str = until_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
        query = f"from:{username_lower} since:{since_str} until:{until_str}"
        
        params = {"query": query, "queryType": "Latest"}
        headers = {"X-API-Key": self.api_key}
        
        all_raw_tweets = []
        next_cursor = None
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                if next_cursor:
                    params["cursor"] = next_cursor
                
                try:
                    response = await client.get(self.base_url, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()

                    tweets = data.get("tweets", [])
                    if tweets:
                        all_raw_tweets.extend(tweets)
                    
                    if data.get("has_next_page") and data.get("next_cursor"):
                        next_cursor = data.get("next_cursor")
                    else:
                        break
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error while fetching tweets for {username}: {e.response.status_code} - {e.response.text}")
                    return None
                except Exception as e:
                    logger.error(f"Error fetching tweets for {username}: {e}", exc_info=True)
                    return None

        if not all_raw_tweets:
            logger.info(f"No new tweets found for {username} since {last_checked_time.isoformat()}")
            await self.redis_client.set(redis_key_last_check, until_time.isoformat())
            return []
        
        newly_added_tweets = []
        for tweet_data in all_raw_tweets:
            # 3. Xử lý và lưu vào Database
            author_data = tweet_data.get("author")
            if not author_data: continue

            author_id_str = author_data.get("id")
            if not author_id_str: continue
            
            author = db.query(TwitterAuthor).filter(TwitterAuthor.author_id == author_id_str).first()
            if not author:
                author = TwitterAuthor(
                    author_id=author_id_str,
                    username=author_data.get("userName"),
                    name=author_data.get("name")
                )
            
            # Cập nhật thông tin tác giả
            author.followers_count = author_data.get("followers")
            author.following_count = author_data.get("following")
            author.statuses_count = author_data.get("statusesCount")
            author.is_verified = author_data.get("isBlueVerified")
            author.profile_picture_url = author_data.get("profilePicture")
            db.add(author)
            db.commit()
            db.refresh(author)

            # Tạo tweet mới nếu chưa tồn tại
            tweet_id_str = tweet_data.get("id")
            if not tweet_id_str: continue

            existing_tweet = db.query(Tweet).filter(Tweet.tweet_id == tweet_id_str).first()
            if not existing_tweet:
                original_text = tweet_data.get("text", "")
                cleaned_text = re.sub(r'https?://\S+', '', original_text).strip()
                created_at_datetime = datetime.strptime(tweet_data.get("createdAt"), "%a %b %d %H:%M:%S %z %Y")
                new_tweet = Tweet(
                    tweet_id=tweet_id_str,
                    author_id=author.id, # Dùng khóa chính tự tăng của bảng author
                    text=cleaned_text,
                    url=tweet_data.get("url"),
                    created_at=created_at_datetime,
                    retweet_count=tweet_data.get("retweetCount"),
                    reply_count=tweet_data.get("replyCount"),
                    like_count=tweet_data.get("likeCount"),
                    quote_count=tweet_data.get("quoteCount"),
                    view_count=tweet_data.get("viewCount"),
                    hashtags=[h['text'] for h in tweet_data.get("entities", {}).get("hashtags", [])]
                )
                db.add(new_tweet)
                db.commit()
                db.refresh(new_tweet)
                newly_added_tweets.append(TweetSchema.from_orm(new_tweet))

        await self.redis_client.set(redis_key_last_check, until_time.isoformat())
        
        return newly_added_tweets
    
    async def scrape_multiple_accounts(
        self,
        usernames: List[str],
        db: Session
    ):
        """
        Hàm mới: Nhận vào một danh sách usernames và chạy các tác vụ cào dữ liệu đồng thời.
        """
        
        tasks = [self.scrape_and_save_tweets(username, db) for username in usernames]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape for '{usernames[i]}': {result}")
            else:
                logger.info(f"Successfully scraped for '{usernames[i]}', found {len(result) if result else 0} new tweets.")
        
        logger.info("Finished concurrent scraping for all requested accounts.")

    async def scrape_and_save_community_tweets(self, community_id: str, db: Session):
        logger.info(f"Starting scrape for community ID: {community_id}")
        community_url = "https://api.twitterapi.io/twitter/community/tweets"
        
        params = {"community_id": community_id}
        headers = {"X-API-Key": self.api_key}
        
        all_raw_tweets = []
        next_cursor = None
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                if next_cursor:
                    params["cursor"] = next_cursor
                
                try:
                    response = await client.get(community_url, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()

                    tweets = data.get("tweets", [])
                    if tweets:
                        all_raw_tweets.extend(tweets)
                    
                    if data.get("has_next", False) and data.get("next_cursor"):
                        next_cursor = data.get("next_cursor")
                    else:
                        break
                except Exception as e:
                    logger.error(f"Error fetching tweets for community {community_id}: {e}", exc_info=True)
                    return

        if not all_raw_tweets:
            logger.info(f"No new tweets found for community {community_id}.")
            return

        logger.info(f"Found {len(all_raw_tweets)} tweets for community {community_id}. Processing...")
        
        community = db.query(TwitterCommunity).filter(TwitterCommunity.id == community_id).first()
        if not community:
            logger.warning(f"Community with ID {community_id} not found in DB. Tweets will be saved without community link.")

        for tweet_data in all_raw_tweets:
            author_data = tweet_data.get("author")
            if not author_data: continue

            author_id_str = author_data.get("id")
            if not author_id_str: continue

            author = db.query(TwitterAuthor).filter(TwitterAuthor.author_id == author_id_str).first()
            if not author:
                author = TwitterAuthor(
                    author_id=author_id_str,
                    username=author_data.get("userName"),
                    name=author_data.get("name"),
                    created_at=datetime.fromisoformat(author_data.get("createdAt").replace("Z", "+00:00"))
                )
            
            author.followers_count = author_data.get("followers")
            author.following_count = author_data.get("following")
            author.statuses_count = author_data.get("statusesCount")
            author.is_verified = author_data.get("isBlueVerified")
            author.profile_picture_url = author_data.get("profilePicture")
            
            db.add(author)
            db.commit()
            db.refresh(author)

            tweet_id_str = tweet_data.get("id")
            if not tweet_id_str: continue
            
            existing_tweet = db.query(Tweet).filter(Tweet.tweet_id == tweet_id_str).first()
            if not existing_tweet:
                try:
                    created_at_dt = datetime.strptime(tweet_data.get("createdAt"), "%a %b %d %H:%M:%S %z %Y")
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse createdAt format for tweet {tweet_id_str}. Skipping.")
                    continue

                new_tweet = Tweet(
                    tweet_id=tweet_id_str,
                    author_id=author.id,
                    text=tweet_data.get("text"),
                    url=tweet_data.get("url"),
                    created_at=created_at_dt,
                    retweet_count=tweet_data.get("retweetCount"),
                    reply_count=tweet_data.get("replyCount"),
                    like_count=tweet_data.get("likeCount"),
                    quote_count=tweet_data.get("quoteCount"),
                    view_count=tweet_data.get("viewCount"),
                    hashtags=[h['text'] for h in tweet_data.get("entities", {}).get("hashtags", [])]
                )
                
                if community:
                    new_tweet.communities.append(community)

                db.add(new_tweet)
                db.commit()

        logger.info(f"Finished scraping for community {community_id}.")