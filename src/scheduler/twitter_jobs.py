import asyncio
import logging
from src.database import get_postgres_db 
from src.helpers.redis_cache import get_redis_client_for_scheduler
from src.services.twitter_scraping_service import TwitterScrapingService
from src.utils.logger.set_up_log_dataFMP import setup_logger

logger = setup_logger(__name__, log_level=logging.INFO)

TARGET_ACCOUNTS = [
    "elonmusk",
    "VitalikButerin",
    "a16z",
    "WatcherGuru",
    "unusual_whales"
]

TARGET_COMMUNITIES = [
    "1494444533033291776", # Community về AI
    "1583852232984825856", # Community về Crypto

    # Crypto & Web3
    "1484950343274614785", # NFT Community
    "1501191599816400896", # Solana Community
    "1643534322852274176", # De.Fi Community (Security in DeFi)
    "1496150495687114753", # Web3 Developers

    # Tài chính & Chứng khoán
    "1585663738183987200", # Stock Market News & Alerts
    "1501289419830538240", # Swing Trading
    "1598015793395998720", # Unusual Whales Community (Theo dõi giao dịch lớn)

    # Công nghệ & AI
    "1557053535614836737", # Artificial Intelligence (AI)
    "1620131804709494784", # Machine Learning
    "1501197478059040768"  # Tech & Startups
]

async def scheduled_community_scrape_job():

    db_connection = get_postgres_db()
    with db_connection.session_scope() as db:
        try:
            scraper = TwitterScrapingService(redis_client=None) 
            
            tasks = [scraper.scrape_and_save_community_tweets(community_id, db) for community_id in TARGET_COMMUNITIES]
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Scheduler: An error occurred during the community scrape job: {e}", exc_info=True)

    logger.info("Scheduler: Finished scheduled community scrape job.")

async def scheduled_twitter_scrape_job():
    """
    Đây là hàm công việc (job) mà APScheduler sẽ gọi theo định kỳ,
    sử dụng đúng pattern quản lý session database.
    """
    logger.info("Scheduler: Starting scheduled Twitter scrape job...")
    
    db_connection = get_postgres_db()
    
    async with get_redis_client_for_scheduler() as redis_client:
        if not redis_client:
            logger.error("Scheduler: Could not get Redis client for Twitter job. Aborting.")
            return
    
        try:
            with db_connection.session_scope() as db:
                logger.info("Scheduler: Database session created for Twitter job.")
                
                scraper = TwitterScrapingService(redis_client=redis_client)
                await scraper.scrape_multiple_accounts(usernames=TARGET_ACCOUNTS, db=db)

        except Exception as e:
            logger.error(f"Scheduler: An error occurred during the scheduled Twitter scrape job: {e}", exc_info=True)

    logger.info("Scheduler: Finished scheduled Twitter scrape job.")