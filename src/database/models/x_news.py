
from src.database.models.base import Base
from sqlalchemy import Column, String, Integer, BigInteger, Boolean, DateTime, ForeignKey, Table, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY

tweet_community_association = Table('tweet_community_association', Base.metadata,
    Column('tweet_id', String, ForeignKey('tweets.tweet_id'), primary_key=True),
    Column('community_id', String, ForeignKey('twitter_communities.id'), primary_key=True)
)

class TwitterCommunity(Base):
    __tablename__ = "twitter_communities"

    id = Column(String, primary_key=True, index=True, comment="Community ID từ Twitter API")
    name = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True))
    member_count = Column(Integer)
    
    # Mối quan hệ nhiều-nhiều với Tweet
    tweets = relationship(
        "Tweet",
        secondary=tweet_community_association,
        back_populates="communities"
    )

class TwitterAuthor(Base):
    __tablename__ = "twitter_authors"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    author_id = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False, index=True, comment="Username (@handle)")
    name = Column(String, nullable=False, comment="Display Name")
    is_verified = Column(Boolean, default=False)
    profile_picture_url = Column(String)
    followers_count = Column(BigInteger)
    following_count = Column(BigInteger)
    statuses_count = Column(BigInteger)
    created_at = Column(DateTime(timezone=True))
    
    tweets = relationship("Tweet", back_populates="author")

class Tweet(Base):
    __tablename__ = "tweets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tweet_id = Column(String, unique=True, nullable=False, comment="Twitter Tweet ID")
    # Khóa ngoại trỏ đến bảng authors
    author_id = Column(Integer, ForeignKey("twitter_authors.id"), nullable=False, index=True)
    
    text = Column(Text, nullable=False)
    url = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    
    retweet_count = Column(BigInteger, default=0)
    reply_count = Column(BigInteger, default=0)
    like_count = Column(BigInteger, default=0)
    quote_count = Column(BigInteger, default=0)
    view_count = Column(BigInteger)
    
    # PostgreSQL hỗ trợ kiểu mảng, rất tiện lợi để lưu hashtags
    hashtags = Column(ARRAY(String))
    
    # Mối quan hệ: Một tweet thuộc về một author
    author = relationship("TwitterAuthor", back_populates="tweets")
    communities = relationship(
        "TwitterCommunity",
        secondary=tweet_community_association,
        back_populates="tweets"
    )