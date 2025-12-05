import chromadb
from chromadb.config import Settings
from openai import OpenAI
import os


class FinancialSituationMemory:
    # Class-level ChromaDB client để share giữa các instances
    _chroma_client = None
    
    @classmethod
    def get_chroma_client(cls):
        """Get or create singleton ChromaDB client"""
        if cls._chroma_client is None:
            # Use persistent storage để giữ collections giữa các runs
            persist_directory = "./chromadb_data"
            os.makedirs(persist_directory, exist_ok=True)
            
            cls._chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
        return cls._chroma_client
    
    def __init__(self, name, config):
        if config["backend_url"] == "http://192.168.1.5:11445":
            self.embedding = "nomic-embed-text"
        else:
            self.embedding = "text-embedding-3-small"
        
        self.client = OpenAI(
            base_url=config["backend_url"],
            api_key=config.get("openai_api_key", "")
        )
        
        # Use singleton client
        self.chroma_client = self.get_chroma_client()
        self.collection_name = name
        
        # Get or create collection với error handling đầy đủ
        self.situation_collection = self._get_or_create_collection(name)
    
    def _get_or_create_collection(self, name):
        """Safely get or create a collection"""
        try:
            # List all existing collections
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            
            if name in existing_collections:
                # Collection exists, get it
                return self.chroma_client.get_collection(name=name)
            else:
                # Collection doesn't exist, create it
                return self.chroma_client.create_collection(name=name)
                
        except Exception as e:
            # Fallback: delete and recreate if corrupted
            print(f"Warning: Issue with collection {name}, recreating: {str(e)}")
            try:
                self.chroma_client.delete_collection(name=name)
            except:
                pass
            return self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding, 
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 1536  # Default embedding size

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice"""
        if not situations_and_advice:
            return
            
        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(f"{self.collection_name}_{offset + i}")
            
            try:
                embedding = self.get_embedding(situation)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding for situation: {e}")
                # Skip this situation if embedding fails
                continue

        if embeddings:  # Only add if we have valid embeddings
            self.situation_collection.add(
                documents=situations,
                metadatas=[{"recommendation": rec} for rec in advice],
                embeddings=embeddings,
                ids=ids,
            )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        try:
            # Check if collection has any data
            if self.situation_collection.count() == 0:
                return []
                
            query_embedding = self.get_embedding(current_situation)

            results = self.situation_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_matches, self.situation_collection.count()),
                include=["metadatas", "documents", "distances"],
            )

            matched_results = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    matched_results.append(
                        {
                            "matched_situation": results["documents"][0][i],
                            "recommendation": results["metadatas"][0][i]["recommendation"],
                            "similarity_score": 1 - results["distances"][0][i],
                        }
                    )

            return matched_results
            
        except Exception as e:
            print(f"Error getting memories: {str(e)}")
            return []

    def reset(self):
        """Reset the collection by deleting all data"""
        try:
            # Delete all items in collection instead of deleting collection itself
            all_ids = [str(i) for i in range(self.situation_collection.count())]
            if all_ids:
                self.situation_collection.delete(ids=all_ids)
        except Exception as e:
            print(f"Error resetting collection: {str(e)}")

# import chromadb
# from chromadb.config import Settings
# from openai import OpenAI


# class FinancialSituationMemory:
#     def __init__(self, name, config):
#         if config["backend_url"] == "http://localhost:11434/v1":
#             self.embedding = "nomic-embed-text"
#         else:
#             self.embedding = "text-embedding-3-small"
#         self.client = OpenAI(base_url=config["backend_url"])
#         self.chroma_client = chromadb.Client(Settings(allow_reset=True))
#         # self.situation_collection = self.chroma_client.create_collection(name=name)
#         print("abc")
#         try:
#             # Try to get existing collection first
#             self.situation_collection = self.chroma_client.get_collection(name=name)
#         except ValueError:
#             # If collection doesn't exist, create it
#             self.situation_collection = self.chroma_client.create_collection(name=name)


#     def get_embedding(self, text):
#         """Get OpenAI embedding for a text"""
        
#         response = self.client.embeddings.create(
#             model=self.embedding, input=text
#         )
#         return response.data[0].embedding

#     def add_situations(self, situations_and_advice):
#         """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

#         situations = []
#         advice = []
#         ids = []
#         embeddings = []

#         offset = self.situation_collection.count()

#         for i, (situation, recommendation) in enumerate(situations_and_advice):
#             situations.append(situation)
#             advice.append(recommendation)
#             ids.append(str(offset + i))
#             embeddings.append(self.get_embedding(situation))

#         self.situation_collection.add(
#             documents=situations,
#             metadatas=[{"recommendation": rec} for rec in advice],
#             embeddings=embeddings,
#             ids=ids,
#         )

#     def get_memories(self, current_situation, n_matches=1):
#         """Find matching recommendations using OpenAI embeddings"""
#         query_embedding = self.get_embedding(current_situation)

#         results = self.situation_collection.query(
#             query_embeddings=[query_embedding],
#             n_results=n_matches,
#             include=["metadatas", "documents", "distances"],
#         )

#         matched_results = []
#         for i in range(len(results["documents"][0])):
#             matched_results.append(
#                 {
#                     "matched_situation": results["documents"][0][i],
#                     "recommendation": results["metadatas"][0][i]["recommendation"],
#                     "similarity_score": 1 - results["distances"][0][i],
#                 }
#             )

#         return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory(name="test_memory", config={"backend_url": "https://api.openai.com/v1"})

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
