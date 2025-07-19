import os.path
import uuid
import random
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams
from src import file_helper
from src import easysalon
from src import global_vars

def create_embedding_function(url: str, key: str, model: str) -> OpenAIEmbeddings:
    """
    Create an OpenAI embedding function for Qdrant using LangChain.
    
    Args:
        url (str): OpenAI API base URL
        key (str): OpenAI API key
        model (str): OpenAI embedding model name
        
    Returns:
        OpenAIEmbeddings: LangChain OpenAI embedding function
    """
    return OpenAIEmbeddings(
        openai_api_key=key,
        openai_api_base=url,
        model=model
    )


def init_db(url: str, key: str, model: str, qdrant_api_key: str, qdrant_url: str = "localhost", qdrant_port: int = 6333) -> tuple:
    """
    Initialize Qdrant database client and embedding function.
    
    Args:
        url (str): OpenAI API base URL
        key (str): OpenAI API key
        model (str): OpenAI embedding model name
        qdrant_url (str): Qdrant server URL (default: localhost)
        qdrant_port (int): Qdrant server port (default: 6333)
        
    Returns:
        tuple: (QdrantClient, OpenAIEmbeddings) - Database client and embedding function
    """
    db_client = setup_qdrant_client(qdrant_api_key, qdrant_url, qdrant_port)
    fn_embedding = create_embedding_function(url, key, model)
    return db_client, fn_embedding


def setup_qdrant_client(api_key: str, qdrant_url: str = "localhost", qdrant_port: int = 6333, embedding_model: str = "text-embedding-3-small") -> QdrantClient:
    """
    Initialize Qdrant client for vector database operations.
    
    Args:
        qdrant_url (str): Qdrant server URL
        qdrant_port (int): Qdrant server port
        
    Returns:
        QdrantClient: Initialized Qdrant client
        
    Raises:
        Exception: If Qdrant client setup fails
    """
    try:
        client = QdrantClient(url=qdrant_url, port=qdrant_port, api_key=api_key)
        return client
    except Exception as e:
        print(f"Error setting up Qdrant client: {str(e)}")
        return None


def create_collection(client: QdrantClient, embedding_fn: OpenAIEmbeddings, collection_name: str, 
                     vector_size: int = 1536, distance: Distance = Distance.COSINE) -> bool:
    """
    Create or recreate a collection in Qdrant with specified parameters.
    
    Args:
        client (QdrantClient): Qdrant client instance
        embedding_fn (OpenAIEmbeddings): Embedding function (used for vector size reference)
        collection_name (str): Name of the collection to create
        vector_size (int): Size of the embedding vectors (default: 1536 for OpenAI)
        distance (Distance): Distance metric for similarity search (default: COSINE)
        
    Returns:
        bool: True if collection created successfully, False otherwise
    """
    try:
        # Delete collection if it exists
        try:
            client.delete_collection(collection_name)
        except:
            pass  # Collection doesn't exist, which is fine
            
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        return True
    except Exception as e:
        print(f"Error creating collection {collection_name}: {str(e)}")
        return False


def get_vectorstore(client: QdrantClient, embedding_fn: OpenAIEmbeddings, collection_name: str) -> Qdrant:
    """
    Get or create a LangChain Qdrant vectorstore instance for the specified collection.
    
    Args:
        client (QdrantClient): Qdrant client instance
        embedding_fn (OpenAIEmbeddings): LangChain embedding function
        collection_name (str): Name of the collection
        
    Returns:
        Qdrant: LangChain Qdrant vectorstore instance
    """
    try:
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_fn
        )
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore for collection {collection_name}: {str(e)}")
        return None


def init_data(db_client: QdrantClient, embedding_fn: OpenAIEmbeddings, run_path: str) -> Dict[str, Qdrant]:
    """
    Initialize and populate Qdrant collections with location, restaurant, and activity data.
    
    Args:
        db_client (QdrantClient): Qdrant client instance
        embedding_fn (OpenAIEmbeddings): LangChain embedding function
        run_path (str): Path to the project root directory
        
    Returns:
        Dict[str, Qdrant]: Dictionary containing vectorstore instances for each collection
    """
    salon = easysalon.Easysalon(api_key=global_vars.EASYSALON_API_KEY)
    vectorstores = {}
    collection_name = global_vars.QDRANT_SALON_DATA_COLLECTION

    try:
        create_collection(db_client, embedding_fn, collection_name)
    except Exception:
        print("Collection already exists")

    collection = get_vectorstore(db_client, embedding_fn, collection_name)

    services = salon.get_services()
    products = salon.get_products()
    packages = salon.get_packages()
    branchs = salon.get_branches()

    points = []
    for item in services:
        branch = ""
        for br in branchs:
            if br["id"] in item["branchIds"]:
                branch+= f"""{br["name"]}\n"""
        doc_text = f"""
            Id: {item["id"]}
            Name: {item['name']}
            Price: {item["price"]}
            Description: {item["description"]}
            Time: {item['time']} min
            Type: Service
            Branch: {branch}
            """
        emb = embedding_fn.embed_query(doc_text.strip())
        points.append(
            PointStruct(
                id=item["id"],
                vector=emb,
                payload={
                    "text": doc_text.strip(),
                    "metadata": {
                        "id": item["id"],
                        "type": "package",
                        "categoryId": item['categoryId'],
                        "branchIds": item['branchIds']
                    }
                }
            )
        )

    for item in products:
        branch = ""
        for br in branchs:
            if br["id"] in item["branchIds"]:
                branch += f"""{br["name"]}\n"""
        doc_text = f"""
            Id: {item["id"]}
            Code: {item['code']}
            Name: {item['name']}
            Volume: {item['volume']}
            Price: {item["price"]}
            Description: {item["description"]}
            Type: Product
            Branch: {branch}
            """
        emb = embedding_fn.embed_query(doc_text.strip())
        points.append(
            PointStruct(
                id=item["id"],
                vector=emb,
                payload={
                    "text": doc_text.strip(),
                    "metadata": {
                        "id": item["id"],
                        "type": "product",
                        "categoryId": item['categoryId'],
                        "branchIds": item['branchIds']
                    }
                }
            )
        )

    for item in packages:
        branch = ""
        for br in branchs:
            if br["id"] in item["branchIds"]:
                branch += f"""{br["name"]}\n"""
        doc_text = f"""
            Id: {item["id"]}
            Code: {item['code']}
            Name: {item['name']}
            Number Of Use: {item['numberOfUse']}
            Use duration: {item['usedInMonth']} month
            Price: {item["price"]}
            Description: {item["description"]}
            Type: Package
            Branch: {branch}
            """
        emb = embedding_fn.embed_query(doc_text.strip())
        points.append(
            PointStruct(
                id=item["id"],
                vector=emb,
                payload={
                    "text": doc_text.strip(),
                    "metadata": {
                        "id": item["id"],
                        "type": "package",
                        "categoryId": item['categoryId'],
                        "branchIds": item['branchIds']
                    }
                }
            )
        )

    if points:
        db_client.upsert(
            collection_name=collection_name,
            points=points
        )
    vectorstores[global_vars.QDRANT_SALON_DATA_COLLECTION] = collection

    # Initialize pretrained questions data
    questions_vectorstore = init_pretrained_questions(db_client, embedding_fn, run_path)
    if questions_vectorstore:
        vectorstores["pretrain_question"] = questions_vectorstore
    
    return vectorstores


def query(vectorstore: Qdrant, user_input: str, n_results: int = 3, 
          metadata_filter: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Query the Qdrant vectorstore for similar documents based on user input.
    
    Args:
        vectorstore (Qdrant): LangChain Qdrant vectorstore instance
        user_input (str): User query text
        n_results (int): Number of results to return (default: 3)
        metadata_filter (Optional[Dict[str, Any]]): Metadata filter for search results
        
    Returns:
        List[str]: List of document contents matching the query
        
    Raises:
        Exception: If query execution fails
    """
    try:
        if metadata_filter:
            # Use similarity search with metadata filter
            results = vectorstore.similarity_search(
                query=user_input,
                k=n_results,
                filter=metadata_filter
            )
        else:
            # Use basic similarity search
            results = vectorstore.similarity_search(
                query=user_input,
                k=n_results
            )
        
        # Extract document contents
        documents = [doc.page_content for doc in results]
        return documents
        
    except Exception as e:
        return [f"Error generating recommendation: {str(e)}"]


def search(
    db_client: QdrantClient,
    embedding_fn: OpenAIEmbeddings,
    collection_name: str,
    query_text: str,
    type_filter: Optional[str] = None,
    limit: int = 5
) -> List[str]:
    """
    Query Qdrant collection for salon data matching the user input text.

    Args:
        db_client (QdrantClient): Qdrant client instance
        embedding_fn (OpenAIEmbeddings): LangChain embedding function
        collection_name (str): Name of the Qdrant collection (e.g., 'salon_data')
        query_text (str): User input text to search (e.g., 'combo services')
        type_filter (Optional[str]): Filter by item type (e.g., 'package', 'service', 'product')
        limit (int): Maximum number of results to return

    Returns:
        List[Dict]: List of matching items with their text and metadata
    """
    try:
        # Validate query text
        if not query_text or not isinstance(query_text, str):
            print("Error: Query text must be a non-empty string")
            return []

        # Generate embedding for the query text
        query_embedding = embedding_fn.embed_query(query_text.strip())
        if not query_embedding:
            print("Error: Failed to generate embedding for query text")
            return []

        # Build filter (if type_filter is provided)
        scroll_filter = None
        if type_filter:
            scroll_filter = {
                "must": [
                    {"key": "metadata.type", "match": {"value": type_filter}}
                ]
            }

        # Perform vector search
        results = db_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=scroll_filter,
            limit=limit,
            with_payload=True
        )

        # Format results
        formatted_results = [
            {
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {}),
                "score": point.score
            }
            for point in results
        ]
        return formatted_results

    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return []


def init_pretrained_questions(db_client: QdrantClient, embedding_fn: OpenAIEmbeddings, run_path: str) -> Qdrant:
    """
    Initialize and populate Qdrant collection with pretrained questions data.
    
    Args:
        db_client (QdrantClient): Qdrant client instance
        embedding_fn (OpenAIEmbeddings): LangChain embedding function
        run_path (str): Path to the project root directory
        
    Returns:
        Qdrant: Vectorstore instance for the pretrained questions collection
    """
    try:
        data_path = os.path.join(run_path, "Data")
        questions_data = file_helper.load_json_data(os.path.join(data_path, "pretrained_questions.json"))
        
        # Create and populate pretrained questions collection
        create_collection(db_client, embedding_fn, "pretrain_question")
        questions_vectorstore = get_vectorstore(db_client, embedding_fn, "pretrain_question")
        
        questions_documents = []
        for item in questions_data["questions"]:
            # Create separate documents for English and Vietnamese questions
            
            # English question document
            english_doc_text = f"""
            Type: Pretrained Question
            Language: English
            Category: {item['category']}
            Question: {item['english']}
            Question ID: {item['id']}
            """
            
            english_doc = Document(
                page_content=english_doc_text.strip(),
                metadata={
                    "id": f"{item['id']}_en",
                    "original_id": item["id"],
                    "type": "pretrained_question",
                    "language": "english",
                    "category": item["category"],
                    "question": item["english"],
                    "translation": item["vietnamese"]
                }
            )
            questions_documents.append(english_doc)
            
            # Vietnamese question document
            vietnamese_doc_text = f"""
            Type: Pretrained Question
            Language: Vietnamese
            Category: {item['category']}
            Question: {item['vietnamese']}
            Question ID: {item['id']}
            """
            
            vietnamese_doc = Document(
                page_content=vietnamese_doc_text.strip(),
                metadata={
                    "id": f"{item['id']}_vi",
                    "original_id": item["id"],
                    "type": "pretrained_question",
                    "language": "vietnamese",
                    "category": item["category"],
                    "question": item["vietnamese"],
                    "translation": item["english"]
                }
            )
            questions_documents.append(vietnamese_doc)
        
        # Add documents to the vectorstore
        questions_vectorstore.add_documents(questions_documents)
        return questions_vectorstore
        
    except Exception as e:
        print(f"Error initializing pretrained questions collection: {str(e)}")
        return None


def query_pretrained_questions(vectorstore: Qdrant, user_input: str, language: str = None, 
                              category: str = None, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Query pretrained questions with language and category filtering.
    
    Args:
        vectorstore (Qdrant): Qdrant vectorstore instance for pretrained questions
        user_input (str): User query text
        language (str, optional): Filter by language ('english' or 'vietnamese')
        category (str, optional): Filter by category (e.g., 'booking', 'pricing')
        n_results (int): Number of results to return (default: 5)
        
    Returns:
        List[Dict[str, Any]]: List of matching questions with metadata
    """
    try:
        # Build metadata filter
        # metadata_filter = {"type": "pretrained_question"}
        # if language:
        #     metadata_filter["language"] = language
        # if category:
        #     metadata_filter["category"] = category
        
        # Perform similarity search
        results = vectorstore.similarity_search(
            query=user_input,
            k=n_results
        )
        
        # Extract and format results
        formatted_results = []
        for doc in results:
            result = {
                "question": doc.metadata.get("question", ""),
                "translation": doc.metadata.get("translation", ""),
                "category": doc.metadata.get("category", ""),
                "language": doc.metadata.get("language", ""),
                "original_id": doc.metadata.get("original_id", ""),
                "content": doc.page_content
            }
            formatted_results.append(result)
        
        return formatted_results
        
    except Exception as e:
        return [{"error": f"Error querying pretrained questions: {str(e)}"}]


def get_suggested_questions(vectorstore: Qdrant, user_input: str, language: str = None, 
                           exclude_categories: List[str] = None) -> List[str]:
    """
    Get suggested questions based on user input using similarity search.
    
    Args:
        vectorstore (Qdrant): Qdrant vectorstore instance for pretrained questions
        user_input (str): User's current input/query
        language (str, optional): Preferred language ('english' or 'vietnamese')
        exclude_categories (List[str], optional): Categories to exclude from suggestions
        
    Returns:
        List[str]: List of suggested questions (3-5 random suggestions)
    """
    try:
        # Random number of results between 3 and 5
        n_results = random.randint(3, 5)
        intitial_results = 15  # Get up to 15 results, but not more than total
        # Perform similarity search
        results = vectorstore.similarity_search(
            query=user_input,
            k=intitial_results
        )
        
        random.sample(results, n_results)  # Randomly select from the results
        
        # print(f"Qdrant results: {results}")  # Debugging line to check results
        # Filter out excluded categories if specified
        filtered_results = []
        for doc in results:
            category = doc.metadata.get("category", "")
            if not exclude_categories or category not in exclude_categories:
                filtered_results.append(doc)
        
        # Extract questions and remove duplicates (same original_id)
        seen_ids = set()
        unique_questions = []
        
        for doc in filtered_results:
            original_id = doc.metadata.get("original_id")
            question = doc.metadata.get("question", "")
            
            if original_id not in seen_ids and question:
                seen_ids.add(original_id)
                unique_questions.append(question)
        
        # Randomly select n_results from the unique questions
        if len(unique_questions) <= n_results:
            return unique_questions
        else:
            return random.sample(unique_questions, n_results)
        
    except Exception as e:
        # Fallback to default suggestions if there's an error
        print(f"Error getting suggested questions: {str(e)}")
        return []
    return []
