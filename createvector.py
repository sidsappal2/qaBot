import uuid
import chromadb
from langchain_chroma import Chroma
from chromadb.config import Settings
from utils import Cleaner

class VectorStore:
    """
    A class to manage a persistent vector store using ChromaDB.
    
    This class handles the creation, updating, and initialization of a vector store 
    for document embeddings. It uses ChromaDB's PersistentClient for storing and 
    retrieving embeddings and metadata.

    Attributes:
        client (chromadb.PersistentClient): A client for interacting with ChromaDB.
        collection (chromadb.Collection): The collection to store and manage documents.
    """
    
    def __init__(self):
        """
        Initializes the VectorStore object.
        
        Creates a ChromaDB persistent client and retrieves or creates a collection named "sidsid".
        """
        self.client = chromadb.PersistentClient()  # Persistent storage for vector data.
        self.collection = self.client.get_or_create_collection("sidsid")  # Create or get a collection.

    def loadVectorStore(self, docs, embeddings):
        """
        Load or update the vector store with new documents and embeddings.

        Args:
            docs (list): A list of documents to be added to the vector store. 
                         Each document should have `page_content` (text) and `metadata` attributes.
            embeddings (callable): A function to compute embeddings for the text data.

        Returns:
            vs (Chroma): The initialized Chroma vector store instance.
            
        Raises:
            Exception: If the vector store cannot be initialized.
        """
        # Retrieve all existing documents in the collection.
        all_docs = self.collection.get()

        # Check if the collection is empty.
        if not self.collection.count():
            # If empty, add all the provided documents to the collection.
            for doc in docs:
                self.collection.add(
                    documents=Cleaner.clean_text(doc.page_content),  # Clean the document content.
                    metadatas=doc.metadata,  # Add metadata for the document.
                    ids=[str(uuid.uuid4())]  # Generate a unique ID for each document.
                )
            print("Adding to collection")
        else:
            # If not empty, delete existing documents and add new ones.
            self.collection.delete(ids=all_docs['ids'])  # Delete all documents by their IDs.
            for doc in docs:
                self.collection.add(
                    documents=Cleaner.clean_text(doc.page_content),
                    metadatas=doc.metadata,
                    ids=[str(uuid.uuid4())]
                )
            print("Adding more to collection")

        # Initialize the vector store with Chroma.
        vs = Chroma(
            client=self.client,  # Use the existing ChromaDB client.
            collection_name="sidsid",  # Name of the collection to interact with.
            embedding_function=embeddings,  # Function to generate embeddings.
        )

        # Check if the vector store was initialized successfully.
        if vs is None:
            print("VectorStore has not been initialized. Please process the URLs first.>>createvector.py")
        else:
            print("Vector store loaded successfully")
            return vs
