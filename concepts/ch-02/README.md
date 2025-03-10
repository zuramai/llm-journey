# Chapter 02


### Embeddings

- There are many kind of text embeddings: **word embeddings**, **sentence embeddings**, **paragraph embeddings**, and **whole document embeddings**. 
- *Retrieval-augmented generation* (RAG) is using sentence embeddings to retrieve data from given knowledge base.
- The most popular embedding method is using Word2Vec
- Word2Vec embedding convert word to numerical representation. If embeddings are two-dimensional, words with similar concepts are closer to each other, e.g. "apple", "oranges", and "strawberry" will be closer to each other and far from the word "Indonesia".
- High-dimensional embeddings are hard to grasp because human brain is limited to perceive three-dimension or fewer.
- Smallest GPT-2 model (117M and 125M) is using 768 dimension of embeddings. Meanwhile GPT-3 model (175B) is using 12288 dimensions.
