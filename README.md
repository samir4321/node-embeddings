node embeddings
---
Node embeddings for spans in the trace topology graph.
Two approaches here

  * **node2vec** like word2vec but for nodes in a graph instead of words in a sentence. the 'sentence' becomes a random walk about a node
  * **svd** svd embedding again using the random walk about its node as context. much faster to train than the node2vec embedding

  