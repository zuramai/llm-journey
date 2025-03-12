# Creating token embeddings

In PyTorch, embeddings initially a random float numbers packed in a tensor. To instantiate the embedding layer, we need two variable:
- `vocab_size`  is the number of tokens. Represented as rows.
- `output_dim` is the dimension of the embedding. Represented as columns.

Usage:
```py
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer)
```
Output:
```
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
```

For example, if I have a tensor containing a token ids, `tensor([2, 3, 5, 1])`, it will return according to the rows order.

```py
input_ids = torch.tensor([2, 3, 5, 1])
print(embedding_layer(input_ids))
```

it will print:

```
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
```

### Self-attention mechanism

Embeddings are meant to be the input of LLM, but self-attention mechanism don't know the position/order of the tokens. Because regardless the position of token in a sequence, it will return the same vector. For example, if we have the same token in a sequence, it will repeat like this:

```py
input_ids = tensor([2, 3, 5, 2])
print(embedding_layer(input_ids))
```

Output:

```
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 1.2753, -0.2010, -0.1606]], grad_fn=<EmbeddingBackward0>)
```

Notice the tensor in row 0 and row 3, they same because they are the same token.