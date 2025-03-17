## Recurrent neural networks (RNN) 

Before the transformers, RNN (recurrent neural network) was the most popular neural network that use encoder-decoder architecture.

How RNN works:
- The previous output is fed as the input in the next step.
- It's suitable for sequential data (like stock market price)
- RNN was the popular choice for machine translation.
- Disadvantage: RNN will gradually loss of context if the data is long or complex, so RNN only works fine if the text is short.


## Attention Mechanism

Note: this Attention Mechanism needs focus to understand it.

### Definition

Self-attention allows an element to have a connection with other elements, by calculating the relevancy between the element and any other element in a sequence.

"Self" in self-attention means the ability to compute the relationship between different parts of a sequence.


### Attention Weights

The first step implementing self-attention is to compute the attention scores, referred as ω.

Attention weights (or attention scores) are calculated by using dot product of the two elements' embedding. 

For example, if we have this simple embeddings:

```py
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3) 
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)
```

If we want to calculate the attention scores ω between the x^2 and every other input elements, we calculate the dot product of x^2 embeddings to x^n.

For example, the attention score between x^1 ("your") and x^2 ("journey"), we can dot product like this:

```py
score = inputs[0].dot(inputs[1]) # output: tensor([0.9544])

# equal to

score = 0.43 * 0.55 + 0.15 * 0.87 + 0.89 * 0.66 # output: tensor([0.9544])
```

To make it more simple, we can loop it. This example below shows how to get attention scores between x^2 and every other elements.

```py
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0]) # create empty tensor with `vocab_size` sized 
for i, embed in enumerate(inputs):
    attn_scores_2[i] = query.dot(embed) # or torch.dot(embed, query)
print(attn_scores_2)

# Output: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

#### Normalize the attention weights

Now we normalize the attention weights to obtain a number that sum up to 1. Useful for interpretation and maintaining training stability.

In below image, there are symbols:

- $\omega_{xy}$ contains the dot product of `inputs[x]` and `inputs[y]` 
- $\alpha_{xy}$ contains the normalized number of $\omega_{xy}$

![](weights.png)


```py
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum()) # this will output 1

# Output: 
# Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
# Sum: tensor(1.0000)
```

Usually it's advisable to use softmax function for normalization. But this way is better to manage extreme values and better gradient properties during training.