# Chapter 01

The chapter 01 in the book explains about the definitions of LLM.

## Keypoints (in my own words)

- Large Language Model succeed because of transformer architecture. Not all LLM are built on top of transformer architecture.
- The word "large" in LLM is just to show the size of LLM

## About LLM and Deep Learning

![superset](images/superset.png)

## Application of LLM

- This chapter explains about LLM capabilities, like text generation, content creation, context understanding.

## Stages of building and using LLM

- Most LLMs developed using PyTorch.
- Explains about the benefits of developing our own LLM, either building from scratch or fine-tuning existing LLM model to our own domain-specific task.
- Local LLM benefits the user low latency.

![process](images/process.png)

To oversimplify the process of developing LLM, first we gather data (tons of data), and then we train it (pretraining), resulting as pretrained base model. But the base model has limited capabilities, only like text completion. We need to train it again with labeled dataset. For example, if you want to make the model to be able to classify object, e.g. a car, you need to have a dataset that contains the image of object (car) and the label of the image ("car"), with thousands row of it. After that, the model should be able to classify an image of cars.


## Transformer Architecture

- Transformer Architecture consist of two modules, encoder and decoder. First, the encoder accept inputs and convert it to a numerical representation (embedding vectors). The vectors also capture the contextual information. 
- Decoder, take the encoded vector, decode it to generate the output text.
- Encoder and decoder consist of many layers.
- Encoder and decoder connected by something called self-attention mechanism.


### Zero-shot learning

- Zero-shot learning is the ability to complete/predict the next output without any examples or context given. Prompt example: `Translate to Indonesian language. Water = `, the output will be "air".
- Few-shot learning refers to the ability to complete the text with few specific examples. Prompt example: `monitro = monitor; glsas = glass; sheo = `, the output will be "shoe"

## Large datasets

- The datasets used in GPT-3 are all from internet, including CommonCrawl, WebText, Books, and Wikipedia.
- That is a huge shit of data. CommonCrawl (470 billion tokens) requires about 570GB of storage.
- GPT-3 costs $4.6 million of computing credits.

### Why LLM can perform task that aren't explicitly trained for that?

- It's called `emergent behavior`, the consequence of being exposed to huge amount of data in many context. They learn pattern, they predict the next word based on the pattern it found. 
- This will increase the probability for the model to hallucinate.
