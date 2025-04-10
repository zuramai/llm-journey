## What is mean=0 and variance=1 and why should we normalize the data?

Mean=0 and Variance=1 means we modify the array of numbers to be around 0. 

Without normalizing the data, as the data flows through neural network, the numbers can grow exponentially large (exploding gradients) or shrink towards zero.

With normalization, the values stay in a controlled range. The network learns to weigh the actual importance.