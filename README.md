# SerenCDR

This is our implementation for the SerenCDR model.

**Dataset:**  
1. _SerenCDRLens_ [[link]](https://github.com/zhefu2/SerenLens), a large ground truth dataset on cross-domain serendipity-oriented recommendation tasks.
2. Amazon Review Data. [[link]](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

 ## Files

- Expectedness_generation.py: Calculating Expectedness score for each user-item pair.
- SerenCDR.py: Our proposed model.

## Environment Settings
 We use Tensorflow as the backend.
 * Tensorflow version: '2.8.0'
 
## Quick Start

1. Pre-calculate the items' expectedness scores for each user and generate the training samples
    ```
    python Expectedness_generation.py
    ```
    
2. Train the SerernCDR model
    ```
    python SerenCDR.py
    ```
