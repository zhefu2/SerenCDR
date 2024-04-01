# SerenCDR

This is our implementation for the SerenCDR model.

**Dataset:**  
1. _SerenCDRLens_ [[link]](https://github.com/zhefu2/SerenLens), a large ground truth dataset on cross-domain serendipity-oriented recommendation tasks.
2. Amazon Review Data (Books). [[link]](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

 ## Files

- Unexpectedness_generation.py: Calculating unexpectedness score for each user-item pair.
- SerenCDR.py: Our proposed model.

## Environment Settings
 We use Tensorflow as the backend.
 * Tensorflow version: '2.8.0'
 
## Quick Start

1. Pre-calculate the items' unexpectedness scores for each user and generate the unexpectedness training set
    ```
    python Unexpectedness_generation.py
    ```
    
2. Train the SerernCDR model
    ```
    python SerenCDR.py
    ```
