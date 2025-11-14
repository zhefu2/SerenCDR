# SerenCDR

This is our implementation for the SerenCDR model in TORS 2025 paper:

**Fu, Z., Niu, X., Wu, X. and Rahman, R., 2025. A deep learning model for cross-domain serendipity recommendations. _ACM Transactions on Recommender Systems_, _3_(3), pp.1-21.** [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3690654)

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
    
2. Train the SerernCDR model (The training data can be found [here](https://drive.google.com/drive/folders/1EDxFkNL2itTvaWLkL0TLjzehOip0AGgT?usp=sharing))
    ```
    python SerenCDR.py
    ```
    
## Reference
Fu, Z., Niu, X., Wu, X. and Rahman, R., 2025. A deep learning model for cross-domain serendipity recommendations. _ACM Transactions on Recommender Systems_, _3_(3), pp.1-21.


```  
@article{fu2025deep,
  title={A deep learning model for cross-domain serendipity recommendations},
  author={Fu, Zhe and Niu, Xi and Wu, Xiangcheng and Rahman, Ruhani},
  journal={ACM Transactions on Recommender Systems},
  volume={3},
  number={3},
  pages={1--21},
  year={2025},
  publisher={ACM New York, NY}
}
```
