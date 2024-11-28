# API-Net
## Pipeline

![framework](/figs/1.jpg)


## Installation
1.
    ```
    https://github.com/songshuaitian/API-Net
    ```

2.
    ```
    conda install -c pytorch pytorch torchvision
    ```

3.
    ```
    pip install -r requirements.txt
    ```

## Prepare
[here.](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2)
(
)
 **RIDCP500**[RIDCP's Repo](https://github.com/RQ-Wu/RIDCP_dehazing)
The final file path should be the same as the following (please check it carefully):

```
┬─ save_models
│   ├─ rtts
│   │   ├─ API-Net.pth
│   │   └─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ rgb500
    │   └─ ... (image filename)
    ├─ depth500
    │   └─ ... (image filename)
    ├─ Seggray
    │   └─ ... (image filename)
    ├─ RTTS
    │   └─ hazy
    │   │    └─ ... (image filename)
    │   └─ Seggray
    │       └─ ... (image filename)
    └─ ... (dataset name)
```

## Training

`configs`

After adjusting the settings, use the following script to initiate the training of the model:

```
CUDA_VISIBLE_DEVICES=X python train.py
```

For example：

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Evaluation

Run the following script to evaluate the trained model with a single GPU.


```sh
CUDA_VISIBLE_DEVICES=X python test.py
```

For example：

```sh
CUDA_VISIBLE_DEVICES=0 python test.py
```


# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.

    Zhou Shen
    Faculty of Information Engineering and Automation
    Kunming University of Science and Technology                                                           
    Email: songshuaitiann@163.com
