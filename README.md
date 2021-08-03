### 文件
```
README.md
api_count_preprocess.py #全连接神经网络(DNN)数据预处理
api_count_train.py      #全连接神经网络训练
cnn.py                  #cnn数据预处理+训练
cp.py                   #提取爬取的exe文件
crawler.py              #GitHub爬虫
crawler_multiprocess.py #GitHub爬虫(多线程)
```
### 数据集

|  | Path | Count |
|---|---|---|
| Black1 | /home/ubuntu/data/sandbox/report | 20767 |
| White1 | /home/ubuntu/data/sandbox/80+ | 4057 |
| White2 | /home/ubuntu/data/sandbox/crawl_white | 11732 |


### 测试结果
## DNN

数据集:Black1+White1  
样本分割:80%训练,20%测试  

|      | **negative** | **positive** |  ||
|---|---|---|---|---|
| **white** | 757 | 30 | 3.96%  | 假阳率 |
| **black** | 18 | 3887 | 99.50%  | 检出率|


## CNN

数据集:Black1+White1  
样本分割:80%训练,20%测试  

|      | **negative** | **positive** |  ||
|---|---|---|---|---|
| **white** | 777 | 12 | 1.52%  | 假阳率 |
| **black** | 31 | 3872 | 99.20%  | 检出率|

数据集:Black1+White1+White2  
样本分割:80%训练,20%测试  

|      | **negative** | **positive** |  ||
|---|---|---|---|---|
| **white** | 3032 | 36 | 1.20%  | 假阳率 |
| **black** | 31 | 4127 | 99.30%  | 检出率|
