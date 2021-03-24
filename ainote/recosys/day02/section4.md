# 3.4 GCP模型预测与部署

## 学习目标

- 目标
  - 了解GCP平台在线预测的使用教程
- 应用
  - 应用完成泛娱乐模型的预测和部署

### 3.4.1 GCP在线预测介绍

在您训练好模型后，Cloud ML Engine 会提供两种类型的预测，以将计算机学到的知识运用到新的样本中。

**在线预测**功能以无服务器的完全托管方式部署机器学习模型，提供实时响应并实现高可用性。我们的全球预测平台可自动扩容以适应任何吞吐量。它提供安全的 Web 端点，方便您将机器学习功能集成到自己的应用中

**批量预测**功能可为异步应用提供无与伦比的吞吐量，从而进行经济高效的推理。它能够轻松扩容，以对 TB 级的生产数据进行推理

### 3.4.2 预测服务实现

* 在云端部署模型：
  * 为模型起名：MODEL_NAME=wdl
  * 在指定云服务器区域创建模型空间： gcloud ml-engine models create REGION
  * 找出已训练模型的完整二进制存储路径(精确至时间戳文件夹)
    * MODEL_BINARIES=gs://$BUCKET_NAME/wdl/1487877383942/
  * 创建模型并指定版本v1：

```python
 # 导出版本
 gcloud ml-engine versions create v1 \
     --model $MODEL_NAME \
     --origin $MODEL_BINARIES \
     --runtime-version 1.10

 # 进行预测
 gcloud ml-engine jobs submit prediction $JOB_NAME \
     --model $MODEL_NAME \
     --version v1 \
     --data-format text \
     --region $REGION \
     --output-path $OUTPUT_PATH/predictions
```

* 批量预测：
  * 设置作业名字：JOB_NAME=wdl_prediction_1
  * 设置预测文件路径： TEST_JSON=./test.json
    * 设置生成文件路径: OUTPUT_PATH=gs://JOB_NAME
  

### 3.4.3 本地训练预测部署

如果没有相关的资源可以在GCP中使用，也提供了本地的方式，只不过没有参数调用，分布式训练的条件

* 目的：本地训练完模型，进行模型导出部署预测
  * 1、提交任务，进行本地训练
  * 2、docker进行部署和预测

1、提交任务，进行本地训练

```shell
TRAIN_DATA=$(pwd)/train_data.csv
EVAL_DATA=$(pwd)/test_data.csv
MODEL_DIR=./output


gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100 \
```

2、docker进行部署和预测

```python
docker run -t --rm -p 8501:8501 -v "/root/recreation_project/recomm/model/output/nss:/models/wdl" -e MODEL_NAME=wdl tensorflow/serving &

curl -d '{"instances": [2,0,1,0,8,3,0,2,1515431417,2,4,6,0,0,2,4,5,0,0,0]}' -X POST http://localhost:8501/v1/models/wdl:predict
```

### 3.4.4 小结

* 远程进行预测了解
* 本地训练和部署