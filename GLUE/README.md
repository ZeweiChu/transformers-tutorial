# GLUE

本文件夹下的代码来自[https://github.com/huggingface/transformers/tree/master/examples/text-classification](https://github.com/huggingface/transformers/tree/master/examples/text-classification)

在使用代码训练之前，先下载GLUE数据集
```
python download_glue_data.py --data_dir /path/to/glue --tasks all
```

run.sh中的代码需要按照GLUE数据的路径做相应的修改
