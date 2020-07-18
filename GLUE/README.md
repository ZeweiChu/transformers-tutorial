# GLUE

本文件夹下的代码来自[https://github.com/huggingface/transformers/tree/master/examples/text-classification](https://github.com/huggingface/transformers/tree/master/examples/text-classification)

在使用代码训练之前，先下载GLUE数据集
```
python download_glue_data.py --data_dir /path/to/glue --tasks all
```

如果你无法从下载GLUE数据集，可以从以下百度云链接下载
链接：https://pan.baidu.com/s/1XG1_RdHCTcPZwyLC2KoCqw 
提取码：uhly

run.sh中的代码需要按照GLUE数据的路径做相应的修改
