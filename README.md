***********************************************************************
# ckoocML --ickooc机器学习


***********************************************************************
本项目主要是对一些常用的算法的实现，以及基于spark的机器学习算法实现。<br>

目前以实现的功能有：
>
* 数据预处理
* 基于spark的LR逻辑回归分类
* 基于spark的DT决策树分类


## 数据预处理
[数据预处理](https://github.com/yhao2014/ckoocML/blob/master/src/main/scala/preprocess/Preprocessor.scala)主要对算法需要用到的数据进行前期的清洗等操作，其中分词等使用到了[HanLP](https://github.com/hankcs/HanLP)相关的代码。<br>

参考[示例代码](https://github.com/yhao2014/ckoocML/blob/master/src/main/scala/PreprocessDemo.scala)

**由于分词使用到的词典和模型较大，因此未上传到github上，大家可以从[HanLP主页](https://github.com/hankcs/HanLP/releases)下载对应版本的模型数据(具体版本见ml模块的pom.xml配置)**<br>
*下载完后将data目录解压到ckooc-ml/dictionaries/hanlp目录下即可*

目前已实现的功能：
>
* 分词
* 去除停用词
* 去除英文
* 去除数字
* 词性标注
* 向量化

### 输入数据格式
数据预处理的输入数据为[中国新闻网](http://www.chinanews.com/)上抓取的数据,分为6个类别`体育`,`军事`,`文化`和`经济`. 分为训练文本和测试文本.

输入文件位置：
* 训练文本: data/classnews/train/
* 测试文本: data/classnews/test/

### 输出数据格式
输出经过分词等预处理之后的数据、特征词索引模型、向量模型<br>

模型保存位置：
* 特征词索引模型: models/preprocession/indexModel
* 向量模型: models/preprocession/vecModel

## 分类
目前主要使用分类算法进行新闻分类，已实现的算法有：
>
* [LR逻辑回归](https://github.com/yhao2014/ckoocML/blob/master/src/main/scala/classification/LRClassifier.scala)
* [DT决策树](https://github.com/yhao2014/ckoocML/blob/master/src/main/scala/classification/DTClassifier.scala)

**上述所有功能的Demo均在[src/main/scala](https://github.com/yhao2014/ckoocML/tree/master/src/main/scala)目录下，包含模型训练和测试两部分(注意，分类需要先进行预处理，因此确保事先运行预处理代码，产生了两个预处理模型)**

---
# 各种测试记录
## 分类记录
### LR分类测试记录
       数据类型        数据大小        分类数        训练文本数/per分类        测试文本数/per分类        分类算法        特征维数         准确度
    新闻分类数据         457M            13               10000                     5000                LR             7000     0.817751203603044
    新闻分类数据         457M            13               10000                     5000                LR             8000     0.8633949371020345
    新闻分类数据         457M            13               10000                     5000                LR            10000     0.8678832116788321
    新闻分类数据         516M            15               10000                     5000                LR             8000     0.7378484249241117
    新闻分类数据         571M            16               10000                     5000                LR             8000     0.8016991074309066
    新闻分类(chinaNews)  683M            2              140000+                   60000+                LR            15000     0.9565125193703613
    新闻分类(chinaNews) 1.98G            6              160000+                   70000+                LR            50000     0.8599223781293982

### SVM分类(二分类)测试记录
       数据类型        数据大小        分类数        训练文本数/per分类        测试文本数/per分类        分类算法        特征维数         Area under ROC
    新闻分类(chinaNews)  683M            2              140000+                   60000+               SVM            15000        0.9649445556688644
    新闻分类(chinaNews)  587M            2              140000+                   60000+               SVM            15000        0.9421810171886563


**说明**
>
* 2分类：国内新闻、国外新闻(683M)/文化、娱乐(587M)
* 6分类：经济、军事、社会、体育、文化、娱乐
* 13分类：公益、健康、交通、教育、经济、军事、历史、农业、时尚、数码、体育、通讯、娱乐
