package params

import utils.Conf

import scala.collection.mutable

/**
  * 预处理过程使用参数
  *
  * Created by yhao on 2017/3/8.
  */
class PreprocessParam extends Serializable  {
  val kvMap: mutable.LinkedHashMap[String, String] = Conf.loadConf("src/main/resources/preprocess.properties")

  val stopwordFilePath: String = kvMap.getOrElse("stopword.file.path", "dictionaries/hanlp/data/dictionary/stopwords.txt")    //停用词表路径
  val segmentType: String = kvMap.getOrElse("segment.type", "StandardSegment")    //分词方式
  val vocabSize: Int = kvMap.getOrElse("vocab.size", "10000").toInt   //特征词汇表大小

  val indexModelPath: String = kvMap.getOrElse("model.index.path", "models/preprocession/indexModel")    //索引模型保存路径
  val vecModelPath: String = kvMap.getOrElse("model.vectorize.path", "models/preprocession/vecModel")   //向量模型保存路径
}
