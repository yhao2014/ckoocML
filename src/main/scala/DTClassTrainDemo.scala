import classification.DTClassifier
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import preprocess.Preprocessor

/**
  * 基于决策树的多分类模型训练
  *
  * Created by yhao on 2017/3/9.
  */
object DTClassTrainDemo extends Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    //    HanLP.Config.enableDebug()

    val spark = SparkSession
      .builder
      .master("local[2]")
      .appName("DT_Train_Demo")
      .getOrCreate()

    val filePath = "data/classnews/train"

    //=== 预处理(清洗、标签索引化、分词、向量化)
    val preprocessor = new Preprocessor
    val trainDF = preprocessor.predict(filePath, spark)._1

    //=== 模型训练
    val dtClassifier = new DTClassifier
    dtClassifier.train(trainDF)

    spark.stop()
  }
}
