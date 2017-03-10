import classification.{DTClassifier, LRClassifier}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Row, SparkSession}
import preprocess.Preprocessor
import utils.Evaluations

/**
  * 多分类模型评估
  *
  * Created by yhao on 2017/3/9.
  */
object MultiClassEvaluationDemo {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    //    HanLP.Config.enableDebug()

    val spark = SparkSession
      .builder
      .master("local[2]")
      .appName("Multi_Class_Evaluation_Demo")
      .getOrCreate()

    val filePath = "data/classnews/predict"

    //=== 预处理(清洗、分词、向量化)
    val preprocessor = new Preprocessor
    val (predictDF, indexModel, _) = preprocessor.predict(filePath, spark)

    //=== DT模型预测
    val dtClassifier = new DTClassifier
    val dtPredictions = dtClassifier.predict(predictDF, indexModel)

    //=== LR模型预测
    val lrClassifier = new LRClassifier
    val lrPredictions = lrClassifier.predict(predictDF, indexModel)

    val predictions = Seq(dtPredictions, lrPredictions)

    for (prediction <- predictions) {
      //=== 模型评估
      val resultRDD = prediction.select("prediction", "indexedLabel").rdd.map { case Row(prediction: Double, label: Double) => (prediction, label) }
      val (precision, recall, f1) = Evaluations.multiClassEvaluate(resultRDD)
      println(s"\n\n========= 评估结果 ==========")
      println(s"\n加权准确率：$precision")
      println(s"加权召回率：$recall")
      println(s"F1值：$f1")
    }

    spark.stop()
  }
}
