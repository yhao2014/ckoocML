import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StopWordsRemover}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import preprocess.Segmenter

/**
  * Created by yhao on 2017/2/11.
  */
object LRClassificationTest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    //    HanLP.Config.enableDebug()

    val spark = SparkSession
      .builder
      .master("local[2]")
      .appName("Segment Test")
      .getOrCreate()

    val filePath = "data/classnews"
    val stopwordPath = "dictionaries/hanlp/data/dictionary/stopwords.txt"


    //数据清洗、转换
    val textDF = clean(filePath, spark)


    //分词
    val segmenter = new Segmenter(spark)
      .setSegmentType("StandardSegment")
      .enableNature(false)
      .setInputCol("content")
      .setOutputCol("tokens")
    val segDF = segmenter.transform(textDF)


    //去除停用词
    val stopwordArray = spark.sparkContext.textFile(stopwordPath).collect()
    val remover = new StopWordsRemover()
      .setStopWords(stopwordArray)
      .setInputCol("tokens")
      .setOutputCol("removed")
    val removedDF = remover.transform(segDF)


    //向量化
    val vectorizer = new CountVectorizer()
      .setVocabSize(15000)
      .setInputCol("removed")
      .setOutputCol("features")
    val parentVecModel = vectorizer.fit(removedDF)

    val numPattern = "[0-9]+".r
    val vocabulary = parentVecModel.vocabulary.flatMap{term =>
      if (term.length == 1 || term.matches(numPattern.regex)) None else Some(term)
    }

    val vecModel = new CountVectorizerModel(Identifiable.randomUID("cntVec"), vocabulary)
    .setInputCol("removed")
    .setOutputCol("features")
    val vectorDF = vecModel.transform(removedDF)

    val Array(train, predict) = vectorDF.randomSplit(Array(0.7, 0.3))


    //LR分类模型训练
    train.persist()
    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.2)
      .setElasticNetParam(0.05)
      .setLabelCol("label")
      .setFeaturesCol("features")
      .fit(train)
    train.unpersist()


    //LR预测
    val predictions = lr.transform(predict)
//    predictions.select("prediction", "label", "probability").show(100, truncate = false)

    //评估效果
    val predictionsRDD = predictions.select("prediction", "label")
      .rdd.
      map { case Row(prediction: Double, label: Double) => (prediction, label) }

    val metrics = new MulticlassMetrics(predictionsRDD)
    val accuracy = metrics.accuracy
    val weightedPrecision = metrics.weightedPrecision
    val weightedRecall = metrics.weightedRecall
    val f1 = metrics.weightedFMeasure

    println("\n\n========= 评估结果 ==========")
    println(s"分类正确率：$accuracy")
    println(s"\n加权准确率：$weightedPrecision")
    println(s"加权召回率：$weightedRecall")
    println(s"F1值：$f1")

    spark.stop()
  }


  def clean(filePath: String, spark: SparkSession): DataFrame = {
    import spark.implicits._
    val textDF = spark.sparkContext.textFile(filePath).flatMap { line =>
      val fields = line.split("\u00EF")
      if (fields.length > 3) {
        val categoryLine = fields(0)
        val categories = categoryLine.split("\\|")
        val category = categories.last

        var label = -1.0
        if (category.contains("文化")) label = 0.0
        else if (category.contains("财经")) label = 1.0
        else if (category.contains("军事")) label = 2.0
        else if (category.contains("体育")) label = 3.0
        else {}

        val title = fields(1)
        val time = fields(2)
        val content = fields(3)
        if (label > -1) Some(label, title, time, content) else None
      } else None
    }.toDF("label", "title", "time", "content")

    textDF
  }
}
