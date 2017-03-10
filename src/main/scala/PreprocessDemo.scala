import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import preprocess.Preprocessor

/**
  * 预处理，包括: "数据清洗", "标签索引化", "分词", "向量化"
  *
  * Created by yhao on 2017/3/8.
  */
object PreprocessDemo extends Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    //    HanLP.Config.enableDebug()

    val spark = SparkSession
      .builder
      .master("local[2]")
      .appName("Preprocess Demo")
      .getOrCreate()

    val filePath = "data/classnews/train"

    val preprocessor = new Preprocessor
    preprocessor.train(filePath, spark)

    spark.stop()
  }
}
