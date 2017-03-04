import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession

/**
  * Created by yhao on 2017/2/17.
  */
object StopWordTest {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("StopWord Remove Test")
      .getOrCreate()

    val dataset = spark.createDataFrame(Seq(
      (0, Seq("I", "saw", "the", "red", "balloon")),
      (1, Seq("Mary", "had", "a", "little", "lamb"))
    )).toDF("id", "tokens")

    val startTime = System.nanoTime()
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")
    val removed = remover.transform(dataset)
    val removeTime = (System.nanoTime() - startTime) / 1e6

    removed.show(false)
    println(s"去除停用词耗时：$removeTime 毫秒！")

    spark.stop()
  }
}
