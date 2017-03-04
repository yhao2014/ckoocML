import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by yhao on 2017/2/14.
  */
object WordCountTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Test").setMaster("local")
    val sc = new SparkContext(conf)

    val filePath = "E:/README.md"

    val count = sc.textFile(filePath, 2).flatMap(_.split("\\s+")).map(x => (x, 1)).reduceByKey(_ + _)
    val sorted = count.sortBy(_._2, ascending = false)
    sorted.take(10).foreach(println)

    sc.stop()
  }

}
