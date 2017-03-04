import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

/**
  * 广播变量的使用，主要用于大型文件（如词典，模型等）需要反复使用的情况。
  * 广播变量有助于减少网络传输次数，提高计算速度
  *
  * Created by yhao on 2017/2/13.
  */
object BroadcastTest {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)

    val argLength = args.length
    val blockSize = if (argLength > 2) args(2) else "4096"

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Broadcast Test")
      .config("spark.broadcast.blocksize", blockSize)
      .getOrCreate()

    val sc = spark.sparkContext

    val slices = if (argLength > 0) args(0).toInt else 2
    val num = if (argLength > 1) args(1).toInt else 100000000

    val arr1 = (0 until num).toArray

    for (i <- 0 until 3) {
      println("Iteration " + i)
      println("==============")
      val startTime = System.nanoTime()
      val barr1 = sc.broadcast(arr1)
      val observedSize = sc.parallelize(1 to 10, slices).map(_ => barr1.value.length)
      observedSize.collect().foreach(println)
      println("Iteration %d took %.0f milliseconds".format(i, (System.nanoTime() - startTime) / 1e6))
    }

    spark.stop()
  }
}
