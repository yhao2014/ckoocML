package preprocess

import com.hankcs.hanlp.seg.Segment
import com.hankcs.hanlp.seg.common.Term
import com.hankcs.hanlp.tokenizer.{IndexTokenizer, NLPTokenizer, SpeedTokenizer, StandardTokenizer}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import preprocess.utils.segment.{MyCRFSegment, MyNShortSegment}

import scala.collection.JavaConversions._

/**
  * Created by yhao on 2017/2/17.
  */
class Segmenter(spark: SparkSession, val uid: String) extends Serializable {

  private var inputCol = ""
  private var outputCol = ""
  private var segmentType = "StandardTokenizer"
  private var enableNature = false

  def setInputCol(value: String): this.type = {
    this.inputCol = value
    this
  }

  def setOutputCol(value: String): this.type = {
    this.outputCol = value
    this
  }

  def setSegmentType(value: String): this.type = {
    this.segmentType = value
    this
  }

  def enableNature(value: Boolean): this.type = {
    this.enableNature = value
    this
  }

  def this(spark: SparkSession) = this(spark, Identifiable.randomUID("segment"))

  def transform(dataset: DataFrame): DataFrame = {
    var segment: Segment = null
    segmentType match {
      case "NShortSegment" =>
        segment = new MyNShortSegment()
      case "CRFSegment" =>
        segment = new MyCRFSegment()
      case _ =>
    }

    val tokens = dataset.select(inputCol).rdd.map { case Row(line: String) =>
      var terms: Seq[Term] = Seq()
      segmentType match {
        case "StandardSegment" =>
          terms = StandardTokenizer.segment(line)
        case "NLPSegment" =>
          terms = NLPTokenizer.segment(line)
        case "IndexSegment" =>
          terms = IndexTokenizer.segment(line)
        case "SpeedSegment" =>
          terms = SpeedTokenizer.segment(line)
        case "NShortSegment" =>
          terms = segment.seg(line)
        case "CRFSegment" =>
          terms = segment.seg(line)
        case _ =>
          println("分词类型错误！")
          System.exit(1)
      }

      val termSeq = terms.map(term =>
        if (this.enableNature) term.toString else term.word
      )

      (line, termSeq)
    }

    import spark.implicits._
    val tokensSet = tokens.toDF(inputCol + "#1", outputCol)

    dataset.join(tokensSet, dataset(inputCol) === tokensSet(inputCol + "#1")).drop(inputCol + "#1")
  }
}
