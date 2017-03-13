package preprocess

import com.hankcs.hanlp.seg.Segment
import com.hankcs.hanlp.seg.common.Term
import com.hankcs.hanlp.tokenizer.{IndexTokenizer, NLPTokenizer, SpeedTokenizer, StandardTokenizer}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Row}
import utils.segment.{MyCRFSegment, MyNShortSegment}

import scala.collection.JavaConversions._

/**
  * 基于HanLP的分词
  *
  * Created by yhao on 2017/2/17.
  */
class Segmenter(val uid: String) extends Serializable {
  //英文字符正则
  private val enExpr = "[A-Za-z]+"
  //数值正则，可以匹配203,2.23,2/12
  private val numExpr = "\\d+(\\.\\d+)?(\\/\\d+)?"
  //匹配英文字母、数字、中文汉字之外的字符
  private val baseExpr = """[^\w-\s+\u4e00-\u9fa5]"""

  private var inputCol: String = ""
  private var outputCol: String = ""
  private var segmentType: String = "StandardTokenizer"
  private var addNature: Boolean = false
  private var delNum: Boolean = false
  private var delEn: Boolean = false
  private var minTermLen: Int = 1
  private var minTermNum: Int = 3

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

  def addNature(value: Boolean): this.type = {
    this.addNature = value
    this
  }

  def isDelNum(value: Boolean): this.type = {
    this.delNum = value
    this
  }

  def isDelEn(value: Boolean): this.type = {
    this.delEn = value
    this
  }

  def setMinTermLen(value: Int): this.type = {
    require(value > 0, "最小词长度必须大于0")
    this.minTermLen = value
    this
  }

  def setMinTermNum(value: Int): this.type = {
    require(value > 0, "行最小词数必须大于0")
    this.minTermNum = value
    this
  }

  def getInputCol: String = this.inputCol

  def getOutputCol: String = this.outputCol

  def getSegmentType: String = this.segmentType

  def this() = this(Identifiable.randomUID("segment"))

  def transform(dataset: DataFrame): DataFrame = {
    val spark = dataset.sparkSession

    var segment: Segment = null
    segmentType match {
      case "NShortSegment" =>
        segment = new MyNShortSegment()
      case "CRFSegment" =>
        segment = new MyCRFSegment()
      case _ =>
    }

    val tokens = dataset.select(inputCol).rdd.flatMap{ case Row(line: String) =>
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


      val termSeq = terms.flatMap { term =>
        val word = term.word
        val nature = term.nature

        if (this.delNum && word.matches(numExpr)) None      //去除数字
        else if (this.delEn && word.matches(enExpr)) None   //去除英文
        else if (word.length < minTermLen) None            //去除过短的词
        else if (this.addNature) Some(word + "/" + nature)
        else Some(word)
      }

      if (termSeq.nonEmpty && termSeq.size >= minTermNum) Some((line, termSeq)) else None   //去除词数过少的行
    }

    import spark.implicits._
    val tokensSet = tokens.toDF(inputCol + "#1", outputCol)

    dataset.join(tokensSet, dataset(inputCol) === tokensSet(inputCol + "#1")).drop(inputCol + "#1")
  }
}
