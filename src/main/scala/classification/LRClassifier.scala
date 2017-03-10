package classification

import java.io.File

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{IndexToString, StringIndexerModel}
import org.apache.spark.sql.DataFrame
import params.ClassParam
import utils.IOUtils

/**
  * 逻辑回归多分类
  *
  * Created by yhao on 2017/3/7.
  */
class LRClassifier extends Serializable {

  /**
    * LR模型训练处理过程, 包括: "模型训练"
    *
    * @param data   训练集
    * @return (向量模型, LR模型)
    */
  def train(data: DataFrame): LogisticRegressionModel = {
    val params = new ClassParam

    //=== LR分类模型训练
    data.persist()
    val lrModel = new LogisticRegression()
      .setMaxIter(params.maxIteration)
      .setRegParam(params.regParam)
      .setElasticNetParam(params.elasticNetParam)
      .setTol(params.converTol)
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .fit(data)
    data.unpersist()
    this.saveModel(lrModel, params)

    lrModel
  }


  /**
    * LR预测过程, 包括"LR预测", "模型评估"
    *
    * @param data     测试集
    * @return 预测DataFrame, 增加字段:"rawPrediction", "probability", "prediction", "predictedLabel"
    */
  def predict(data: DataFrame, indexModel: StringIndexerModel): DataFrame = {
    val params = new ClassParam
    val lrModel = this.loadModel(params)

    //=== LR预测
    val predictions = lrModel.transform(data)

    //=== 索引转换为label
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(indexModel.labels)
    val result = labelConverter.transform(predictions)

    result
  }


  /**
    * 保存模型
    *
    * @param lrModel  LR模型
    * @param params 配置参数
    */
  def saveModel(lrModel: LogisticRegressionModel, params: ClassParam): Unit = {
    val filePath = params.modelLRPath
    val file = new File(filePath)
    if (file.exists()) {
      println("LR模型已存在，新模型将覆盖原有模型...")
      IOUtils.delDir(file)
    }

    lrModel.save(filePath)
    println("LR模型已保存！")
  }


  /**
    * 加载模型
    *
    * @param params 配置参数
    * @return LR模型
    */
  def loadModel(params: ClassParam): LogisticRegressionModel = {
    val filePath = params.modelLRPath
    val file = new File(filePath)
    if (!file.exists()) {
      println("LR模型不存在，即将退出！")
      System.exit(1)
    } else {
      println("开始加载LR模型...")
    }

    val lrModel = LogisticRegressionModel.load(filePath)
    println("LR模型加载成功！")

    lrModel
  }
}
