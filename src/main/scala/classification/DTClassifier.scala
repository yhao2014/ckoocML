package classification

import java.io.File

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexerModel}
import org.apache.spark.sql.DataFrame
import params.ClassParam
import utils.IOUtils

/**
  * 决策树多分类
  *
  * Created by yhao on 2017/3/9.
  */
class DTClassifier extends Serializable {
  /**
    * 决策树模型训练处理过程, 包括: "模型训练"
    *
    * @param data   训练集
    * @return (向量模型, 决策树模型)
    */
  def train(data: DataFrame): DecisionTreeClassificationModel = {
    val params = new ClassParam

    //=== LR分类模型训练
    data.persist()
    val dtModel = new DecisionTreeClassifier()
      .setMinInfoGain(params.minInfoGain)
      .setMaxDepth(params.maxDepth)
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .fit(data)
    data.unpersist()
    this.saveModel(dtModel, params)

    dtModel
  }


  /**
    * 决策树预测过程, 包括"决策树预测", "模型评估"
    *
    * @param data     测试集
    * @param indexModel 索引模型
    * @return 预测DataFrame, 增加字段:"rawPrediction", "probability", "prediction", "predictedLabel"
    */
  def predict(data: DataFrame, indexModel: StringIndexerModel): DataFrame = {
    val params = new ClassParam
    val dtModel = this.loadModel(params)

    //=== DT预测
    val predictions = dtModel.transform(data)

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
    * @param dtModel  决策树模型
    * @param params 配置参数
    */
  def saveModel(dtModel: DecisionTreeClassificationModel, params: ClassParam): Unit = {
    val filePath = params.modelDTPath
    val file = new File(filePath)
    if (file.exists()) {
      println("决策树模型已存在，新模型将覆盖原有模型...")
      IOUtils.delDir(file)
    }

    dtModel.save(filePath)
    println("决策树模型已保存！")
  }


  /**
    * 加载模型
    *
    * @param params 配置参数
    * @return 决策树模型
    */
  def loadModel(params: ClassParam): DecisionTreeClassificationModel = {
    val filePath = params.modelDTPath
    val file = new File(filePath)
    if (!file.exists()) {
      println("决策树模型不存在，即将退出！")
      System.exit(1)
    } else {
      println("开始加载决策树模型...")
    }

    val dtModel = DecisionTreeClassificationModel.load(filePath)
    println("决策树模型加载成功！")

    dtModel
  }
}
