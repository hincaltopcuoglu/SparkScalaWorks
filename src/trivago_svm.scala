import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import scala.math.sqrt
import org.apache.spark.rdd.RDD.rddToOrderedRDDFunctions
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.types.{StructType,StructField,StringType,IntegerType};
import org.apache.spark.sql.Row;

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.sql._


import org.apache.spark.rdd.RDD

object trivago_svm extends App {
  
val conf = new SparkConf().setMaster("local[*]").setAppName("trivago_svm")
val sc = new SparkContext(conf)
val sqlContext = new org.apache.spark.sql.SQLContext(sc)

import sqlContext.implicits._
  
val data = sc.textFile("C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/case_study_reco_ratings_train.csv")

def createRowRDD(rdd:RDD[String],anArray:org.apache.spark.broadcast.Broadcast[Array[Int]]) : org.apache.spark.rdd.RDD[org.apache.spark.sql.Row]  = {
    val rowRDD = rdd.map(_.split(";")).map(_.map({y => try {y.toDouble} catch {case _ : Throwable => 0.0}})).map(p => Row.fromSeq(anArray.value map p))
    return rowRDD
}
  
val arrVar = sc.broadcast(0 to 3 toArray)

val dx = createRowRDD(data,arrVar)

val dictFile = "C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/header.txt"

var arrName = new Array[String](4)
for (line <- Source.fromFile(dictFile).getLines) {
    arrName = line.split(';').map(_.trim).toArray
}

var schemaString = arrName.mkString(";")
// Import Row
import org.apache.spark.sql.Row
// Import RDD
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.types.DoubleType

val schema = StructType(schemaString.split("\t").map(fieldName => StructField(fieldName, DoubleType, true)))

val trainDF = sqlContext.createDataFrame(dx,schema)

trainDF.printSchema

def toLabeledPoint(dataDF:org.apache.spark.sql.DataFrame) : org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint] = {
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.regression.LabeledPoint
    val targetInd = dataDF.columns.indexOf("p")
    val ignored = List("p")
    val featInd = dataDF.columns.diff(ignored).map(dataDF.columns.indexOf(_))
    val dataLP = dataDF.rdd.map(r => LabeledPoint(r.getDouble(targetInd),
     Vectors.dense(featInd.map(r.getDouble(_)).toArray)))
    return dataLP
}

val trainLP = toLabeledPoint(trainDF)

val numIterations = 100
val model = SVMWithSGD.train(trainLP, numIterations)


import org.apache.spark.sql.functions._
  

  
  // for target data set
 val csv_target = sc.textFile("C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/case_study_reco_ratings_target3.csv")
  
 //val header = csv_target.first() 
 //val datax = csv_target.filter(row => row != header)
  
 val arrVar_target = sc.broadcast(0 to 3 toArray)

 val dx_target = createRowRDD(csv_target,arrVar_target)

 val dictFile_target = "C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/header_target.txt"

 var arrName_target = new Array[String](4)
 for (line <- Source.fromFile(dictFile_target).getLines) {
    arrName_target = line.split(';').map(_.trim).toArray
 }

var schemaString_target = arrName_target.mkString(";")

val schema_target = StructType(schemaString_target.split("\t").map(fieldName => StructField(fieldName, DoubleType, true)))

val targetxDF = sqlContext.createDataFrame(dx_target,schema_target)

targetxDF.printSchema

val targetLP = toLabeledPoint(targetxDF)

val scoreAndLabels = targetLP.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()

  
}