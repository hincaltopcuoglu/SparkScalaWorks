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

object trivago_collaborative extends App {
  
val conf = new SparkConf().setMaster("local[*]").setAppName("trivago")
val sc = new SparkContext(conf)
val sqlContext = new org.apache.spark.sql.SQLContext(sc)

import sqlContext.implicits._
import org.apache.spark.sql.Row
// Import RDD
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.types.DoubleType
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating


case class Rating(user_id: Int, item_id: Int, rating: Double)
def parseRating(str: String): Rating = {
  val fields = str.split(";")
  assert(fields.size == 3)
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
}

val xx = sc.textFile("C:/Users/Administrator/Desktop/trivago_dataset_reco/case_study_reco_ratings_train.csv")

val headerx = xx.first() 
 val datax = xx.filter(row => row != headerx)
 
  val ratings=datax.map(parseRating).toDF()
  
  
import org.apache.spark.ml.evaluation.RegressionEvaluator


  
 val als = new ALS()  
  .setMaxIter(5)
   .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
  
val modelx = als.fit(ratings)

val yy = sc.textFile("C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/case_study_reco_ratings_target4.csv")

val headery = yy.first() 
 val datay = yy.filter(row => row != headery)

 val ratings_target=datay.map(parseRating).toDF()
 
 
 modelx.setColdStartStrategy("drop")
 
val predictionsx = modelx.transform(ratings_target)

val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)

println(s"Root-mean-square error = $rmse")





  
}