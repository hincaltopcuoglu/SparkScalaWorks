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

object trivago extends App {
  


val conf = new SparkConf().setMaster("local[*]").setAppName("trivago")
val sc = new SparkContext(conf)
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  
import sqlContext.implicits._
  
val csv = sc.textFile("C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/case_study_reco_ratings_train.csv")

val csv_d = csv.map(p => (p(3).toInt,(p(0).toInt,p(1).toInt,p(2).toInt)))
  
val data = csv.map(l => l.split(";")).map(l => (l(0).toInt,l(1).toInt,l(2).toInt,l(3).toInt))




//val df = sqlContext.read
  //  .format("com.databricks.spark.csv")
	//  .option("delimiter", ";")
  //  .option("header", "true")
  //  .load("C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/case_study_reco_ratings_train.csv")
    
    

    
    
def createRowRDD(rdd:RDD[String],anArray:org.apache.spark.broadcast.Broadcast[Array[Int]]) : org.apache.spark.rdd.RDD[org.apache.spark.sql.Row]  = {
    val rowRDD = rdd.map(_.split(";")).map(_.map({y => try {y.toDouble} catch {case _ : Throwable => 0.0}})).map(p => Row.fromSeq(anArray.value map p))
    return rowRDD
}
  
val arrVar = sc.broadcast(0 to 3 toArray)

val dx = createRowRDD(csv,arrVar)

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


//val splits = trainLP.randomSplit(Array(0.75, 0.25), seed = 2L)
//val training = splits(0).cache()
//val test = splits(1)

 //val ignored = List("p")

 //val featInd = df.columns.diff(ignored).map(df.columns.indexOf(_))

 //val targetInd = df.columns.indexOf("p") 

//val deneme = df.rdd.map(r => LabeledPoint(
   //r.getDouble(targetInd), // Get target value
   // Map feature indices to values
  // Vectors.dense(featInd.map(r.getDouble(_)).toArray) 
//))


val model2 = NaiveBayes.train(trainLP, lambda = 0.01, modelType = "multinomial")

val labels = model2.labels

val features = trainLP.map(lp => lp.features)

import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)
  
  val lrModel = lr.fit(output2)
  
  val predictions = lrModel.transform(output_target2)
  
  predictions.agg(min($"probability"), max($"probability"))
  
  import org.apache.spark.sql.functions._
  
  predictions.agg(max(predictions(predictions.columns(6))), min(predictions(predictions.columns(6)))).show()

//
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.sql.functions.{concat, lit}

import sqlContext.implicits._

import org.apache.spark.ml.feature.StringIndexer

val assembler = new VectorAssembler()
  .setInputCols(Array("user_id", "item_id", "rating"))
  .setOutputCol("features")
  
  
  val output = assembler.transform(trainDF)

  val columnsRenamed = Seq("label") 
  
  val output2 = output.withColumnRenamed("p","label")

 val model3 = new NaiveBayes().fit(output2)
 
 val assembler2 = new VectorAssembler()
  .setInputCols(Array("user_id", "item_id", "rating"))
  .setOutputCol("features")
 
 
  val output_target = assembler2.transform(targetxDF)
  
   val output_target2 = output_target.withColumnRenamed("p","label")
 
 val predictions = model3.transform(output_target2)
 
 predictions.show()
 
 
 val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
  
val accuracy = evaluator.evaluate(predictions)
println("Test set accuracy = " + accuracy)
 
//


import org.apache.spark.sql.functions._
  

  
  // for target data set
 val csv_target = sc.textFile("C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/case_study_reco_ratings_target3.csv")
  
 val header = csv_target.first() 
 val data = csv_target.filter(row => row != header)
  
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

//val parsedData = data.map(s => Vectors.dense(s.split(';').map(_.toDouble)))

val predictionAndLabel = targetLP.map(p => (model2.predict(p.features), p.label))

val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / targetLP.count()

  val labelAndPreds = targetLP.map { point =>
  val prediction = model2.predictProbabilities(point.features)
  (prediction)
}
  
  val d1x = labelAndPreds.map {x =>
    val deger1=(x._1)    
    val deger2=deger1(0)-(deger1(0)%0.01)
    val deger3=(x._2)
    (deger2,deger3)
  }
  
  
  val xx1 = targetLP.map(p => (p.features,p.label))
  
  
 
  
  
  
  //////////////////////////////////////////////////////////////////////////////
  import org.apache.spark.mllib.recommendation.ALS
  import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating




import sqlContext.implicits._
  
val csvx = sc.textFile("C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/case_study_reco_ratings_train2.csv")

val headerx = csvx.first() 
 val datax = csvx.filter(row => row != headerx)

val ratings1 = datax.map(l => l.split(";")).map(l => (l(0).toInt,l(1).toInt,l(2).toDouble))

val ratings = datax.map(_.split(';') match { case Array(user_id, item_id, rating) =>
  Rating(user_id.toInt, item_id.toInt, rating.toDouble)
})

val rank = 10
val numIterations = 10
val model = ALS.train(ratings, rank, numIterations, 0.01)


val usersProducts = ratings.map { case Rating(user, product, rate) =>
  (user, product)
}
  
  val predictions =
  model.predict(usersProducts).map { case Rating(user, product, rate) =>
    ((user, product), rate)
  }
  
  val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
  ((user, product), rate)
}.join(predictions)

val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()
println("Mean Squared Error = " + MSE)


/////////////////////////////////////////////////////////

case class Rating(user_id: Int, item_id: Int, rating: Double)
def parseRating(str: String): Rating = {
  val fields = str.split(";")
  assert(fields.size == 3)
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
}

val xx = sc.textFile("C:/Users/htopcuoglu/Desktop/trivago_dataset_reco/case_study_reco_ratings_train2.csv")

val headerx = xx.first() 
 val datax = xx.filter(row => row != headerx)
 
  val ratings=datax.map(parseRating).toDF()
  
  
  import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
  
  val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("user_id")
  .setItemCol("item_id")
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


































