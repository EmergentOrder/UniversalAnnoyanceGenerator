package com.goldstar

import org.apache.http.util.EntityUtils
import org.apache.http.entity.ContentType
import org.apache.http.entity.StringEntity
import org.apache.http.client.config.RequestConfig
import org.apache.http.auth.{ AuthScope, UsernamePasswordCredentials }
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.SparseVector
import org.elasticsearch.client.RestClient
import org.apache.http.HttpHost
import grizzled.slf4j.Logger
import org.apache.http.impl.nio.client.HttpAsyncClientBuilder
import org.elasticsearch.client.RestClientBuilder
import org.apache.http.impl.client.BasicCredentialsProvider
import org.elasticsearch.client.RestClientBuilder.HttpClientConfigCallback
import org.apache.http.impl.nio.client.HttpAsyncClientBuilder
import org.json4s.JArray
import org.json4s.JInt
import org.json4s.JValue
import org.json4s.JObject
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import annoy4s._
import com.typesafe.config.ConfigFactory
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption._
import com.actionml.EsClient

import scala.reflect.io.Directory
import scala.collection.JavaConverters._

object UniversalAnnoyanceGenerator extends App {

val conf = ConfigFactory.parseResources("application.conf")

val numAnnoyTrees = conf.getInt("UniversalAnnoyanceGenerator.numAnnoyTrees")

implicit val formats = DefaultFormats

case class EventWithContent(id: Int, features: Vector[Double])

val logger = Logger("UniversalAnnoyanceGenerator")
val esIndex: String = conf.getString("UniversalAnnoyanceGenerator.esIndex")
val numItems =  conf.getInt("UniversalAnnoyanceGenerator.numItems") //Max allowable: 10000 in ES by default; change with index.max_result_window setting

val inputVectorFilename = conf.getString("UniversalAnnoyanceGenerator.inputVectorFilename")

val esHost = conf.getString("UniversalAnnoyanceGenerator.esHost")
val esPort = conf.getInt("UniversalAnnoyanceGenerator.esPort")

val annoyResultDir = conf.getString("UniversalAnnoyanceGenerator.annoyResultDir")

val fieldNames: Array[String] = conf.getStringList("UniversalAnnoyanceGenerator.fieldNames").toArray.map(x => x.toString)

val idField:JObject = ("field" -> "id")
  val json =
      ("size" -> numItems) ~
        ("query" ->
          ("exists" ->
                ( fieldNames.map{x => 
                    val field:JObject = ("field" -> x)
                    field
                  }.fold(idField)(_ ~ _)
                )
          )
        )

    val compactJson = compact(render(json))
    logger.info(s"compact json is: ${compactJson}")

    val eventsWithContent : Seq[(Int,Int)] 
= EsClient.search(compactJson, esIndex, client) match {
      case Some(items) =>
        val hits = (items \ "hits" \ "hits").extract[Seq[JValue]]
        hits.map { hit =>
          val id = (hit \ "_id").extract[String].stripPrefix("Event-").toInt
          val source = hit \ "_source"

          val fieldIds = fieldNames.map{x =>
                           (source \ x).extract[Seq[Int]]
                         }

          //TODO : Generalize
          val field1Ids = fieldIds(0) 

          field1Ids.map(y => (id, y))
        }.flatten

      case None => Nil
  } 

  EsClient.close()


  val path = Paths.get(inputVectorFilename)

  val spark = SparkSession
    .builder()
    .appName("UniversalAnnoyanceGenerator")
    .config("spark.master", "local[*]")
    .getOrCreate()

  val vectorFieldNames = fieldNames.map(x => x + "_vector")

  val df = spark.createDataFrame(eventsWithContent).toDF((Array("id") ++ fieldNames):_*)
  val encoder = new OneHotEncoderEstimator()
    .setInputCols(fieldNames)
    .setOutputCols(vectorFieldNames)

  val assembler = (new VectorAssembler()
                    .setInputCols(vectorFieldNames)
                    .setOutputCol("features") )


  val pipeline = new Pipeline().setStages(Array(encoder,assembler))
    
  val model = pipeline.fit(df)

  import spark.implicits._

  val encoded = model.transform(df)

  val assembled: Dataset[EventWithContent] = encoded.select("id", "features").rdd.map{
    x => x.getAs[Int](0) -> x.getAs[SparseVector](1).toDense.toArray
  }.toDF("id","features").as[EventWithContent]

  val inputVectors = assembled.collect().map(x => x.id + " " + x.features.mkString(" ")).mkString("\n")

  val wrote = Files.write(path, inputVectors.getBytes("UTF-8"))

  val tmpAnnoyDir = annoyResultDir + "TMP" 

  val annoy = Annoy.create[Int](inputVectorFilename, numAnnoyTrees, outputDir = tmpAnnoyDir, Euclidean)

  val indexPath = Paths.get(tmpAnnoyDir + "/annoy-index")
  val dimensionPath = Paths.get(tmpAnnoyDir + "/dimension")
  val idsPath = Paths.get(tmpAnnoyDir + "/ids")
  val metricPath = Paths.get(tmpAnnoyDir + "/metric")

  //Possible race between this and the loader
  Files.move(indexPath, Paths.get(annoyResultDir + "/annoy-index"), ATOMIC_MOVE)
  Files.move(dimensionPath, Paths.get(annoyResultDir + "/dimension"), ATOMIC_MOVE)
  Files.move(idsPath, Paths.get(annoyResultDir + "/ids"), ATOMIC_MOVE)
  Files.move(metricPath, Paths.get(annoyResultDir + "/metric"), ATOMIC_MOVE)

  spark.stop()

  private lazy val client: RestClient = {
    EsClient.open(Seq(new HttpHost(esHost, esPort, "http")), None)
   }




}
