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
val oneHotEncoderFilename = conf.getString("UniversalAnnoyanceGenerator.oneHotEncoderFilename")

val esHost = conf.getString("UniversalAnnoyanceGenerator.esHost")
val esPort = conf.getInt("UniversalAnnoyanceGenerator.esPort")

val annoyResultDir = conf.getString("UniversalAnnoyanceGenerator.annoyResultDir")

val fieldNames = conf.getStringList("UniversalAnnoyanceGenerator.fieldNames")

//Hard-coded to 2 fields 
val field1 = fieldNames.get(0)
val field2 = fieldNames.get(1)

  val json =
      ("size" -> numItems) ~
        ("query" ->
          ("exists" ->
                ("field" -> "id") ~
                ("field" -> field1) ~
                ("field" -> field2)) 
           ) 

    val compactJson = compact(render(json))
    logger.info(s"compact json is: ${compactJson}")

    val eventsWithContent : Seq[(Int, Int, Int)] 
= EsClient.search(compactJson, esIndex, client) match {
      case Some(items) =>
        val hits = (items \ "hits" \ "hits").extract[Seq[JValue]]
        hits.map { hit =>
          val id = (hit \ "_id").extract[String].stripPrefix("Event-").toInt
          val source = hit \ "_source"


          val field1Ids = (source \ field1).extract[Seq[Int]]
          val field2Ids = (source \ field2).extract[Seq[Int]]
//          val distributionChannelIds = (hit \ "distribution_channel_ids").extract[Seq[Int]]

          field2Ids.map(x => field1Ids.map(y => (id, y, x)))
        }.flatten.flatten 
  } 

  EsClient.close()


  val path = Paths.get(inputVectorFilename)

  val spark = SparkSession
    .builder()
    .appName("UniversalAnnoyanceGenerator")
    .config("spark.master", "local[*]")
    .getOrCreate()

  val df = spark.createDataFrame(eventsWithContent).toDF("id", field1, field2)
  val encoder = new OneHotEncoderEstimator()
    .setInputCols(Array(field1, field2))
    .setOutputCols(Array(field1 + "_vector", field2 + "_vector"))
 
  val assembler = (new VectorAssembler()
                    .setInputCols(Array(field1 +"_vector", field2 + "_vector"))
                    .setOutputCol("features") )


  val pipeline = new Pipeline().setStages(Array(encoder,assembler))
    
  val model = pipeline.fit(df)

  import spark.implicits._

  val encoded = model.transform(df)

  val assembled: Dataset[EventWithContent] = encoded.select("id", "features").rdd.map{
    x => x.getAs[Int](0) -> x.getAs[SparseVector](1).toDense.toArray
  }.toDF("id","features").as[EventWithContent]

  val inputVectors = assembled.collect().map(x => x.id + " " + x.features.mkString(" ")).mkString("\n")
//  logger.info(encoded)

//  logger.info(assembled)

//  logger.info(inputVectors)

  val wrote = Files.write(path, inputVectors.getBytes("UTF-8"))

  val tmpAnnoyDir = annoyResultDir + "TMP" 

  val annoy = Annoy.create[Int](inputVectorFilename, numAnnoyTrees, outputDir = tmpAnnoyDir, Euclidean)

  val indexPath = Paths.get(tmpAnnoyDir + "/annoy-index")
  val dimensionPath = Paths.get(tmpAnnoyDir + "/dimension")
  val idsPath = Paths.get(tmpAnnoyDir + "/ids")
  val metricPath = Paths.get(tmpAnnoyDir + "/metric")

  Files.move(indexPath, Paths.get(annoyResultDir + "/annoy-index"), ATOMIC_MOVE)
  Files.move(dimensionPath, Paths.get(annoyResultDir + "/dimension"), ATOMIC_MOVE)
  Files.move(idsPath, Paths.get(annoyResultDir + "/ids"), ATOMIC_MOVE)
  Files.move(metricPath, Paths.get(annoyResultDir + "/metric"), ATOMIC_MOVE)

  spark.stop()

  private lazy val client: RestClient = {
    EsClient.open(Seq(new HttpHost(esHost, esPort, "http")), None)
   }

   //From the UR
   /*
    /** Performs a search using the JSON query String
   *
   *  @param query the JSON query string parable by Elasticsearch
   *  @param indexName the index to search
   *  @return a [PredictedResults] collection
   */
  def search(query: String, indexName: String): Option[JValue] = {
    logger.info(s"Query:\n${query}")
    val response = client.performRequest(
      "POST",
      s"/$indexName/_search",
      Map.empty[String, String].asJava,
      new StringEntity(query, ContentType.APPLICATION_JSON))
    response.getStatusLine.getStatusCode match {
      case 200 =>
        logger.info(s"Got source from query: ${query}")
        Some(parse(EntityUtils.toString(response.getEntity)))
      case _ =>
        logger.info(s"Query: ${query}\nproduced status code: ${response.getStatusLine.getStatusCode}")
        None
    }
  }

    def open(
    hosts: Seq[HttpHost],
    basicAuth: Option[(String, String)] = None): RestClient = {
    val newClient = _sharedRestClient match {
      case Some(c) => c
      case None => {
        var builder = RestClient.builder(hosts: _*)
        builder = basicAuth match {
          case Some((username, password)) => builder.setHttpClientConfigCallback(
            new BasicAuthProvider(username, password))
          case None => builder
        }
        builder.setRequestConfigCallback(new RestClientBuilder.RequestConfigCallback() {
          @Override
          def customizeRequestConfig(requestConfigBuilder: RequestConfig.Builder): RequestConfig.Builder = {
            return requestConfigBuilder.setConnectTimeout(30000)
              .setSocketTimeout(200000);
          }
        })
          .setMaxRetryTimeoutMillis(200000);
        builder.build()
      }
    }
    _sharedRestClient = Some(newClient)
    newClient
  }

  def close(): Unit = {
    if (!_sharedRestClient.isEmpty) {
      _sharedRestClient.get.close()
      _sharedRestClient = None
    }
  }

  class BasicAuthProvider(
  val username: String,
  val password: String)
    extends HttpClientConfigCallback {

  val credentialsProvider = new BasicCredentialsProvider()
  credentialsProvider.setCredentials(
    AuthScope.ANY,
    new UsernamePasswordCredentials(username, password))

  override def customizeHttpClient(
    httpClientBuilder: HttpAsyncClientBuilder): HttpAsyncClientBuilder = {
    httpClientBuilder.setDefaultCredentialsProvider(credentialsProvider)
  }
}

*/

}

