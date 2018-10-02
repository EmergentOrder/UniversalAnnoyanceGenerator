/*
 * Copyright ActionML, LLC under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * ActionML licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.actionml

import java.net.URLEncoder
import java.util

import grizzled.slf4j.Logger
import org.apache.http.util.EntityUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.elasticsearch.client.RestClient
import org.apache.http.HttpHost
import org.apache.http.auth.{ AuthScope, UsernamePasswordCredentials }
import org.apache.http.entity.ContentType
import org.apache.http.entity.StringEntity
import org.apache.http.client.config.RequestConfig
import org.apache.http.impl.client.BasicCredentialsProvider
import org.apache.http.impl.nio.client.HttpAsyncClientBuilder
import org.apache.http.nio.entity.NStringEntity
import org.joda.time.DateTime
import org.json4s.jackson.JsonMethods._
import org.elasticsearch.client.RestClient
import org.elasticsearch.client.RestClientBuilder
import org.elasticsearch.client.RestClientBuilder.HttpClientConfigCallback

import org.json4s.JValue
import org.json4s.DefaultFormats
import org.json4s.JsonAST.JString
// import org.json4s.native.Serialization.writePretty

import scala.collection.immutable
import scala.collection.parallel.mutable
import scala.collection.JavaConverters._

/** Elasticsearch notes:
 *  1) every query clause will affect scores unless it has a constant_score and boost: 0
 *  2) the Spark index writer is fast but must assemble all data for the index before the write occurs
 *  3) to use like a DB you must specify that the index of fields are `not_analyzed` so they won't be lowercased,
 *    stemmed, tokenized, etc. Then the values are literal and must match exactly what is in the query (no analyzer)
 *    Finally the correlator fields should be norms: true to enable norms, this make the score equal to the sum
 *    of dot products divided by the length of each vector. This is the definition of "cosine" similarity.
 *  4) the client, either transport client for < ES5 or the rest client for >= ES5 there should have a timeout set since
 *    the default is very long, several seconds.
 *
 */

/** Defines methods to use on Elasticsearch 5 through the REST client.*/
object EsClient  {
  @transient lazy val logger: Logger = Logger[this.type]

  implicit val formats = DefaultFormats

  var _sharedRestClient: Option[RestClient] = None

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

  /** Performs a search using the JSON query String
   *
   *  @param query the JSON query string parable by Elasticsearch
   *  @param indexName the index to search
   *  @return a [PredictedResults] collection
   */
  def search(query: String, indexName: String, client: RestClient): Option[JValue] = {
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
