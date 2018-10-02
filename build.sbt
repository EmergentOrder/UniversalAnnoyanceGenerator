val sparkVersion = "2.3.1"

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.11.12",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "URAnnoyIndexGenerator",
    libraryDependencies ++= Seq("com.typesafe" % "config" % "1.3.2",
                                "org.apache.httpcomponents" % "httpclient" % "4.5.6",
                                "org.elasticsearch.client" % "elasticsearch-rest-client" % "5.6.9",
                                "org.apache.httpcomponents" % "httpasyncclient" % "4.1.4", 
                                "org.clapper" %% "grizzled-slf4j" % "1.0.2", 
                                "org.json4s" %% "json4s-jackson" % "3.2.11",
                                "net.pishen" %% "annoy4s" % "0.6.0",
                                "org.apache.spark" %% "spark-sql" % sparkVersion,
                                "org.apache.spark" %% "spark-mllib" % sparkVersion)
  )
