from pyspark.sql import SparkSession

def get_spark_session(app_name="WQD7007GP_Spark_App"):
    """
    Creates and returns a SparkSession instance with Hive support enabled.
    All tasks should call this function to get a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .getOrCreate()
    return spark
