# tasks/task_d_anomaly.py

# Import all necessary generic and specific libraries
import time
import json
from datetime import datetime

# Import PySpark functions and classes
# CORRECTED LINE: Added 'count' to the import list
from pyspark.sql.functions import col, length, size, split, avg, stddev, count
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import DoubleType

# Import our utility modules
from utils.spark_session import get_spark_session
from utils.hbase_connector import get_hbase_connection

def run_task_d():
    """
    Executes Task D: Anomaly Detection on both user and review levels.
    This script uses KMeans as a proxy for anomaly detection and follows the
    standardized workflow for model training, saving, and metadata registration.
    """
    # 1. Initialization
    # =================
    spark = get_spark_session("Task D - Anomaly Detection")
    print("Task D: Anomaly Detection and Registration started.")
    model_metadata_list = []

    try:
        reviews_df = spark.sql("SELECT * FROM reviews")
        reviews_df.cache()

        # --- Reusable Model Experiment Function ---
        def run_experiment(df, feature_cols, model_estimator, model_name, model_params, task_name_desc, row_key_prefix):
            """A generic function to handle a full anomaly detection experiment."""
            print(f"\n--- Processing {model_name} for {task_name_desc} ---")
            
            # 3. Model Training & Evaluation
            # ===============================
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
            scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
            pipeline = Pipeline(stages=[assembler, scaler, model_estimator])
            
            start_time = time.time()
            model = pipeline.fit(df)
            training_time = time.time() - start_time
            
            predictions = model.transform(df)
            
            # --- Rich Metrics Calculation ---
            # For anomaly detection with KMeans, we analyze the distribution of points.
            # Points in very small clusters can be considered anomalies.
            cluster_counts_df = predictions.groupBy("cluster_id").count().orderBy("count")
            cluster_counts_list = [row.asDict() for row in cluster_counts_df.collect()]
            
            # Define anomaly by cluster size (e.g., clusters with <1% of total data)
            total_count = predictions.count()
            anomaly_threshold = total_count * 0.01 
            anomalous_clusters = [c for c in cluster_counts_list if c['count'] < anomaly_threshold]
            anomaly_count = sum(c['count'] for c in anomalous_clusters)

            detailed_metrics_dict = {
                "anomaly_threshold_by_cluster_size": int(anomaly_threshold),
                "total_anomalies_found": anomaly_count,
                "cluster_distribution": cluster_counts_list
            }
            detailed_metrics_json = json.dumps(detailed_metrics_dict)
            
            # Primary evaluation score (placeholder, could be avg distance, etc.)
            # We use the fraction of anomalies as a score for this example.
            evaluation_score = anomaly_count / total_count if total_count > 0 else 0.0

            # 4. Model & Metadata Persistence
            # ================================
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = f"hdfs://localhost:9000/user/fizz/models/{model_name.lower()}_{timestamp_str}"
            
            print(f"Saving {model_name} model to: {model_path}")
            model.save(model_path)
            
            # Prepare structured metadata dictionary
            model_metadata = {
                'row_key': f'{row_key_prefix}_{model_name.lower()}_{timestamp_str}',
                'info:task_name': task_name_desc,
                'info:model_name': model_name,
                'info:run_timestamp': timestamp_str,
                'metrics:training_time_seconds': str(round(training_time, 2)),
                'metrics:evaluation_score': str(round(evaluation_score, 4)),
                'metrics:score_type': 'Anomaly_Fraction',
                'metrics:parameters_json': json.dumps(model_params),
                'artifact:model_hdfs_path': model_path,
                'details:rich_metrics_json': detailed_metrics_json
            }
            model_metadata_list.append(model_metadata)
            print(f"{model_name} - Anomaly Fraction: {evaluation_score:.4f}, Training Time: {training_time:.2f}s")

        # =================================================================
        # 2. USER-LEVEL ANOMALY DETECTION
        # =================================================================
        print("\n--- Starting User-Level Anomaly Detection ---")
        user_features_df = reviews_df.groupBy("author.id") \
            .agg(
                count("*").alias("review_count"),
                avg(length(col("text"))).alias("avg_text_length"),
                avg(col("author.num_helpful_votes")).alias("avg_helpful_votes")
            ).na.fill(0)
        
        user_feature_cols = ["review_count", "avg_text_length", "avg_helpful_votes"]
        kmeans_user_params = {"k": 10, "seed": 42} # Use more clusters to find small outlier groups
        kmeans_user = KMeans(featuresCol="features", predictionCol="cluster_id", **kmeans_user_params)
        
        run_experiment(user_features_df, user_feature_cols, kmeans_user, "KMeans_User_Anomaly", kmeans_user_params, "Task D - User Anomaly", "task_d_user")

        # =================================================================
        # 2. REVIEW-LEVEL ANOMALY DETECTION
        # =================================================================
        print("\n--- Starting Review-Level Anomaly Detection ---")
        review_features_df = reviews_df.withColumn("text_length", length(col("text"))) \
                                       .withColumn("word_count", size(split(col("text"), r"\\s+"))) \
                                       .select("id", "text_length", "word_count").na.fill(0)

        review_feature_cols = ["text_length", "word_count"]
        kmeans_review_params = {"k": 10, "seed": 42}
        kmeans_review = KMeans(featuresCol="features", predictionCol="cluster_id", **kmeans_review_params)
        
        run_experiment(review_features_df, review_feature_cols, kmeans_review, "KMeans_Review_Anomaly", kmeans_review_params, "Task D - Review Anomaly", "task_d_review")

        # 5. Write all collected metadata to HBase
        # =========================================
        print("\n--- Writing model experiment metadata to HBase table 'model_registry' ---")
        connection = get_hbase_connection()
        if connection:
            try:
                table = connection.table('model_registry')
                with table.batch() as b:
                    for metadata in model_metadata_list:
                        row_key = metadata.pop('row_key')
                        data_to_write = {k.encode('utf-8'): str(v).encode('utf-8') for k, v in metadata.items()}
                        b.put(row_key.encode('utf-8'), data_to_write)
                print("Metadata successfully written to HBase.")
            finally:
                connection.close()
                print("HBase connection closed.")

    except Exception as e:
        print(f"!!! An error occurred during the execution of Task D. !!!\nError details: {e}")
        
    finally:
        # 6. Shutdown
        # ==============
        if 'spark' in locals():
            spark.stop()
        print("\nTask D: Anomaly Detection experiment and registration finished.")

