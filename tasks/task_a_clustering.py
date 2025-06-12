# tasks/task_a_clustering.py

import time
from datetime import datetime

# Import necessary functions and classes from PySpark
from pyspark.sql.functions import col, avg, stddev_pop
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator

# Import our utility modules
from utils.spark_session import get_spark_session
from utils.hbase_connector import get_hbase_connection

def run_task_a():
    """
    Executes Task A: Compares K-Means and GMM, saves the trained models to HDFS,
    and records their evaluation metrics and artifacts to the HBase 'model_registry' table.
    """
    spark = get_spark_session("Task A - Hotel Clustering Model Comparison")
    print("Task A: Model Comparison and Registration started.")
    
    try:
        # 1. Load and prepare data (same as before)
        print("Loading and preparing data...")
        reviews_df = spark.sql("SELECT offering_id, ratings.* FROM reviews")
        reviews_df.cache()
        rating_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
        agg_exprs = [avg(col(c)).alias(f"avg_{c}") for c in rating_cols] + \
                    [stddev_pop(col(c)).alias(f"stddev_{c}") for c in rating_cols]
        hotel_stats_df = reviews_df.groupBy("offering_id").agg(*agg_exprs).na.fill(0)
        feature_cols = [c for c in hotel_stats_df.columns if c != 'offering_id']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

        model_metadata_list = []

        # =================================================================
        # 2. K-MEANS MODEL EXPERIMENT
        # =================================================================
        print("\n--- Processing K-Means Model Experiment ---")
        kmeans = KMeans(k=5, seed=1, featuresCol="features", predictionCol="cluster_id")
        pipeline_kmeans = Pipeline(stages=[assembler, scaler, kmeans])
        
        start_time_kmeans = time.time()
        model_kmeans = pipeline_kmeans.fit(hotel_stats_df)
        training_time_kmeans = time.time() - start_time_kmeans
        
        predictions_kmeans = model_kmeans.transform(hotel_stats_df)
        evaluator = ClusteringEvaluator(featuresCol='features', predictionCol='cluster_id')
        silhouette_kmeans = evaluator.evaluate(predictions_kmeans)
        
        # --- Save the K-Means model to HDFS ---
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # CORRECTED HDFS PATH: Added 'localhost:9000'
        kmeans_model_path = f"hdfs://localhost:9000/user/fizz/models/kmeans_{timestamp_str}"
        
        print(f"Saving K-Means model to: {kmeans_model_path}")
        model_kmeans.save(kmeans_model_path)
        
        # --- Prepare K-Means metadata for HBase ---
        kmeans_metadata = {
            'row_key': f'task_a_kmeans_{timestamp_str}',
            'info:task_name': 'Task A - Clustering',
            'info:model_name': 'KMeans',
            'info:run_timestamp': timestamp_str,
            'metrics:training_time_seconds': str(round(training_time_kmeans, 2)),
            'metrics:evaluation_score': str(round(silhouette_kmeans, 4)),
            'metrics:score_type': 'Silhouette',
            'artifact:model_hdfs_path': kmeans_model_path
        }
        model_metadata_list.append(kmeans_metadata)
        print(f"K-Means - Silhouette Score: {silhouette_kmeans:.4f}, Training Time: {training_time_kmeans:.2f}s")

        # =================================================================
        # 3. GAUSSIAN MIXTURE MODEL (GMM) EXPERIMENT
        # =================================================================
        print("\n--- Processing Gaussian Mixture Model (GMM) Experiment ---")
        gmm = GaussianMixture(k=5, seed=1, featuresCol="features", predictionCol="cluster_id")
        pipeline_gmm = Pipeline(stages=[assembler, scaler, gmm])

        start_time_gmm = time.time()
        model_gmm = pipeline_gmm.fit(hotel_stats_df)
        training_time_gmm = time.time() - start_time_gmm

        predictions_gmm = model_gmm.transform(hotel_stats_df)
        silhouette_gmm = evaluator.evaluate(predictions_gmm)

        timestamp_str_gmm = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # CORRECTED HDFS PATH: Added 'localhost:9000'
        gmm_model_path = f"hdfs://localhost:9000/user/fizz/models/gmm_{timestamp_str_gmm}"

        print(f"Saving GMM model to: {gmm_model_path}")
        model_gmm.save(gmm_model_path)

        gmm_metadata = {
            'row_key': f'task_a_gmm_{timestamp_str_gmm}',
            'info:task_name': 'Task A - Clustering',
            'info:model_name': 'GMM',
            'info:run_timestamp': timestamp_str_gmm,
            'metrics:training_time_seconds': str(round(training_time_gmm, 2)),
            'metrics:evaluation_score': str(round(silhouette_gmm, 4)),
            'metrics:score_type': 'Silhouette',
            'artifact:model_hdfs_path': gmm_model_path
        }
        model_metadata_list.append(gmm_metadata)
        print(f"GMM - Silhouette Score: {silhouette_gmm:.4f}, Training Time: {training_time_gmm:.2f}s")
        
        # =================================================================
        # 4. Write all collected metadata to HBase
        # =================================================================
        print("\n--- Writing model metadata to HBase table 'model_registry' ---")
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
            except Exception as hbase_e:
                print(f"Error writing to HBase: {hbase_e}")
            finally:
                connection.close()
                print("HBase connection closed.")

    except Exception as e:
        print(f"!!! An error occurred during the execution of Task A. !!!\nError details: {e}")
        
    finally:
        if 'spark' in locals():
            spark.stop()
        print("\nTask A: Model experiment and registration finished.")

