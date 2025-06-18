# tasks/task_a_clustering.py

# Import all necessary generic and specific libraries
import time
import json
from datetime import datetime

# Import PySpark functions and classes for data manipulation and feature engineering
from pyspark.sql.functions import col, avg, stddev_pop
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler

# Import clustering algorithms to be compared
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator

# Import our utility modules
from utils.spark_session import get_spark_session
from utils.hbase_connector import get_hbase_connection

def run_task_a():
    """
    Executes Task A: Compares K-Means and GMM, saves the models, and records
    rich evaluation metrics (including cluster sizes and model parameters) to HBase.
    """
    # 1. Initialization
    # =================
    spark = get_spark_session("Task A - Hotel Clustering Model Comparison")
    print("Task A: Model Comparison and Registration started.")
    model_metadata_list = []

    try:
        # 2. Data Loading and Feature Engineering
        # =========================================
        print("Loading data and preparing for clustering...")
        reviews_df = spark.sql("SELECT offering_id, ratings.* FROM reviews")
        reviews_df.cache()
        
        # Define columns for aggregation
        rating_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
        agg_exprs = [avg(col(c)).alias(f"avg_{c}") for c in rating_cols] + \
                    [stddev_pop(col(c)).alias(f"stddev_{c}") for c in rating_cols]
        
        # Create a statistical profile for each hotel
        hotel_stats_df = reviews_df.groupBy("offering_id").agg(*agg_exprs).na.fill(0)
        
        # Prepare data for MLlib
        feature_cols = [c for c in hotel_stats_df.columns if c != 'offering_id']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

        # --- Reusable Model Experiment Function ---
        def run_experiment(model_estimator, model_name, model_params):
            print(f"\n--- Processing {model_name} Model Experiment ---")
            
            # 3. Model Training & Evaluation
            # ===============================
            pipeline = Pipeline(stages=[assembler, scaler, model_estimator])
            
            start_time = time.time()
            model = pipeline.fit(hotel_stats_df)
            training_time = time.time() - start_time
            
            predictions = model.transform(hotel_stats_df)
            
            evaluator = ClusteringEvaluator(featuresCol='features', predictionCol='cluster_id')
            silhouette_score = evaluator.evaluate(predictions)

            # --- NEW: Calculate detailed metrics (Cluster Sizes) ---
            cluster_sizes_df = predictions.groupBy("cluster_id").count().orderBy("cluster_id")
            cluster_sizes_list = [row.asDict() for row in cluster_sizes_df.collect()]
            
            detailed_metrics_dict = {
                "cluster_sizes": cluster_sizes_list
            }
            detailed_metrics_json = json.dumps(detailed_metrics_dict)
            
            # 4. Model & Metadata Persistence
            # ================================
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = f"hdfs://localhost:9000/user/fizz/models/{model_name.lower()}_{timestamp_str}"
            
            print(f"Saving {model_name} model to: {model_path}")
            model.save(model_path)
            
            # Prepare structured metadata dictionary for HBase
            model_metadata = {
                'row_key': f'task_a_{model_name.lower()}_{timestamp_str}',
                'info:task_name': 'Task A - Clustering',
                'info:model_name': model_name,
                'info:run_timestamp': timestamp_str,
                'metrics:training_time_seconds': str(round(training_time, 2)),
                'metrics:evaluation_score': str(round(silhouette_score, 4)),
                'metrics:score_type': 'Silhouette',
                'metrics:parameters_json': json.dumps(model_params), # Store model parameters
                'artifact:model_hdfs_path': model_path,
                'details:rich_metrics_json': detailed_metrics_json # Store rich metrics
            }
            model_metadata_list.append(model_metadata)
            print(f"{model_name} - Silhouette Score: {silhouette_score:.4f}, Training Time: {training_time:.2f}s")

        # --- Run experiments for both models ---
        kmeans_params = {"k": 5, "seed": 1}
        run_experiment(KMeans(featuresCol="features", predictionCol="cluster_id", **kmeans_params), "KMeans", kmeans_params)

        gmm_params = {"k": 5, "seed": 1}
        run_experiment(GaussianMixture(featuresCol="features", predictionCol="cluster_id", **gmm_params), "GMM", gmm_params)
        
        # 5. Write all collected metadata to HBase
        # =========================================
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
            finally:
                connection.close()
                print("HBase connection closed.")

    except Exception as e:
        print(f"!!! An error occurred during the execution of Task A. !!!\nError details: {e}")
        
    finally:
        # 6. Shutdown
        # ==============
        if 'spark' in locals():
            spark.stop()
        print("\nTask A: Model experiment and registration finished.")

