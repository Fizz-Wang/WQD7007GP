# tasks/task_b_classification.py

# Import all necessary generic and specific libraries
import time
import json
from datetime import datetime
import calendar

# Import PySpark functions and classes
from pyspark.sql.functions import col, udf, when, regexp_extract
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Import classification models to be compared
from pyspark.ml.classification import DecisionTreeClassifier, NaiveBayes

# Import evaluators for classification tasks
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import our utility modules
from utils.spark_session import get_spark_session
from utils.hbase_connector import get_hbase_connection

# Define UDF for date conversion
def month_str_to_int(s):
    """Converts month name in strings like 'December 2012' to integer 1-12."""
    try:
        mon = s.split()[0]
        return list(calendar.month_name).index(mon)
    except Exception:
        return None

udf_month = udf(month_str_to_int, IntegerType())

def run_task_b():
    """
    Executes Task B: Compares Decision Tree and Naive Bayes for room experience
    classification, saves the models, and records rich metrics to HBase.
    """
    # 1. Initialization
    # =================
    spark = get_spark_session("Task B - Room Experience Classification")
    print("Task B: Classification and Registration started.")
    
    try:
        # 2. Data Loading and Feature Engineering
        # =========================================
        print("Loading data and preparing for classification...")
        df = spark.sql("""
            SELECT 
                id as review_id,
                CAST(ratings['service'] AS DOUBLE) AS service_rating,
                CAST(ratings['cleanliness'] AS DOUBLE) AS cleanliness_rating,
                CAST(ratings['overall'] AS DOUBLE) AS overall_rating,
                CAST(ratings['value'] AS DOUBLE) AS value_rating,
                CAST(ratings['location'] AS DOUBLE) AS location_rating,
                CAST(ratings['sleep_quality'] AS DOUBLE) AS sleep_quality_rating,
                CAST(ratings['rooms'] AS DOUBLE) AS rooms_rating,
                date_stayed
            FROM reviews WHERE ratings['rooms'] IS NOT NULL
        """)
        
        # --- Binning and Balancing (as per the original script) ---
        p33, p66 = df.approxQuantile("rooms_rating", [0.3333, 0.6666], 0.01)
        df_labeled = df.withColumn("rooms_label", when(col("rooms_rating") <= p33, "Low").when(col("rooms_rating") <= p66, "Medium").otherwise("High"))
        
        counts = df_labeled.groupBy("rooms_label").count().collect()
        max_count = max(r['count'] for r in counts)
        df_balanced = None
        for row in counts:
            ratio = max_count / row['count']
            # --- CORRECTED LINE ---
            # Changed row['label'] to the correct column name row['rooms_label']
            sampled = df_labeled.filter(col("rooms_label") == row['rooms_label']).sample(withReplacement=True, fraction=ratio, seed=42)
            df_balanced = sampled if df_balanced is None else df_balanced.union(sampled)
            
        # --- Final Feature Preparation ---
        df_feat = df_balanced.withColumn("month", udf_month(col("date_stayed"))) \
                             .withColumn("year", regexp_extract(col("date_stayed"), r"(\d{4})", 1).cast(IntegerType()))
        
        feature_cols = ["service_rating", "cleanliness_rating", "overall_rating", "value_rating", "location_rating", "sleep_quality_rating", "month", "year"]
        df_feat = df_feat.na.drop(subset=feature_cols + ["rooms_label"])
        df_feat.cache()

        # Split data into training and test sets
        (trainingData, testData) = df_feat.randomSplit([0.8, 0.2], seed=42)
        
        # Prepare pipeline stages that are common to both models
        indexer = StringIndexer(inputCol="rooms_label", outputCol="label").fit(df_feat)
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

        model_metadata_list = []

        # --- Reusable Model Experiment Function ---
        def run_experiment(model_estimator, model_name, model_params):
            print(f"\n--- Processing {model_name} Model Experiment ---")
            
            # 3. Model Training & Evaluation
            # ===============================
            pipeline = Pipeline(stages=[indexer, assembler, model_estimator])
            
            start_time = time.time()
            model = pipeline.fit(trainingData)
            training_time = time.time() - start_time
            
            predictions = model.transform(testData)
            
            # --- Rich Metrics Calculation ---
            # Primary metric: Macro-averaged F1-score
            evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
            f1_score = evaluator_f1.evaluate(predictions)

            # Detailed metrics: Precision, Recall, and Confusion Matrix
            evaluator_pr = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
            precision = evaluator_pr.evaluate(predictions, {evaluator_pr.metricName: "weightedPrecision"})
            recall = evaluator_pr.evaluate(predictions, {evaluator_pr.metricName: "weightedRecall"})
            
            # Confusion Matrix
            confusion_matrix_df = predictions.groupBy("label", "prediction").count()
            confusion_matrix_list = [row.asDict() for row in confusion_matrix_df.collect()]
            
            detailed_metrics_dict = {
                "f1_score_macro": round(f1_score, 4),
                "precision_weighted": round(precision, 4),
                "recall_weighted": round(recall, 4),
                "confusion_matrix": confusion_matrix_list
            }
            detailed_metrics_json = json.dumps(detailed_metrics_dict)
            
            # 4. Model & Metadata Persistence
            # ================================
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = f"hdfs://localhost:9000/user/fizz/models/{model_name.lower()}_{timestamp_str}"
            
            print(f"Saving {model_name} model to: {model_path}")
            model.save(model_path)
            
            # Prepare structured metadata dictionary
            model_metadata = {
                'row_key': f'task_b_{model_name.lower()}_{timestamp_str}',
                'info:task_name': 'Task B - Room Experience Classification',
                'info:model_name': model_name,
                'info:run_timestamp': timestamp_str,
                'metrics:training_time_seconds': str(round(training_time, 2)),
                'metrics:evaluation_score': str(round(f1_score, 4)),
                'metrics:score_type': 'Macro F1-Score',
                'metrics:parameters_json': json.dumps(model_params),
                'artifact:model_hdfs_path': model_path,
                'details:rich_metrics_json': detailed_metrics_json
            }
            model_metadata_list.append(model_metadata)
            print(f"{model_name} - F1 Score: {f1_score:.4f}, Training Time: {training_time:.2f}s")

        # --- Run experiments for both models ---
        dt_params = {"maxDepth": 5}
        run_experiment(DecisionTreeClassifier(labelCol="label", featuresCol="features", **dt_params), "DecisionTree", dt_params)
        
        nb_params = {"smoothing": 1.0}
        run_experiment(NaiveBayes(labelCol="label", featuresCol="features", **nb_params), "NaiveBayes", nb_params)
        
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
        print(f"!!! An error occurred during the execution of Task B. !!!\nError details: {e}")
        
    finally:
        # 6. Shutdown
        # ==============
        if 'spark' in locals():
            spark.stop()
        print("\nTask B: Classification experiment and registration finished.")

