# tasks/task_c_sentiment.py

# Import all necessary generic and specific libraries
import time
import json
from datetime import datetime

# Import PySpark functions and classes for data manipulation and feature engineering
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler

# Import classification models to be compared
from pyspark.ml.classification import LogisticRegression, LinearSVC

# Import evaluators for classification tasks
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Import our utility modules
from utils.spark_session import get_spark_session
from utils.hbase_connector import get_hbase_connection

def run_task_c():
    """
    Executes Task C: Compares Logistic Regression and SVM for sentiment analysis,
    saves the models, and records rich evaluation metrics to HBase.
    """
    # 1. Initialization
    # =================
    spark = get_spark_session("Task C - Sentiment Analysis Model Comparison")
    print("Task C: Sentiment Analysis and Registration started.")
    model_metadata_list = []

    try:
        # 2. Data Loading and Feature Engineering
        # =========================================
        print("Loading data and preparing for sentiment analysis...")
        
        # Load only the columns needed for this task
        reviews_df = spark.sql("SELECT id, text, ratings.overall FROM reviews")
        
        # Create the binary label: overall rating >= 4 is positive (1.0), else negative (0.0)
        labeled_df = reviews_df.withColumn("label", when(col("overall") >= 4.0, 1.0).otherwise(0.0))
        
        # Split data into training and test sets for supervised learning
        (trainingData, testData) = labeled_df.randomSplit([0.8, 0.2], seed=42)
        trainingData.cache()
        testData.cache()

        # --- Define the text processing and feature engineering pipeline ---
        # Stage 1: Tokenize the review text into words
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        # Stage 2: Remove common stop words
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        # Stage 3: Convert words to a numerical vector using HashingTF
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=20000)
        # Stage 4: Apply IDF to weigh the terms
        idf = IDF(inputCol="raw_features", outputCol="features")

        # The base pipeline for feature engineering is the same for both models
        feature_pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
        
        # --- Run the feature engineering pipeline ---
        feature_model = feature_pipeline.fit(trainingData)
        training_features = feature_model.transform(trainingData)
        test_features = feature_model.transform(testData)

        # --- Model Experiment Function ---
        # To avoid code duplication, we create a function to handle model experiments
        def run_experiment(model_estimator, model_name):
            print(f"\n--- Processing {model_name} Model Experiment ---")
            
            # 3. Model Training
            # =================
            start_time = time.time()
            model = model_estimator.fit(training_features)
            training_time = time.time() - start_time
            
            # Make predictions on the test set
            predictions = model.transform(test_features)
            
            # --- Evaluation with rich metrics ---
            # Primary metric: Area Under ROC (AUC)
            evaluator_auc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
            auc_score = evaluator_auc.evaluate(predictions)

            # Detailed metrics: F1, Precision, Recall
            evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
            f1_score = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
            precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
            recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})

            # Detailed metrics: Confusion Matrix
            confusion_matrix_df = predictions.groupBy("label", "prediction").count()
            confusion_matrix_list = [row.asDict() for row in confusion_matrix_df.collect()]
            
            # Package detailed metrics into a JSON string
            detailed_metrics_dict = {
                "f1_score": round(f1_score, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "confusion_matrix": confusion_matrix_list
            }
            detailed_metrics_json = json.dumps(detailed_metrics_dict)
            
            # 4. Model & Metadata Persistence
            # ================================
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = f"hdfs://localhost:9000/user/fizz/models/{model_name.lower()}_{timestamp_str}"
            print(f"Saving {model_name} model to: {model_path}")
            model.save(model_path)

            # Prepare structured metadata for HBase
            model_metadata = {
                'row_key': f'task_c_{model_name.lower()}_{timestamp_str}',
                'info:task_name': 'Task C - Sentiment Analysis',
                'info:model_name': model_name,
                'info:run_timestamp': timestamp_str,
                'metrics:training_time_seconds': str(round(training_time, 2)),
                'metrics:evaluation_score': str(round(auc_score, 4)),
                'metrics:score_type': 'AUC',
                'artifact:model_hdfs_path': model_path,
                'details:rich_metrics_json': detailed_metrics_json
            }
            model_metadata_list.append(model_metadata)
            print(f"{model_name} - AUC: {auc_score:.4f}, Training Time: {training_time:.2f}s")

        # --- Run experiments for both models ---
        run_experiment(LogisticRegression(featuresCol="features", labelCol="label"), "LogisticRegression")
        run_experiment(LinearSVC(featuresCol="features", labelCol="label"), "LinearSVM")
        
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
        print(f"!!! An error occurred during the execution of Task C. !!!\nError details: {e}")
        
    finally:
        # 6. Shutdown
        # ==============
        if 'spark' in locals():
            spark.stop()
        print("\nTask C: Sentiment Analysis experiment and registration finished.")


