from utils.spark_session import get_spark_session
import time
from datetime import datetime
import calendar

from pyspark.sql.functions import col, udf, when, regexp_extract
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from utils.hbase_connector import get_hbase_connection

def month_str_to_int(s):
    """
    Converts month name in strings like 'December 2012' to integer 1-12.
    """
    try:
        mon = s.split()[0]
        return list(calendar.month_name).index(mon)
    except Exception:
        return None

# Register UDF
udf_month = udf(month_str_to_int, IntegerType())

def run_task_b():
    """Executes Task B: Room Experience Classification."""
    """
    Executes Task B: Room Experience Classification.
    Steps:
      1. Load raw data from Hive
      2. Compute quantile-based bins for rooms_rating
      3. Balance classes via oversampling
      4. Extract date features
      5. Train Decision Tree and Naive Bayes, evaluate macro F1
      6. Persist models to HDFS
      7. Write metadata to HBase
    """

    spark = get_spark_session("Task B - Room Experience Classification")
    print("Task B: Classification started.")
    # 1. Load raw data
    df = spark.sql(
        """
        SELECT
          CAST(ratings['service'] AS DOUBLE)       AS service_rating,
          CAST(ratings['cleanliness'] AS DOUBLE)   AS cleanliness_rating,
          CAST(ratings['overall'] AS DOUBLE)       AS overall_rating,
          CAST(ratings['value'] AS DOUBLE)         AS value_rating,
          CAST(ratings['location'] AS DOUBLE)      AS location_rating,
          CAST(ratings['sleep_quality'] AS DOUBLE) AS sleep_quality_rating,
          CAST(ratings['rooms'] AS DOUBLE)         AS rooms_rating,
          date_stayed,
          id                                      AS review_id
        FROM reviews
        WHERE ratings['rooms'] IS NOT NULL
        """
    )
    total = df.count()
    print(f"Loaded {total} records from Hive.")

    # 2. Compute quantiles for binning
    p33, p66 = df.approxQuantile("rooms_rating", [0.3333, 0.6666], 0.01)
    print(f"Quantile thresholds -> 33%: {p33:.4f}, 66%: {p66:.4f}")

    # 2.1 Assign labels based on quantiles
    df_labeled = df.withColumn(
        "rooms_label",
        when(col("rooms_rating") <= p33, "Low")
        .when(col("rooms_rating") <= p66, "Medium")
        .otherwise("High")
    )
    print("Class distribution before balancing:")
    df_labeled.groupBy("rooms_label").count().show()

    # 3. Balance classes via oversampling
    counts = df_labeled.groupBy("rooms_label").count().collect()
    max_count = max(r['count'] for r in counts)
    df_balanced = None
    for row in counts:
        label = row['rooms_label']
        cnt = row['count']
        ratio = max_count / cnt
        sampled = df_labeled.filter(col("rooms_label") == label).sample(True, ratio, seed=42)
        sampled_count = sampled.count()
        print(f"Oversampled label={label}: target={max_count}, got={sampled_count}, ratio={ratio:.2f}")
        df_balanced = sampled if df_balanced is None else df_balanced.union(sampled)

    print("Class distribution after balancing:")
    df_balanced.groupBy("rooms_label").count().show()

    # 4. Date feature extraction
    df_feat = df_balanced.withColumn("month", udf_month(col("date_stayed"))) \
                         .withColumn("year", regexp_extract(col("date_stayed"), r"(\d{4})", 1).cast(IntegerType()))
    df_feat = df_feat.filter(col("month").isNotNull() & col("year").isNotNull())
    feature_cols = [
    "service_rating", "cleanliness_rating", "overall_rating",
    "value_rating", "location_rating", "sleep_quality_rating",
    "month", "year"]
    df_feat = df_feat.na.drop(subset=feature_cols)
    print(f"Records after date parsing: {df_feat.count()}")

    # 5. Prepare pipeline and split data
    indexer = StringIndexer(inputCol="rooms_label", outputCol="label").fit(df_feat)
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5)
    nb = NaiveBayes(labelCol="label", featuresCol="features", smoothing=1.0)

    train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
    print(f"Train count={train_df.count()}, Test count={test_df.count()}")

    # 6. Train models
    start_dt = time.time()
    pipeline_dt = Pipeline(stages=[indexer, assembler, dt])
    model_dt = pipeline_dt.fit(train_df)
    dt_time = time.time() - start_dt

    start_nb = time.time()
    pipeline_nb = Pipeline(stages=[indexer, assembler, nb])
    model_nb = pipeline_nb.fit(train_df)
    nb_time = time.time() - start_nb

    # 7. Evaluate
    pred_dt = model_dt.transform(test_df)
    pred_nb = model_nb.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_dt = evaluator.evaluate(pred_dt)
    f1_nb = evaluator.evaluate(pred_nb)
    print(f"DecisionTree F1={f1_dt:.4f}, time={dt_time:.2f}s")
    print(f"NaiveBayes   F1={f1_nb:.4f}, time={nb_time:.2f}s")

    # 8. Save models
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    dt_path = f"hdfs://localhost:9000/user/fizz/models/decision_tree_{ts}"
    nb_path = f"hdfs://localhost:9000/user/fizz/models/naive_bayes_{ts}"
    model_dt.write().overwrite().save(dt_path)
    model_nb.write().overwrite().save(nb_path)
    print(f"Saved models to {dt_path} and {nb_path}")

    # 9. Write metadata to HBase
    conn = get_hbase_connection()
    if conn:
        table = conn.table('model_registry')
        row_dt = f"task2_dt_{ts}"
        meta_dt = {
            'info:task_name': 'Task B - Room Experience Classification',
            'info:model_name': 'DecisionTree',
            'info:run_timestamp': ts,
            'metrics:training_time_seconds': str(round(dt_time,2)),
            'metrics:evaluation_score': str(round(f1_dt,4)),
            'metrics:score_type': 'MacroF1',
            'artifact:model_hdfs_path': dt_path
        }
        row_nb = f"task2_nb_{ts}"
        meta_nb = {
            'info:task_name': 'Task B - Room Experience Classification',
            'info:model_name': 'NaiveBayes',
            'info:run_timestamp': ts,
            'metrics:training_time_seconds': str(round(nb_time,2)),
            'metrics:evaluation_score': str(round(f1_nb,4)),
            'metrics:score_type': 'MacroF1',
            'artifact:model_hdfs_path': nb_path
        }
        with table.batch() as b:
            b.put(row_dt.encode(), {k.encode(): v.encode() for k,v in meta_dt.items()})
            b.put(row_nb.encode(), {k.encode(): v.encode() for k,v in meta_nb.items()})
        conn.close()
        print("Metadata written to HBase")

    spark.stop()
    print("=== Task B finished ===")
