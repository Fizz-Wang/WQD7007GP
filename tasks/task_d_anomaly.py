import time
from datetime import datetime
from pyspark.sql.functions import col, count, avg, length, stddev, when, datediff, unix_timestamp, min as spark_min, max as spark_max, countDistinct, mean, size, split, regexp_replace, hour, udf
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import DoubleType
from utils.spark_session import get_spark_session
from utils.hbase_connector import get_hbase_connection

def safe_mean(df, col_name, default=0):
    result = df.select(mean(col_name)).first()[0]
    return result if result is not None else default

def safe_left_join(df1, df2, join_col):
    if df2 is None or df2.count() == 0:
        return df1
    return df1.join(df2, on=join_col, how='left_outer')

def run_task_d():
    spark = get_spark_session("Task D - Final Anomaly Detection")
    model_metadata_list = []

    try:
        # 推荐根据数据量动态调整以下参数
        reviews = spark.table("reviews").repartition(8)  # 提高并发度，减少单个 task 压力

        # =================== 用户层特征工程 ===================
        filtered_reviews_user = reviews.filter((col("author.id").isNotNull()))
        base_user = filtered_reviews_user.select(col("author.id").alias("user_id"))

        feat_review_count = base_user.groupBy("user_id").agg(count("*").alias("review_count"))
        feat_text_length = filtered_reviews_user.groupBy(col("author.id").alias("user_id")).agg(
            avg(length(col("text"))).alias("avg_text_length")
        )
        feat_helpful_votes = filtered_reviews_user.groupBy(col("author.id").alias("user_id")).agg(
            avg(col("author.num_helpful_votes")).alias("avg_helpful_votes")
        )

        user_features = feat_review_count
        for feat in [feat_text_length, feat_helpful_votes]:
            user_features = safe_left_join(user_features, feat, join_col="user_id")

        print("[User] After join count:", user_features.count())

        avg_text_length_mean = safe_mean(user_features, "avg_text_length")
        avg_helpful_votes_mean = safe_mean(user_features, "avg_helpful_votes")

        fill_user = {
            "review_count": 0,
            "avg_text_length": avg_text_length_mean,
            "avg_helpful_votes": avg_helpful_votes_mean
        }
        user_features = user_features.fillna(fill_user)

        assembler_user = VectorAssembler(inputCols=list(fill_user.keys()), outputCol="user_vector")
        assembled_user = assembler_user.transform(user_features)

        if assembled_user.count() > 0:
            scaler_user = StandardScaler(inputCol="user_vector", outputCol="scaled_user", withMean=True, withStd=True)
            scaled_user = scaler_user.fit(assembled_user).transform(assembled_user)

            kmeans_user = KMeans(featuresCol="scaled_user", predictionCol="cluster", k=3, seed=42)
            kmeans_model_user = kmeans_user.fit(scaled_user)
            clustered_user = kmeans_model_user.transform(scaled_user)

            centers_user = kmeans_model_user.clusterCenters()

            def compute_distance(vector, cluster):
                center = centers_user[cluster]
                return float(sum([(vector[i]-center[i])**2 for i in range(len(center))])**0.5)

            distance_udf_user = udf(compute_distance, DoubleType())
            clustered_user = clustered_user.withColumn("distance_to_center", distance_udf_user(col("scaled_user"), col("cluster")))

            avg_distance_user = clustered_user.select(mean("distance_to_center")).first()[0]

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_path_user = f"hdfs://localhost:9000/user/fizz/models/kmeans_user_{timestamp}"
            metadata_user = {
                'row_key': f'task_d_user_{timestamp}',
                'info:task_name': 'Task D - User Anomaly (KMeans Spark)',
                'info:model_name': 'KMeans_User',
                'info:run_timestamp': timestamp,
                'metrics:training_time_seconds': "Spark inline",
                'metrics:evaluation_score': str(round(avg_distance_user, 4)),
                'metrics:score_type': "ClusterDistance",
                'artifact:model_hdfs_path': model_path_user
            }
            model_metadata_list.append(metadata_user)

        # =================== 评论层特征工程 ===================
        filtered_reviews_review = reviews.filter(col("text").isNotNull())

        feat_text_length_r = filtered_reviews_review.select(
            col("id").alias("review_id"),
            length(col("text")).alias("text_length")
        )
        feat_word_count_r = filtered_reviews_review.select(
            col("id").alias("review_id"),
            size(split(col("text"), r"\\s+")).alias("word_count")
        )

        review_features = feat_text_length_r
        for feat in [feat_word_count_r]:
            review_features = safe_left_join(review_features, feat, join_col="review_id")

        print("[Review] After join count:", review_features.count())

        text_length_mean = safe_mean(review_features, "text_length")
        word_count_mean = safe_mean(review_features, "word_count")

        fill_review = {
            "text_length": text_length_mean,
            "word_count": word_count_mean
        }
        review_features = review_features.fillna(fill_review)

        assembler_review = VectorAssembler(inputCols=list(fill_review.keys()), outputCol="review_vector")
        assembled_review = assembler_review.transform(review_features)

        if assembled_review.count() > 0:
            scaler_review = StandardScaler(inputCol="review_vector", outputCol="scaled_review", withMean=True, withStd=True)
            scaled_review = scaler_review.fit(assembled_review).transform(assembled_review)

            kmeans_review = KMeans(featuresCol="scaled_review", predictionCol="cluster", k=3, seed=42)
            kmeans_model_review = kmeans_review.fit(scaled_review)
            clustered_review = kmeans_model_review.transform(scaled_review)

            centers_review = kmeans_model_review.clusterCenters()

            def compute_distance_r(vector, cluster):
                center = centers_review[cluster]
                return float(sum([(vector[i]-center[i])**2 for i in range(len(center))])**0.5)

            distance_udf_review = udf(compute_distance_r, DoubleType())
            clustered_review = clustered_review.withColumn("distance_to_center", distance_udf_review(col("scaled_review"), col("cluster")))

            avg_distance_review = clustered_review.select(mean("distance_to_center")).first()[0]

            model_path_review = f"hdfs://localhost:9000/user/fizz/models/kmeans_review_{timestamp}"
            metadata_review = {
                'row_key': f'task_d_review_{timestamp}',
                'info:task_name': 'Task D - Review Anomaly (KMeans Spark)',
                'info:model_name': 'KMeans_Review',
                'info:run_timestamp': timestamp,
                'metrics:training_time_seconds': "Spark inline",
                'metrics:evaluation_score': str(round(avg_distance_review, 4)),
                'metrics:score_type': "ClusterDistance",
                'artifact:model_hdfs_path': model_path_review
            }
            model_metadata_list.append(metadata_review)

        # =================== 写入 HBase ===================
        connection = get_hbase_connection()
        if connection:
            try:
                table = connection.table('model_registry')
                with table.batch() as b:
                    for metadata in model_metadata_list:
                        row_key = metadata.pop('row_key')
                        data_to_write = {k.encode('utf-8'): str(v).encode('utf-8') for k, v in metadata.items()}
                        b.put(row_key.encode('utf-8'), data_to_write)
            finally:
                connection.close()

    finally:
        if 'spark' in locals():
            spark.stop()
