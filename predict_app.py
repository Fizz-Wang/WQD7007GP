# predict_script.py

import argparse
import json
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel

def main(model_path, text_to_predict):
    """
    A dedicated Spark script to load a pre-trained model and make a single prediction.
    """
    spark = SparkSession.builder \
        .appName("Prediction Script") \
        .getOrCreate()
        
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the saved Pipeline model from HDFS
        model = PipelineModel.load(model_path)
        
        # Create a one-row DataFrame from the input text
        # The column name 'text' must match the inputCol of the Tokenizer in the saved model
        schema = ["text"]
        data = [(text_to_predict,)]
        df_to_predict = spark.createDataFrame(data, schema)
        
        # Use the model to make a prediction
        prediction_result = model.transform(df_to_predict)
        
        # Extract the prediction result
        # This will depend on the type of model.
        # For clustering (KMeans/GMM), the column is 'cluster_id'
        # For classification, it's 'prediction'
        result_row = prediction_result.first()
        
        output = {}
        if 'cluster_id' in result_row.schema.names:
            output['cluster_id'] = result_row['cluster_id']
        if 'prediction' in result_row.schema.names:
            output['prediction'] = result_row['prediction']
            # Also get the probability for classification models
            if 'probability' in result_row.schema.names:
                # Convert vector to list for JSON serialization
                output['probability'] = result_row['probability'].toArray().tolist()

        # Print the result as a JSON string to standard output.
        # The Flask app will capture this output.
        print(json.dumps(output))

    except Exception as e:
        # Print error to stderr so Flask can capture it
        import sys
        print(f"Prediction failed: {e}", file=sys.stderr)

    finally:
        spark.stop()


if __name__ == "__main__":
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HDFS path to the saved Spark ML model")
    parser.add_argument("--text", required=True, help="The input text to classify or cluster")
    args = parser.parse_args()
    
    main(args.model, args.text)


