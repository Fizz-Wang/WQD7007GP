#!/bin/bash
# ==============================================================================
#  Official All-in-One Launcher for WQD7007GP Project
#
#  Features:
#  1. Automatically checks and starts background services (Hive Metastore, HBase Thrift).
#  2. Sets up the correct Python and Java library environment.
#  3. Submits and runs the specified Spark analysis task.
#
#  Usage:
#  ./run_spark.sh <task_number>
#  Example: ./run_spark.sh 1
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- [Part 1/3] Checking background service status... ---
echo "--- [Part 1/3] Checking background service status... ---"

# Check for Hive Metastore service (process name: RunJar)
# jps lists Java processes, grep finds a specific process
if ! jps | grep -q "RunJar"; then
    echo "Hive Metastore service is not running, starting it in the background..."
    nohup hive --service metastore > ~/hive_metastore.log 2>&1 &
    # Wait a few seconds to ensure the service has enough time to initialize
    sleep 10
    echo "Hive Metastore has been started."
else
    echo "Hive Metastore is already running."
fi

# Check for HBase Thrift service (process name: ThriftServer)
if ! jps | grep -q "ThriftServer"; then
    echo "HBase Thrift service is not running, starting it in the background..."
    nohup hbase-daemon.sh start thrift > ~/hbase_thrift.log 2>&1 &
    # Again, wait for the service to initialize
    sleep 10
    echo "HBase Thrift service has been started."
else
    echo "HBase Thrift service is already running."
fi

echo "--- [Part 1/3] Background service check complete. ---"
echo ""


# --- [Part 2/3] Configuring Spark execution environment... ---
echo "--- [Part 2/3] Configuring Spark execution environment... ---"

# Check if a task number was provided
if [ -z "$1" ]; then
    echo "Error: Please provide at least one task number."
    echo "Usage: ./run_spark.sh <task_number_1> [task_number_2] ..."
    exit 1
fi

# Critical Configuration (Paths for Python virtual environment and dependency JARs)
VENV_PYTHONPATH="$HOME/.pyenv/versions/pyspark-3.7-env/lib/python3.7/site-packages"
HIVE_JSON_SERDE_JAR="/home/fizz/hive/hcatalog/share/hcatalog/hive-hcatalog-core-1.2.2.jar"

# Set the PYTHONPATH environment variable
export PYTHONPATH="$VENV_PYTHONPATH"

echo "Python library path has been set to: $PYTHONPATH"
echo "Including JAR file: $HIVE_JSON_SERDE_JAR"
echo "--- [Part 2/3] Environment configuration complete. ---"
echo ""


# --- [Part 3/3] Submitting Spark task... ---
echo "--- [Part 3/3] Submitting Spark task... ---"

# Execute the spark-submit command with all necessary configurations
spark-submit \
  --jars "$HIVE_JSON_SERDE_JAR" \
  --conf "spark.driver.extraClassPath=$HIVE_JSON_SERDE_JAR" \
  --conf "spark.driver.extraPythonPath=$PYTHONPATH" \
  --conf "spark.executor.extraPythonPath=$PYTHONPATH" \
  main.py "$@"

echo "--- [Part 3/3] Spark task submission complete. ---"

