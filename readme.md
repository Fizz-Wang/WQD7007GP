## WQD7007 Big Data Project: TripAdvisor Hotel Review Analysis Platform (Complete Guide)

### Project Overview

This project aims to leverage a big data technology stack for in-depth analysis of the TripAdvisor hotel review dataset. All our team members will develop on a **shared central server (virtual machine) uniformly configured and maintained by the team leader.**

This project will focus on applying the **Apache Spark MLlib** library to explore and solve four machine learning problems: clustering, classification, sentiment analysis, and anomaly detection. All analysis tasks, model saving, and final result registration will follow a standardized process.

------

### Step One: Server Initial Environment Configuration (Completed by Team Leader, For Team Reference)

All operations in this stage have been completed once by the team leader on the central server. Team members do not need to perform these operations, but understanding these configurations will aid in better development.

#### 1.1 Core Component Installation:

- **Java:** `openjdk-8-jdk`
- **Hadoop:** `2.7.7`
- **Spark:** `2.4.8` (for Hadoop 2.7)
- **Hive:** `1.2.2`
- **HBase:** `1.4.13`
- **Scala:** `2.11.12`

All components are installed under the `/usr/local/` directory, with corresponding environment variables (`$HADOOP_HOME`, `$SPARK_HOME`, etc.) set.

#### 1.2 Python Environment Calibration:

- `Python 3.7.17` was installed using the `pyenv` tool to ensure full compatibility with Spark 2.4.8.
- A dedicated virtual environment named `pyspark-3.7-env` was created.

#### 1.3 Core Service Configuration (Crucial):

- **Hadoop Pseudo-Distributed Configuration:** `core-site.xml`, `hdfs-site.xml`, and other files have been modified to allow Hadoop to run in a distributed mode on a single machine.
- **Hive Metastore Configuration:** `hive-site.xml` has been created in the `$HIVE_HOME/conf/` directory, and `hive.metastore.uris` has been configured to `thrift://localhost:9083`, allowing it to run as an independent service.
- **HBase on HDFS Configuration:** `hbase-site.xml` has been modified in the `$HBASE_HOME/conf/` directory, pointing `hbase.rootdir` to `hdfs://localhost:9000/hbase`, ensuring HBase data is stored on HDFS.
- **Spark Connection to Hive Configuration:** The `hive-site.xml` file has been copied from the Hive configuration directory to Spark's configuration directory, allowing Spark to automatically find Hive's metadata.

#### 1.4 Background Service Startup:

All required background services have been permanently run in the background using the `nohup` command. Team members do not need to manually start them after logging in.

- Hadoop HDFS (`NameNode`, `DataNode`)
- HBase Master & RegionServer (`HMaster`, `HRegionServer`)
- Hive Metastore (`RunJar` process)
- HBase Thrift Server (`ThriftServer` process)

------

### Step Two: Data Preparation (Completed by Team Leader, For Team Reference)

#### 2.1 Data Upload to HDFS:

The raw JSON dataset has been uploaded to the `/user/fizz/project_data/` directory in HDFS.

#### 2.2 Create Data Tables in Hive:

The team leader has logged into Hive and executed the `ADD JAR` command to load the `hive-hcatalog-core-*.jar` required for JSON processing. SQL statements were executed to successfully create an external table named `reviews`.

#### 2.3 Create Model Registry Table in HBase:

The team leader has logged into HBase Shell and executed the `create 'model_registry', 'info', 'metrics', 'artifact'` command to successfully create the table for storing model experiment results.

**In summary:** The server is now a ready-to-use "big data lab" with all data and tables prepared. Everyone can log in and start experimenting at any time.

------

### Step Three: Remote Connection and Team Collaboration (SSH & Git) - (Mandatory for All Members)

This is the core workflow that every team member needs to master. We will securely connect to the server via SSH and then use a professional Git process for code version control and synchronization.

#### 3.1 Remote Server Connection (Based on ZeroTier)

This guide will walk you through securely connecting to our development server via ZeroTier.

**Get the Network ID:** Before starting, please note the 16-digit "Network ID": `17d709436c084884`

**Install the ZeroTier Client on your computer:**

- Open your browser and visit the ZeroTier official download page: `https://www.zerotier.com/download/`
- Download and install the corresponding version based on your operating system (Windows/Mac/Linux).

**Join the specified virtual network:**

- After installation, find the "Join New Network..." option in ZeroTier.
- Enter the 16-digit Network ID obtained above in the pop-up window, then click "Join".

**【Very Important】Notify the administrator for authorization:**

- After completing the previous step, please immediately notify the administrator (Runcheng Wang). They need to authorize your device to join the network from the administration backend. Otherwise, your connection will not be successful.

**Formally log in using SSH:**

- After the administrator informs you "Authorized", open your terminal (or CMD/PowerShell).

Enter the following command and press Enter. You will be prompted to enter the password for the fizz user.

ssh fizz@10.147.17.31

#### 3.2 Git Team Collaboration Workflow (Key!)

After successfully logging into the server, we will follow a professional Git process that includes identity configuration, branch management, and Pull Requests.

#### 3.2.1 Clone your own workspace (Each member needs to do this only once)

This is the first and most critical step in setting up your personal development environment.

**Create and enter the main workspace:**

Bash

```
mkdir -p /home/fizz/WQD7007GP
cd /home/fizz/WQD7007GP
```

**Clone the central repository into your personal folder:** Use the `git clone` command. This command will automatically create a folder for you, download all the code, and set up the connection to the remote repository.

Bash

```
# Replace <your_folder_name> with your own name or group name, e.g., "Runcheng" or "GroupB"
git clone https://github.com/Fizz-Wang/WQD7007GP.git <your_folder_name>
```

Example:

git clone https://github.com/Fizz-Wang/WQD7007GP.git Runcheng

#### 3.2.2 First-time Git Configuration (Complete within your personal workspace)

Enter your newly cloned workspace:

cd <your_folder_name>

\# e.g.: cd Runcheng

**Set your Git username and email** (Please ensure this matches your GitHub account):

Bash

```
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

This configuration is specific to the current repository and will not affect other people's workspaces.

#### 3.2.3 GitHub Push Permissions Setup (Completed by Team Leader and Members Cooperatively)

To allow all members to push code to the central repository, we need to set up permissions.

**Team Leader's Operation (on GitHub website):**

- Go to the project repository page: `https://github.com/Fizz-Wang/WQD7007GP`.
- Click **Settings** -> **Collaborators and teams** -> **Add people**.
- Invite each team member as a "Collaborator" via their GitHub username or email.

**Team Member's Operation:**

- Check your email and accept the invitation from GitHub.

Team Member's Operation (on their own GitHub website):

For security, we will use a Personal Access Token (PAT) instead of directly using a password.

- Visit your GitHub **Settings** -> **Developer settings** -> **Personal access tokens** -> **Tokens (classic)**.
- Click **Generate new token**, give it a description (e.g., "WQD7007 Project Token"), check the **repo** scope, and then generate the token.
- **Immediately copy and save this token; it will only appear once!**

#### 3.2.4 Daily Development Cycle (Branch & Pull Request Model)

This is the core of our daily work, ensuring the stability of the `main` branch and the reviewability of code.

**Enter the directory:** `cd /home/fizz/WQD7007GP/<your_folder_name>`

**Activate environment:** `pyenv local pyspark-3.7-env` (usually, entering the directory will automatically activate it)

**Synchronize main branch:** Before starting new feature development, always ensure your local `main` branch is up-to-date.

Bash

```
git checkout main
git pull origin main
```

Create a new branch: Create an independent branch for your new task. Branch names should be clear, for example, feature/task-a-kmeans or fix/hbase-connector.

git checkout -b <your_new_branch_name>

**Write and commit code:** Develop on this new branch using editors like `nano`. After completion, commit the changes to your own branch.

Bash

```
git add .
git commit -m "feat(TaskA): Complete K-Means model training and evaluation"
```

Push branch to GitHub: Push your local branch to the remote repository.

git push origin <your_new_branch_name>

The first time you push, Git will ask for your GitHub username and password. Please paste your generated Personal Access Token (PAT) in the password field.

**Create a Pull Request:**

- Open your browser and go to the project's GitHub page. You should see a yellow banner prompting you that you just pushed a new branch.
- Click the **Compare & pull request** button.
- Fill in the title and description, explaining what functionality you have completed, then click **Create pull request**.

**Code Review and Merge (Completed by Team Leader):**

- The team leader (or other designated members) will review your submitted code.
- Once confirmed, the team leader will click the **Merge pull request** button on GitHub to safely merge your code into the `main` branch.

------

### Step Four: Core Analysis and Output Standardization

To ensure standardized team collaboration, our project adopts a unified code structure. Please examine it in your personal workspace and understand the role of each part.

**Project Code Structure Analysis:**

- `tasks/` directory: This is where the core code for all our analysis tasks is stored. Each group writes their scripts here, such as `task_a_clustering.py`, `task_b_classification.py`, etc.
- `utils/` directory: Stores "utility" modules that all groups can share. For example, `spark_session.py` (used to create Spark sessions) and `hbase_connector.py` (used to connect to HBase), which prevents us from reinventing the wheel.
- `main.py` file: This is the project's "central dispatcher." It calls the corresponding task scripts in the `tasks/` directory based on the task number we input.
- `run_spark.sh` file: This is our project's official "ignition" script. Since directly running `spark-submit` requires many complex configurations, we have encapsulated all configurations in this script. Everyone must run tasks through this script.

**How to run your analysis tasks:**

- Ensure you are currently in the root of your personal workspace (e.g., `/home/fizz/WQD7007GP/Runcheng`).

- Grant execute permissions to `run_spark.sh` (each member only needs to do this once): `chmod +x run_spark.sh`

- Based on the task list defined in 

  ```
  main.py
  ```

  , enter the corresponding number to run your task.

  Bash

  ```
  # Syntax: ./run_spark.sh <your_task_number>
  
  # For example, Group A members running their clustering task:
  ./run_spark.sh 1
  
  # Group B members running their classification task:
  ./run_spark.sh 2
  ```

  This script will automatically check and start background services, then submit your Spark job with the correct configuration.

**Generic Code Structure Template (Applicable to all groups)**

To help all groups understand and follow the standardized process, we provide a generic script template. Each group can copy this structure and replace the parts marked `!!!` with their task-specific core logic.

Python

```
# tasks/task_x_template.py

# Import all necessary generic libraries
import time
from datetime import datetime

# =====================================================================
# !!! Import specific libraries required for your task !!!
# =====================================================================
# ... (e.g.: from pyspark.ml.clustering import KMeans) ...

from pyspark.ml import Pipeline
from utils.spark_session import get_spark_session
from utils.hbase_connector import get_hbase_connection

def run_task_x(): # <-- Please replace 'x' with your group's identifier
    """A standardized task execution function, following five core steps."""
    
    # 1. Initialization
    spark = get_spark_session("Task X - Your Task Description")
    model_metadata_list = []

    try:
        # 2. Data Loading and Feature Engineering
        raw_df = spark.sql("SELECT * FROM reviews")
        # !!! Write your group's exclusive feature engineering code here !!!
        
        # 3. Model Training and Evaluation
        # !!! Define the model and evaluator your group will use here !!!
        start_time = time.time()
        # ... training and evaluation ...
        training_time = time.time() - start_time
        evaluation_score = 0.0 # <-- Replace with actual evaluation score
        metric_name = "Your_Metric_Name" # <-- Replace with actual evaluation metric name

        # 4. Model and Metadata Persistence (Core)
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = "Your_Model_Name" # <-- Replace with the name of the model you used
        model_path = f"hdfs://localhost:9000/user/fizz/models/{model_name.lower()}_{timestamp_str}"
        # trained_model.save(model_path)

        # !!! Prepare structured metadata dictionary (all groups must follow this structure) !!!
        model_metadata = {
            'row_key': f'task_x_{model_name.lower()}_{timestamp_str}', # <-- Modify task identifier
            'info:task_name': 'Task X - Your Task Description', # <-- Modify task description
            'info:model_name': model_name,
            'info:run_timestamp': timestamp_str,
            'metrics:training_time_seconds': str(round(training_time, 2)),
            'metrics:evaluation_score': str(round(evaluation_score, 4)),
            'metrics:score_type': metric_name,
            'artifact:model_hdfs_path': model_path
        }
        model_metadata_list.append(model_metadata)

        # Write metadata to HBase
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
        # 5. Finalization
        if 'spark' in locals():
            spark.stop()
```

------

### Step Five: Verifying Results

After running the `./run_spark.sh {your_number}` (e.g., `./run_spark.sh 1`) script, you need to check two locations to confirm your results have been successfully saved.

#### 5.1 View Model Metadata in HBase

Start HBase Shell: hbase shell

Scan full table content: scan 'model_registry'

You should see the "experiment records" generated from your recent task run, containing information such as model name, evaluation score, training duration, etc.

#### 5.2 View Saved Models in HDFS

Exit HBase Shell (type exit and press Enter).

List the models directory in HDFS: hdfs dfs -ls /user/fizz/models

You should see a new directory named with your model name and timestamp, indicating that the model file has been successfully saved.

------

### Step Six: Results Display (Flask)

Once all analysis tasks are complete, the `model_registry` table becomes our results "leaderboard."

**Backend Logic:** The Flask application will connect directly to HBase and query the `model_registry` table.

**Functionality Display:**

- **Model Leaderboard:** On one page, display all experiment records in a table format, sortable by "evaluation score," providing an intuitive comparison of all models across all tasks.
- **Real-time Prediction Interface:** Provide an API that receives new hotel review data. The backend will, based on the request, find the best-performing model path from HBase, load the model from HDFS using `PipelineModel.load()`, perform a prediction, and return the result.

This final application will serve as the centralized showcase for our entire project's achievements, transforming complex data analysis processes into intuitive, commercially valuable information.
