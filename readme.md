### WQD7007 大数据项目：TripAdvisor酒店评论分析平台 (完整版指南)

#### **项目概览**

本项目旨在利用大数据技术栈，对TripAdvisor酒店评论数据集进行深度分析。我们所有团队成员将在一个**共享的、由组长统一配置和维护的中央服务器（虚拟机）**上进行开发。

本项目将重点应用 **Apache Spark MLlib** 库，探索和解决四种机器学习问题：聚类、分类、情感分析和异常检测。所有分析任务、模型保存、以及最终成果的注册，都将遵循一套标准化的流程。

### **第一步：服务器初始环境配置 (由组长完成，供团队参考)**

本阶段的所有操作都已由组长在中央服务器上一次性完成，团队成员**无需操作**，但理解这些配置有助于更好地进行开发。

- **1.1 核心组件安装**:
  - **Java**: `openjdk-8-jdk`
  - **Hadoop**: `2.7.7`
  - **Spark**: `2.4.8` (for Hadoop 2.7)
  - **Hive**: `1.2.2`
  - **HBase**: `1.4.13`
  - **Scala**: `2.11.12`
  - 所有组件均安装在 `/usr/local/` 目录下，并设置了相应的环境变量 (`$HADOOP_HOME`, `$SPARK_HOME` 等)。
- **1.2 Python环境校准**:
  - 使用`pyenv`工具安装了**Python 3.7.17**，以确保与Spark 2.4.8的完全兼容。
  - 创建了一个名为`pyspark-3.7-env`的专用虚拟环境。
- **1.3 核心服务配置 (关键)**:
  - **Hadoop伪分布式配置**: 已修改`core-site.xml`, `hdfs-site.xml`等文件，使Hadoop能在单机上以分布式模式运行。
  - **Hive Metastore配置**: 已在`$HIVE_HOME/conf/`目录下创建`hive-site.xml`，并配置`hive.metastore.uris`为`thrift://localhost:9083`，使其作为一个独立服务运行。
  - **HBase on HDFS配置**: 已在`$HBASE_HOME/conf/`目录下修改`hbase-site.xml`，将`hbase.rootdir`指向`hdfs://localhost:9000/hbase`，确保HBase数据存储在HDFS上。
  - **Spark连接Hive配置**: 已将`hive-site.xml`文件从Hive配置目录复制到Spark的配置目录，让Spark能自动找到Hive的元数据。
- **1.4 后台服务启动**:
  - 所有必需的后台服务已通过`nohup`命令在后台永久运行，团队成员登录后无需手动启动。
    - Hadoop HDFS (`NameNode`, `DataNode`)
    - HBase Master & RegionServer (`HMaster`, `HRegionServer`)
    - Hive Metastore (`RunJar` 进程)
    - HBase Thrift Server (`ThriftServer` 进程)

### **第二步：数据准备工作 (由组长完成，供团队参考)**

- **2.1 数据上传至HDFS**:
  - 原始JSON数据集已上传至HDFS的`/user/fizz/project_data/`目录。
- **2.2 在Hive中创建数据表**:
  - 组长已登录Hive，并执行`ADD JAR`命令加载了处理JSON所需的`hive-hcatalog-core-*.jar`。
  - 执行了SQL语句，成功创建了名为`reviews`的外部表。
- **2.3 在HBase中创建模型注册表**:
  - 组长已登录HBase Shell，并执行了`create 'model_registry', 'info', 'metrics', 'artifact'`命令，成功创建了用于存储模型实验结果的表。

**一句话总结：服务器已经是一个准备就绪的、所有数据和表都已备好的“大数据实验室”，大家可以随时登录开始做实验。**

### **第三步：远程连接与团队协作 (SSH & Git) - (所有成员必读)**

这是**每一位团队成员都需要掌握的核心工作流程**。我们通过SSH安全地连接到服务器，然后使用专业化的Git流程进行代码的版本控制与同步。

#### **3.1 远程连接服务器 (基于ZeroTier)**

本指南将引导您如何通过ZeroTier安全地远程连接到我们的开发服务器。

1. **获取网络ID**: 在开始前，请注意16位的“网络ID (Network ID)”：`17d709436c084884`

2. **在您的电脑上安装ZeroTier客户端**:

   - 请打开浏览器，访问ZeroTier官方下载页面：https://www.zerotier.com/download/
   - 根据您的操作系统（Windows/Mac/Linux），下载并安装对应的版本。

3. **加入指定的虚拟网络**:

   - 安装完成后，找到ZeroTier的“Join New Network...”选项。
   - 在弹出的窗口中输入上面获取的16位网络ID，然后点击“Join”。

4. **【非常重要】通知管理员授权**:

   - 完成上一步后，请**立即通知管理员（Runcheng Wang）**，他需要在管理后台授权您的设备加入网络。否则，您的连接将无法成功。

5. **使用SSH正式登录**:

   - 等待管理员告知您“已授权”后，打开您的终端（或CMD/PowerShell）。

   - 输入以下命令并按回车。系统会提示您输入`fizz`用户的密码。

     ```
     ssh fizz@10.147.17.31
     ```

#### **3.2 Git团队协作流程 (重点！)**

成功登录服务器后，我们将遵循一套包含**身份配置、分支管理和合并请求 (Pull Request)** 的专业Git流程。

##### **3.2.1 首次Git配置 (每个成员仅需一次)**

虽然我们都使用`fizz`用户登录服务器，但为了在GitHub上明确区分每个人的代码贡献，**每个人都必须在自己克隆的仓库里，配置自己的Git身份**。

1. **进入你自己的工作区**: `cd /home/fizz/WQD7007GP/<你的文件夹名>` (例如: `cd /home/fizz/WQD7007GP/Runcheng`)

2. **设置你的Git用户名和邮箱**: (请确保这与你自己的GitHub账户一致)

   ```
   git init
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   ```

   这个配置是**针对当前仓库**的，不会影响到其他人的工作区。

##### **3.2.2 GitHub推送权限设置 (由组长和组员配合完成)**

为了让所有成员都能向中央仓库推送代码，我们需要进行权限设置。

1. **组长操作 (在GitHub网站上)**:
   - 进入项目仓库页面 `https://github.com/Fizz-Wang/WQD7007GP`。
   - 点击 `Settings` -> `Collaborators and teams` -> `Add people`。
   - 通过GitHub用户名或邮箱，邀请每一位团队成员成为“协作者 (Collaborator)”。
2. **组员操作**:
   - 检查你的邮箱，接受来自GitHub的邀请。
3. **组员操作 (在自己的GitHub网站上)**:
   - 为了安全，我们不直接使用密码，而是使用**个人访问令牌 (Personal Access Token - PAT)**。
   - 访问你的GitHub `Settings` -> `Developer settings` -> `Personal access tokens` -> `Tokens (classic)`。
   - 点击 `Generate new token`，给它一个描述（如 "WQD7007 Project Token"），勾选 `repo` 权限范围，然后生成令牌。
   - **立即复制并保存好这个令牌**，它只会出现一次！

##### **3.2.3 日常开发循环 (Branch & Pull Request模型)**

这是我们日常工作的核心，它确保了`main`分支的稳定和代码的可审查性。

1. **进入目录**: `cd /home/fizz/WQD7007GP/<你的文件夹名>`

2. **激活环境**: `pyenv local pyspark-3.7-env` (通常进入目录会自动激活)

3. **同步主分支**: 在开始新功能开发前，务必确保你的本地`main`分支是最新版本。

   ```
   git checkout main
   git pull origin main
   ```

4. **创建新分支**: 为你的新任务创建一个独立的分支。分支命名要清晰，例如 `feature/task-a-kmeans` 或 `fix/hbase-connector`。

   ```
   git checkout -b <你的新分支名>
   ```

5. **编写与提交代码**: 在这个新分支上，使用`nano`等编辑器进行开发。完成后，将更改提交到**你自己的这个分支**上。

   ```
   git add .
   git commit -m "feat(TaskA): 完成K-Means模型训练与评估"
   ```

6. **推送分支到GitHub**: 将你的本地分支推送到远程仓库。

   ```
   git push origin <你的新分支名>
   ```

   当你第一次推送时，Git会要求你输入GitHub的**用户名**和**密码**。请在密码处**粘贴你生成的个人访问令牌 (PAT)**。

7. **创建合并请求 (Pull Request)**:

   - 打开你的浏览器，进入项目GitHub页面。你应该会看到一个黄色的提示条，提示你刚刚推送了一个新分支。
   - 点击`Compare & pull request`按钮。
   - 填写标题和描述，说明你完成了什么功能，然后点击`Create pull request`。

8. **代码审查与合并 (由组长完成)**:

   - 组长（或其他指定成员）会审查你提交的代码。
   - 确认无误后，组长会在GitHub上点击`Merge pull request`按钮，将你的代码安全地合并到`main`分支中。

### **第四步：核心分析与成果标准化**

为了保证团队协作的规范性，我们的项目采用了一个统一的代码结构。请在你的个人工作区中查看，并理解每个部分的作用。

- **项目代码结构解析**:

  - `tasks/` 目录: 这是我们所有分析任务的核心代码存放区。每个小组都在这里编写自己的脚本，例如`task_a_clustering.py`, `task_b_classification.py`等。
  - `utils/` 目录: 存放所有小组都可以共用的“工具”模块。比如`spark_session.py`（用来创建Spark会话）和`hbase_connector.py`（用来连接HBase），这能避免我们重复造轮子。
  - `main.py` 文件: 这是整个项目的“总调度中心”。它会根据我们输入的任务编号，去调用`tasks/`目录中对应的任务脚本。
  - `run_spark.sh` 文件: 这是我们项目的**官方“点火”脚本**。由于直接运行`spark-submit`需要很多复杂的配置，我们把所有配置都封装在了这个脚本里。**所有人都必须通过这个脚本来运行任务**。

- **如何运行你的分析任务**:

  1. 确保你当前在你的个人工作区根目录下（例如`/home/fizz/WQD7007GP/Runcheng`）。

  2. 赋予`run_spark.sh`执行权限（每个成员只需要做一次）：

     ```
     chmod +x run_spark.sh
     ```

  3. 根据`main.py`中定义的任务列表，输入对应的编号来运行你的任务。

     ```
     # 语法: ./run_spark.sh <你的任务编号>
     
     # 例如，A组同学运行自己的聚类任务：
     ./run_spark.sh 1
     
     # B组同学运行自己的分类任务：
     ./run_spark.sh 2
     ```

  这个脚本会自动检查并启动后台服务，然后用正确的配置来提交你的Spark作业。

#### **通用代码结构模板 (适用于所有小组)**

为了帮助所有小组理解并遵循标准化的流程，我们提供一个通用的脚本模板。**每个小组都可以复制这个结构，并替换其中标记为`!!!`的、与自己任务相关的核心逻辑部分**。

```
# tasks/task_x_template.py

# 导入所有必要的通用库
import time
from datetime import datetime

# =====================================================================
# !!! 导入您任务所需的特定库 !!!
# =====================================================================
# ... (例如: from pyspark.ml.clustering import KMeans) ...

from pyspark.ml import Pipeline
from utils.spark_session import get_spark_session
from utils.hbase_connector import get_hbase_connection

def run_task_x(): # <-- 请将'x'替换成您的小组代号
    """一个标准化的任务执行函数，遵循五个核心步骤。"""
    
    # 1. 初始化
    spark = get_spark_session("Task X - Your Task Description")
    model_metadata_list = []

    try:
        # 2. 数据加载与特征工程
        raw_df = spark.sql("SELECT * FROM reviews")
        # !!! 在这里编写您小组专属的特征工程代码 !!!
        
        # 3. 模型训练与评估
        # !!! 在这里定义您小组要使用的模型和评估器 !!!
        start_time = time.time()
        # ... 训练与评估 ...
        training_time = time.time() - start_time
        evaluation_score = 0.0 # <-- 替换为真实的评估分数
        metric_name = "Your_Metric_Name" # <-- 替换为真实的评估指标名

        # 4. 模型与元数据持久化 (核心)
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = "Your_Model_Name" # <-- 替换为您使用的模型名称
        model_path = f"hdfs://localhost:9000/user/fizz/models/{model_name.lower()}_{timestamp_str}"
        # trained_model.save(model_path)

        # !!! 准备结构化的元数据字典 (所有小组都必须遵循这个结构) !!!
        model_metadata = {
            'row_key': f'task_x_{model_name.lower()}_{timestamp_str}', # <-- 修改任务代号
            'info:task_name': 'Task X - Your Task Description', # <-- 修改任务描述
            'info:model_name': model_name,
            'info:run_timestamp': timestamp_str,
            'metrics:training_time_seconds': str(round(training_time, 2)),
            'metrics:evaluation_score': str(round(evaluation_score, 4)),
            'metrics:score_type': metric_name,
            'artifact:model_hdfs_path': model_path
        }
        model_metadata_list.append(model_metadata)

        # 将元数据写入HBase
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
        # 5. 结束
        if 'spark' in locals():
            spark.stop()
```

### **第五步：成果检验 (Verifying Results)**

在您运行完 `./run_spark.sh {你的数字}`（./run_spark.sh 1） 脚本后，您需要检查两处地方，以确认您的成果已成功保存。

#### **5.1 查看HBase中的模型元数据**

1. **启动HBase Shell**: `hbase shell`
2. **扫描全表内容**: `scan 'model_registry'`
   - 您应该能看到刚刚运行任务时生成的、包含模型名称、评估分数、训练时长等信息的“实验记录”。

#### **5.2 查看HDFS中保存的模型**

1. **退出HBase Shell** (输入`exit`并回车)。
2. **列出HDFS中的`models`目录**: `hdfs dfs -ls /user/fizz/models`
   - 您应该能看到一个以您的模型名和时间戳命名的**新目录**，这代表模型文件已成功保存。

### **第六步：成果展示 (Flask)**

所有分析任务完成后，`model_registry`表就成了我们成果的“排行榜”。

- **后端逻辑**: Flask应用将直接连接到HBase，查询`model_registry`表。
- **功能展示**:
  1. **模型排行榜**: 在一个页面上，以表格形式展示所有实验记录，可以根据“评估分数”进行排序，直观地对比所有模型在所有任务上的表现。
  2. **实时预测接口**: 提供一个API，接收新的酒店评论数据。后端根据请求，从HBase找到表现最好的模型路径，用`PipelineModel.load()`加载HDFS上的模型，进行预测，并将结果返回。

这个最终的应用将作为我们整个项目成果的集中体现，将复杂的数据分析流程，转化为直观、有商业价值的信息。
