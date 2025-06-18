# test_app.py

import json
from flask import Flask, render_template_string

# 导入我们项目中的HBase连接工具
# 为了让这个脚本能找到utils模块，请确保您在项目的根目录下运行它
# (例如 /home/fizz/WQD7007GP/Runcheng)
from utils.hbase_connector import get_hbase_connection

# 初始化Flask应用
app = Flask(__name__)

# --- HTML 模板 ---
# 为了方便，我们直接将HTML代码写在Python脚本里
# 它会自动生成一个美观的表格来展示HBase中的数据
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WQD7007GP - 模型评估中心</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background-color: #f8f9fa; color: #212529; }
        .container { max-width: 90%; margin: 2rem auto; padding: 2rem; background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #343a40; text-align: center; }
        table { width: 100%; border-collapse: collapse; margin-top: 2rem; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #dee2e6; }
        th { background-color: #e9ecef; font-weight: 600; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #e2e6ea; }
        .error { color: #dc3545; font-weight: bold; }
        .success { color: #28a745; font-weight: bold; }
        .code { background-color: #e9ecef; padding: 2px 6px; border-radius: 4px; font-family: "Courier New", Courier, monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>WQD7007GP - 模型注册与评估中心</h1>
        {% if error %}
            <p class="error">错误: {{ error }}</p>
        {% else %}
            <p class="success">成功从HBase加载了 {{ data|length }} 条模型实验记录。</p>
            <table>
                <thead>
                    <tr>
                        <th>实验ID (Row Key)</th>
                        <th>任务名称</th>
                        <th>模型名称</th>
                        <th>评估指标</th>
                        <th>评估分数</th>
                        <th>训练时长 (秒)</th>
                        <th>模型HDFS路径</th>
                    </tr>
                </thead>
                <tbody>
                    {% for item in data %}
                    <tr>
                        <td><span class="code">{{ item.row_key }}</span></td>
                        <td>{{ item.info_task_name }}</td>
                        <td>{{ item.info_model_name }}</td>
                        <td>{{ item.metrics_score_type }}</td>
                        <td><b>{{ item.metrics_evaluation_score }}</b></td>
                        <td>{{ item.metrics_training_time_seconds }}</td>
                        <td><span class="code">{{ item.artifact_model_hdfs_path }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% endif %}
    </div>
</body>
</html>
"""

# 定义网页的根路由 ("/")
@app.route('/')
def index():
    """
    当用户访问主页时，此函数会被调用。
    它会连接HBase，获取数据，并渲染HTML页面。
    """
    model_data = []
    error_message = None

    # 1. 连接HBase
    connection = get_hbase_connection()

    if not connection:
        error_message = "无法连接到HBase。请确认HBase和Thrift服务正在后台运行。"
    else:
        try:
            # 2. 获取 'model_registry' 表
            table = connection.table('model_registry')

            # 3. 扫描全表数据
            for key, data in table.scan():
                # 将HBase返回的bytes数据解码成字符串
                processed_item = {
                    'row_key': key.decode('utf-8'),
                    # 使用.get()方法以避免因缺少某个字段而报错
                    'info_task_name': data.get(b'info:task_name', b'N/A').decode('utf-8'),
                    'info_model_name': data.get(b'info:model_name', b'N/A').decode('utf-8'),
                    'metrics_score_type': data.get(b'metrics:score_type', b'N/A').decode('utf-8'),
                    'metrics_evaluation_score': data.get(b'metrics:evaluation_score', b'N/A').decode('utf-8'),
                    'metrics_training_time_seconds': data.get(b'metrics:training_time_seconds', b'N/A').decode('utf-8'),
                    'artifact_model_hdfs_path': data.get(b'artifact:model_hdfs_path', b'N/A').decode('utf-8'),
                }
                model_data.append(processed_item)
            
            # 按评估分数排序，方便对比
            model_data.sort(key=lambda x: x['metrics_evaluation_score'], reverse=True)

        except Exception as e:
            error_message = f"从HBase读取数据时发生错误: {e}"
        finally:
            # 4. 关闭连接
            connection.close()

    # 5. 渲染HTML页面
    # 将我们处理好的数据(model_data)和任何错误信息(error_message)传递给HTML模板
    return render_template_string(HTML_TEMPLATE, data=model_data, error=error_message)

# --- Flask应用启动配置 ---
if __name__ == '__main__':
    # 关键：设置 host='0.0.0.0' 来让应用监听所有网络接口，
    # 这样你的队友才能通过ZeroTier的IP地址访问到它。
    # port=5000 是Flask的默认端口。
    app.run(host='0.0.0.0', port=5000, debug=True)


