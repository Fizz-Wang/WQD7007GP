#!/usr/bin/env python3
import happybase
import threading
import time
import random
import argparse

def load_keys(path):
    """从 keys.txt 里读取所有 row key"""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def worker(table_name, keys, duration, counter, lock, host, port):
    """压测线程函数：在 duration 秒内不停随机读 keys"""
    conn = happybase.Connection(host=host, port=port)
    table = conn.table(table_name)
    end_time = time.time() + duration
    local_count = 0
    while time.time() < end_time:
        key = random.choice(keys)
        _ = table.row(key.encode('utf-8'))
        local_count += 1
    conn.close()
    with lock:
        counter[0] += local_count

def main():
    p = argparse.ArgumentParser(description="HBase QPS 压测脚本")
    p.add_argument('--table',   default='model_registry', help='HBase 表名')
    p.add_argument('--keys',    default='keys.txt',        help='Row key 文件')
    p.add_argument('--host',    default='localhost',        help='HBase Thrift Host')
    p.add_argument('--port',    type=int, default=9090,     help='HBase Thrift Port')
    p.add_argument('--threads', type=int, default=20,       help='并发线程数')
    p.add_argument('--duration',type=int, default=60,       help='压测时长（秒）')
    args = p.parse_args()

    keys = load_keys(args.keys)
    if not keys:
        print("没有加载到任何 key，请检查 keys.txt")
        return

    print(f"Loaded {len(keys)} keys, starting {args.threads} threads for {args.duration}s...")

    counter = [0]
    lock = threading.Lock()
    threads = []

    # 启动线程
    for i in range(args.threads):
        t = threading.Thread(
            target=worker,
            args=(
                args.table,
                keys,
                args.duration,
                counter,
                lock,
                args.host,
                args.port
            )
        )
        t.start()
        threads.append(t)

    # 等待所有线程结束
    start_time = time.time()
    for t in threads:
        t.join()
    elapsed = time.time() - start_time

    total_q = counter[0]
    qps = total_q / elapsed if elapsed > 0 else 0.0

    print(f"\n总查询数: {total_q}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"平均 QPS: {qps:.2f}")

if __name__ == '__main__':
    main()

