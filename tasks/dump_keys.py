#!/usr/bin/env python3
import happybase

TABLE = 'model_registry'
OUT   = 'keys.txt'
LIMIT = 100    # 导出前 N 条

def main():
    # 1. 连接 HBase
    conn = happybase.Connection(host='localhost', port=9090)
    table = conn.table(TABLE)

    # 2. 扫描写文件
    with open(OUT, 'w') as f:
        count = 0
        for key, _ in table.scan(limit=LIMIT):
            f.write(key.decode('utf-8') + '\n')
            count += 1

    conn.close()
    print(f"Dumped {count} keys to {OUT}")

if __name__ == '__main__':
    main()

