# utils/hbase_connector.py

import happybase

# HBase server host. Modify if your HBase is not on localhost.
HBASE_HOST = 'localhost'

def get_hbase_connection():
    """
    Establishes and returns a connection to the HBase Thrift server.
    This function is used by tasks to get a connection object for writing data.
    """
    try:
        # Create a connection to the HBase server.
        # It's important to handle potential connection errors.
        connection = happybase.Connection(HBASE_HOST)
        
        # The connection is opened on demand by happybase, but we can open it
        # explicitly to verify that the connection is good.
        connection.open()
        
        print("Successfully connected to HBase.")
        return connection
        
    except Exception as e:
        # If the connection fails, print an error message and return None.
        print(f"Failed to connect to HBase at '{HBASE_HOST}': {e}")
        return None

