import mysql.connector

# Replace these values with your database connection details
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "07032001",
    "database": "fashionstorewebsite",
}

try:
    # Connect to the MySQL database
    connection = mysql.connector.connect(**db_config)

    if connection.is_connected():
        cursor = connection.cursor()

        # Execute a SQL query
        cursor.execute(
            "SELECT * FROM fashionstorewebsite.product_info_for_ui;")

        # Fetch and print the results
        for row in cursor.fetchall():
            print(row)

except mysql.connector.Error as e:
    print(f"Error: {e}")
finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
