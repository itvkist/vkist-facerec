import sqlite3
import pandas as pd

# Database and table details
database_name = "face_reg.db"  # Replace with your database file name
excel_file_name = "exported_timeline_monthly.xlsx"  # Output Excel file name

# Connect to the SQLite database
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

# SQL query to fetch name, date, time, and month
query = """
SELECT 
    p.name AS PersonName, 
    STRFTIME('%d/%m/%Y', DATETIME(CAST(t.timestamp / 1000 AS INTEGER), 'unixepoch', '+7 hours')) AS Date,
    STRFTIME('%H:%M', DATETIME(CAST(t.timestamp / 1000 AS INTEGER), 'unixepoch', '+7 hours')) AS Time,
    STRFTIME('%m-%Y', DATETIME(CAST(t.timestamp / 1000 AS INTEGER), 'unixepoch', '+7 hours')) AS Month
FROM 
    People p
JOIN 
    Timeline t 
ON 
    p.access_key = t.person_access_key;
"""

try:
    # Execute the query and fetch all rows
    cursor.execute(query)
    rows = cursor.fetchall()

    # Create a DataFrame from the query result
    df = pd.DataFrame(rows, columns=["PersonName", "Date", "Time", "Month"])

    # Create a Pandas Excel writer object
    with pd.ExcelWriter(excel_file_name, engine="openpyxl") as writer:
        # Group data by month
        for month, group in df.groupby("Month"):
            # Within each month, group by PersonName and Date to get first and last appearances
            summary = (
                group.groupby(["PersonName", "Date"])
                .agg(
                    FirstAppearance=("Time", "min"),
                    LastAppearance=("Time", "max")
                )
                .reset_index()
            )
            # Save each month's summary to a separate sheet
            summary.to_excel(writer, index=False, sheet_name=month)

    print(f"Data successfully exported to {excel_file_name}, with each month in a separate sheet.")
except sqlite3.Error as e:
    print(f"An error occurred: {e}")
finally:
    # Close the database connection
    conn.close()