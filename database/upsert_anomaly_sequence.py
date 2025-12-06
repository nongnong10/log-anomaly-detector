def batch_upsert_anomaly_sequences(connection, block_data):
    """
    Batch upsert multiple anomaly_sequence records for better performance

    Args:
        connection: Database connection
        block_data: List of tuples (block_id, label)
    """
    if not block_data:
        return True

    try:
        cursor = connection.cursor()

        # Batch upsert query
        query = """
                INSERT INTO anomaly_sequence (
                    block_id,
                    label
                )
                VALUES (
                    %s, %s
                )
                ON CONFLICT (block_id)
                DO UPDATE SET label = EXCLUDED.label;
                """

        # Prepare data with labels
        prepared_data = [
            (block_id, label)
            for block_id, label in block_data
        ]

        # Execute batch upsert
        cursor.executemany(query, prepared_data)
        connection.commit()

        affected_rows = cursor.rowcount
        cursor.close()

        print(f"Batch upserted {affected_rows} anomaly_sequence records")
        return True

    except Exception as e:
        print(f"Failed to batch upsert anomaly_sequence: {e}")
        try:
            connection.rollback()
        except:
            pass
        return False