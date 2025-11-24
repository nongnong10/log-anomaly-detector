def upsert_log_block(connection, block_id, event_sequence, has_data, anomaly_score):
    """Insert or update log_block record using existing connection"""
    try:
        cursor = connection.cursor()

        # Upsert query
        query = """
        INSERT INTO log_block (
            block_id,
            event_sequence,
            has_data,
            anomaly_score,
            created_at,
            updated_at
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            NOW(),
            NOW()
        )
        ON CONFLICT (block_id)
        DO UPDATE SET
            event_sequence = EXCLUDED.event_sequence,
            has_data = EXCLUDED.has_data,
            anomaly_score = EXCLUDED.anomaly_score,
            updated_at = NOW();
        """

        # Execute query with rounded anomaly_score for consistency
        cursor.execute(query, (block_id, event_sequence, has_data, round(anomaly_score, 6)))
        connection.commit()

        cursor.close()
        return True

    except Exception as e:
        print(f"Failed to upsert log_block for {block_id}: {e}")
        try:
            connection.rollback()
        except:
            pass
        return False

def batch_upsert_log_blocks(connection, block_data):
    """
    Batch upsert multiple log_block records for better performance

    Args:
        connection: Database connection
        block_data: List of tuples (block_id, event_sequence, has_data, anomaly_score)
    """
    if not block_data:
        return True

    try:
        cursor = connection.cursor()

        # Batch upsert query
        query = """
        INSERT INTO log_block (
            block_id,
            event_sequence,
            has_data,
            anomaly_score,
            created_at,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, NOW(), NOW()
        )
        ON CONFLICT (block_id)
        DO UPDATE SET
            event_sequence = EXCLUDED.event_sequence,
            has_data = EXCLUDED.has_data,
            anomaly_score = EXCLUDED.anomaly_score,
            updated_at = NOW();
        """

        # Prepare data with rounded anomaly scores
        prepared_data = [
            (block_id, event_sequence, has_data, round(anomaly_score, 6))
            for block_id, event_sequence, has_data, anomaly_score in block_data
        ]

        # Execute batch upsert
        cursor.executemany(query, prepared_data)
        connection.commit()

        affected_rows = cursor.rowcount
        cursor.close()

        print(f"Batch upserted {affected_rows} log_block records")
        return True

    except Exception as e:
        print(f"Failed to batch upsert log_blocks: {e}")
        try:
            connection.rollback()
        except:
            pass
        return False
