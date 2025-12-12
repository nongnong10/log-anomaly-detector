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
            event_sequence = COALESCE(EXCLUDED.event_sequence, log_block.event_sequence),
            has_data = COALESCE(EXCLUDED.has_data, log_block.has_data),
            anomaly_score = EXCLUDED.anomaly_score,
            updated_at = NOW();
        """

        rounded_score = round(anomaly_score, 6) if anomaly_score is not None else None
        cursor.execute(query, (block_id, event_sequence, has_data, rounded_score))
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
            event_sequence = COALESCE(EXCLUDED.event_sequence, log_block.event_sequence),
            has_data = COALESCE(EXCLUDED.has_data, log_block.has_data),
            anomaly_score = EXCLUDED.anomaly_score,
            updated_at = NOW();
        """

        prepared_data = []
        for block_id, event_sequence, has_data, anomaly_score in block_data:
            rounded = round(anomaly_score, 6) if anomaly_score is not None else None
            prepared_data.append((block_id, event_sequence, has_data, rounded))

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
