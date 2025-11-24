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

        # Execute query
        cursor.execute(query, (block_id, event_sequence, has_data, anomaly_score))
        connection.commit()

        cursor.close()
        return True

    except Exception as e:
        print(f"Failed to upsert log_block: {e}")
        connection.rollback()
        return False
