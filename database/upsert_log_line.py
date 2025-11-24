def upsert_log_lines(connection, block_id, log_lines):
    """
    Delete existing log lines for block_id and batch insert new ones using existing connection

    Args:
        connection: Database connection
        block_id: Block ID to process
        log_lines: List of dictionaries with keys: pid, level, component, content, event_id
    """
    if not log_lines:
        print(f"No log lines to process for block {block_id}")
        return True

    try:
        cursor = connection.cursor()

        # Begin transaction
        cursor.execute("BEGIN;")

        # Step 1: Delete existing log lines for block_id
        delete_query = """
        DELETE FROM log_line
        WHERE block_id = %s;
        """
        cursor.execute(delete_query, (block_id,))
        deleted_count = cursor.rowcount

        # Step 2: Batch insert new log lines
        insert_query = """
        INSERT INTO log_line (
            pid,
            level,
            component,
            content,
            event_id,
            block_id,
            created_at,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, NOW(), NOW()
        );
        """

        # Prepare data for batch insert with validation
        insert_data = []
        for log_line in log_lines:
            # Handle potential None values and convert appropriately
            pid = log_line.get('pid')
            if pid is not None:
                try:
                    pid = int(pid)
                except (ValueError, TypeError):
                    pid = None

            insert_data.append((
                pid,
                log_line.get('level', 'INFO'),
                log_line.get('component', 'UNKNOWN'),
                log_line.get('content', ''),
                log_line.get('event_id', ''),
                block_id
            ))

        # Execute batch insert
        cursor.executemany(insert_query, insert_data)
        inserted_count = cursor.rowcount
        print(f"Block {block_id}: deleted {deleted_count}, inserted {inserted_count} log lines")

        # Commit transaction
        cursor.execute("COMMIT;")
        cursor.close()
        return True

    except Exception as e:
        print(f"Failed to upsert log_lines for {block_id}: {e}")
        try:
            cursor.execute("ROLLBACK;")
            connection.rollback()
        except:
            pass
        return False