from pydantic import BaseModel
class CountAnomalyLogLines(BaseModel):
    block_id: str
    total_anomalous_lines: int
    total_line: int

def count_anomaly_log_lines(connection, block_id):
    """
    Count anomaly log lines for a given block ID
    Args:
        connection: Database connection
        block_id: The block ID to filter log lines
    """
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM log_line
        WHERE block_id = %s AND is_anomaly = TRUE;
        """,
        (block_id,)
    )
    totalAnomalies = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM log_line
        WHERE block_id = %s;
        """,
        (block_id,)
    )
    totalLines= cursor.fetchone()[0]
    cursor.close()

    return CountAnomalyLogLines(
        block_id=block_id,
        total_anomalous_lines=totalAnomalies,
        total_line=totalLines
    )

class LogLineItem(BaseModel):
    line_id: int
    pid: int | None = None
    level: str | None = None
    component: str | None = None
    event_id: str | None = None
    block_id: str | None = None
    content: str | None = None

def get_log_lines(connection, block_id):
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT line_id, created_at, updated_at, pid, level, component, event_id, block_id, content
        FROM log_line
        WHERE block_id = %s
        ORDER BY updated_at DESC, line_id ASC;
        """,
        (block_id,)
    )
    rows = cursor.fetchall()
    log_lines = [
        LogLineItem(
            line_id=r[0],
            pid=r[3],
            level=r[4],
            component=r[5],
            event_id=r[6],
            block_id=r[7],
            content=r[8],
        )
        for r in rows
    ]
    return log_lines
