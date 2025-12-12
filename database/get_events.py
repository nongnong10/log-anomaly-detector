from pydantic import BaseModel
class Event(BaseModel):
    event_id: str
    event_template: str
    occurrences: int


def get_all_event_templates(connection):
    """
    Get all event templates from the database
    Args:
        connection: Database connection
    """
    cursor = connection.cursor()
    # 1. Prepare query
    query = """
            SELECT event_id, event_template, occurrences
            FROM event;
            """
    # 2. Execute query
    cursor.execute(query)
    rows = cursor.fetchall()

    events = [
        Event(
            event_id=r[0],
            event_template=r[1],
            occurrences=r[2],
        )
        for r in rows
    ]
    cursor.close()

    return events