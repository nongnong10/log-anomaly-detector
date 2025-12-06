from fastapi import APIRouter, Query, HTTPException, Request
from pydantic import BaseModel
from typing import List
from datetime import datetime  # added

router = APIRouter()

class LogLineItem(BaseModel):
    line_id: int
    created_at: datetime | None = None  # changed type from str to datetime
    updated_at: datetime | None = None  # changed type from str to datetime
    pid: int | None = None
    level: str | None = None
    component: str | None = None
    event_id: str | None = None
    block_id: str | None = None
    content: str | None = None

class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total: int

class LogLinesResponse(BaseModel):
    log_lines: List[LogLineItem]
    pagination: PaginationMeta

@router.get("/list-log-lines", response_model=LogLinesResponse)
def list_log_lines(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, gt=0, le=1000),
    block_ids: List[str] = Query(default=[]),
):
    conn = getattr(request.app.state, "db_conn", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="DB not connected")
    offset = (page - 1) * page_size

    base_count_sql = "SELECT COUNT(*) FROM log_line"
    base_select_sql = """SELECT line_id, created_at, updated_at, pid, level, component, event_id, block_id, content
                         FROM log_line"""
    params: List = []
    if block_ids:
        placeholders = ",".join(["%s"] * len(block_ids))
        where_clause = f" WHERE block_id IN ({placeholders})"
        base_count_sql += where_clause
        base_select_sql += where_clause
        params.extend(block_ids)
    select_sql = base_select_sql + " ORDER BY updated_at DESC, line_id ASC LIMIT %s OFFSET %s"  # updated ordering
    params_select = params + [page_size, offset]

    try:
        with conn.cursor() as cur:
            # total count
            cur.execute(base_count_sql, params)
            total = cur.fetchone()[0]

            # page rows
            cur.execute(select_sql, params_select)
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    log_lines = [
        LogLineItem(
            line_id=r[0],
            created_at=r[1],  # now datetime
            updated_at=r[2],  # now datetime
            pid=r[3],
            level=r[4],
            component=r[5],
            event_id=r[6],
            block_id=r[7],
            content=r[8],
        )
        for r in rows
    ]
    return LogLinesResponse(
        log_lines=log_lines,
        pagination=PaginationMeta(page=page, page_size=page_size, total=total),
    )
