from fastapi import APIRouter, Query, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class LogLineItem(BaseModel):
    line_id: int
    date: str | None = None
    time: str | None = None
    pid: int | None = None
    level: str | None = None
    component: str | None = None
    event_id: str | None = None

class LogBlockItem(BaseModel):
    date: str | None = None
    time: str | None = None
    block_id: str
    is_anomaly: bool
    anomaly_score: float
    log_lines: List[LogLineItem]

class SequencesPagination(BaseModel):
    page: int
    page_size: int
    total_sequences: int
    anomalous_sequences: int
    anomalous_ratio: float

class LogSequencesResponse(BaseModel):
    log_blocks: List[LogBlockItem]
    pagination: SequencesPagination

@router.get("/list-log-sequences", response_model=LogSequencesResponse)
def list_log_sequences(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, gt=0, le=1000),
    block_ids: List[str] = Query(default=[]),
):
    conn = getattr(request.app.state, "db_conn", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="DB not connected")
    offset = (page - 1) * page_size
    # Step 1: page block_ids from log_block (only where has_data = TRUE)
    where_conditions = ["has_data = TRUE"]
    params: List = []
    if block_ids:
        placeholders = ",".join(["%s"] * len(block_ids))
        where_conditions.append(f"block_id IN ({placeholders})")
        params.extend(block_ids)
    where_clause = " WHERE " + " AND ".join(where_conditions)
    total_sql = "SELECT COUNT(*) FROM log_block" + where_clause
    anomalous_count_sql = (
        "SELECT COUNT(*) FROM anomaly_sequence a "
        "JOIN log_block lb ON a.block_id = lb.block_id "
        "WHERE a.label='Anomaly' AND lb.has_data = TRUE"
        + ("" if not block_ids else f" AND a.block_id IN ({','.join(['%s']*len(block_ids))})")
    )
    page_sql = "SELECT block_id FROM log_block" + where_clause + " LIMIT %s OFFSET %s"
    page_params = params + [page_size, offset]
    try:
        with conn.cursor() as cur:
            cur.execute(total_sql, params)
            total_sequences = cur.fetchone()[0]
            cur.execute(anomalous_count_sql, ([] if not block_ids else block_ids))
            anomalous_sequences = cur.fetchone()[0]
            cur.execute(page_sql, page_params)
            selected_block_ids = [r[0] for r in cur.fetchall()]
            # Step 2: labels for paged block_ids
            labels_map: Dict[str, str] = {bid: "Normal" for bid in selected_block_ids}
            if selected_block_ids:
                placeholders_lbl = ",".join(["%s"] * len(selected_block_ids))
                lbl_sql = f"SELECT block_id, label FROM anomaly_sequence WHERE block_id IN ({placeholders_lbl})"
                cur.execute(lbl_sql, selected_block_ids)
                for b_id, label in cur.fetchall():
                    labels_map[b_id] = label
            # Step 3: fetch log_lines for these block_ids
            log_lines_map: Dict[str, List[LogLineItem]] = {bid: [] for bid in selected_block_ids}
            if selected_block_ids:
                placeholders_lines = ",".join(["%s"] * len(selected_block_ids))
                lines_sql = (
                    "SELECT line_id, date, time, pid, level, component, event_id, block_id "
                    f"FROM log_line WHERE block_id IN ({placeholders_lines}) ORDER BY line_id"
                )
                cur.execute(lines_sql, selected_block_ids)
                for r in cur.fetchall():
                    log_lines_map[r[7]].append(
                        LogLineItem(
                            line_id=r[0],
                            date=r[1],
                            time=r[2],
                            pid=r[3],
                            level=r[4],
                            component=r[5],
                            event_id=r[6],
                        )
                    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    log_blocks: List[LogBlockItem] = []
    for bid in selected_block_ids:
        lines = log_lines_map.get(bid, [])
        date_val = lines[0].date if lines else None
        time_val = lines[0].time if lines else None
        is_anomaly = labels_map.get(bid) == "Anomaly"
        anomaly_score = 1.0 if is_anomaly else 0.0
        log_blocks.append(
            LogBlockItem(
                date=date_val,
                time=time_val,
                block_id=bid,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                log_lines=lines,
            )
        )
    anomalous_ratio = (anomalous_sequences / total_sequences) if total_sequences else 0.0
    return LogSequencesResponse(
        log_blocks=log_blocks,
        pagination=SequencesPagination(
            page=page,
            page_size=page_size,
            total_sequences=total_sequences,
            anomalous_sequences=anomalous_sequences,
            anomalous_ratio=anomalous_ratio,
        ),
    )
