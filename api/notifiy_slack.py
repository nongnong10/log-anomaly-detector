import os
import json
import requests

def send_slack_alert(block_id: str, anomaly_score: float, is_anomaly: bool = True, webhook_env: str = "SLACK_WEBHOOK_URL") -> bool:
    """
    Send a Slack Block Kit alert using webhook URL defined in environment.
    - block_id: e.g. "blk_-1608999687919862906"
    - anomaly_score: numeric score (will be formatted)
    - is_anomaly: boolean flag
    Returns True if POST succeeded (2xx), False otherwise.
    """
    webhook_url = os.environ.get(webhook_env)
    if not webhook_url:
        # missing webhook: skip but do not raise
        print(f"[send_slack_alert] No webhook configured in env {webhook_env}; skipping Slack alert")
        return False

    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "⚠️ [WARN] Anomaly Detected",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Block ID:*\n{block_id}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Anomaly Score:*\n{float(anomaly_score)}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Is Anomaly:*\n{str(bool(is_anomaly)).lower()}"
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "plain_text",
                        "text": "Please investigate immediately.",
                        "emoji": True
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=5)
        if 200 <= resp.status_code < 300:
            return True
        else:
            print(f"[send_slack_alert] Slack returned status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[send_slack_alert] Exception when sending Slack alert: {e}")
        return False

