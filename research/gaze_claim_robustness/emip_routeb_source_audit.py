#!/usr/bin/env python3
"""Non-outcome EMIP OSF source audit for THMS Route B.

Does not fit gaze models, inspect comprehension answers, or release participant data.
Schema/availability-only audit; see audit report for decision gates.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import requests

ROOT_URL = "https://api.osf.io/v2/nodes/53kts/files/"
OUT = Path(os.environ.get("EMIP_AUDIT_OUT", "emip_routeb_audit"))
OUT.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "THMS-EMIP-feasibility-audit/1.0"})
FILES = []
WARN = []
REQUESTS = 0


def api_get(url: str):
    global REQUESTS
    REQUESTS += 1
    if REQUESTS > 250:
        raise RuntimeError("API call cap reached; audit is incomplete")
    response = SESSION.get(url, timeout=40)
    response.raise_for_status()
    return response.json()


def get_pages(url: str):
    while url:
        response = api_get(url)
        for item in response.get("data", []):
            yield item
        url = response.get("links", {}).get("next")


def scan():
    roots = list(get_pages(ROOT_URL))
    for root in roots:
        links = root.get("links", {})
        rel = root.get("relationships", {}).get("files", {}).get("links", {}).get("related", {})
        url = rel.get("href") if isinstance(rel, dict) else None
        # OSF providers expose their top-level folder at /files/osfstorage/.
        if not url and root.get("id", "").endswith("osfstorage"):
            url = "https://api.osf.io/v2/nodes/53kts/files/osfstorage/"
        if not url:
            candidate = links.get("related", {}).get("href") if isinstance(links.get("related"), dict) else links.get("related")
            if candidate and "/files/" in candidate:
                url = candidate
        if not url:
            WARN.append("Unable to identify provider folder URL: " + str(root.get("id")))
            continue
        queue = deque([("", url)])
        seen = set()
        while queue:
            prefix, address = queue.popleft()
            if address in seen:
                continue
            seen.add(address)
            for item in get_pages(address):
                attr = item.get("attributes", {})
                name = str(attr.get("name") or item.get("id"))
                kind = attr.get("kind")
                path = f"{prefix}/{name}".strip("/")
                if kind == "file":
                    FILES.append({
                        "path": path, "size": attr.get("size"),
                        "id": item.get("id"), "download_url": item.get("links", {}).get("download"),
                        "modified": attr.get("date_modified"),
                    })
                elif kind == "folder":
                    rel = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {})
                    address2 = rel.get("href") if isinstance(rel, dict) else None
                    if not address2:
                        related = item.get("links", {}).get("related")
                        address2 = related.get("href") if isinstance(related, dict) else related
                    if address2:
                        queue.append((path, address2))
                    else:
                        WARN.append("Folder not traversable: "+path)
        if len(FILES) > 5000:
            raise RuntimeError("Unexpectedly large file list; examine manually")


def prefix_text(url: str, max_bytes=160_000) -> str:
    with SESSION.get(url, stream=True, timeout=75) as res:
        res.raise_for_status()
        content = bytearray()
        for chunk in res.iter_content(16384):
            content += chunk
            if len(content) >= max_bytes:
                break
    return content.decode("utf-8-sig", errors="replace")


def main():
    status = {"audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "osf_node": "53kts", "source_url": ROOT_URL, "status":"pending",
              "file_count":0, "metadata":{}, "raw":{}, "stimuli":{},
              "warnings":WARN}
    try:
        scan()
        status["file_count"] = len(FILES)
        status["top_level"] = dict(Counter(f["path"].split("/")[0] for f in FILES))
        with (OUT/"source_manifest.csv").open("w", newline="", encoding="utf-8") as out:
            cols = ["path","size","id","modified"]  # do not expose signed URLs
            writer=csv.DictWriter(out, fieldnames=cols); writer.writeheader()
            writer.writerows({k:f.get(k) for k in cols} for f in sorted(FILES,key=lambda x:x["path"]))

        meta = next((f for f in FILES if f["path"].lower().endswith("emip_metadata.csv")), None)
        raw_files = [f for f in FILES if re.search(r"rawdata/.*\\.(tsv|txt|csv)$", f["path"], re.I)]
        if not raw_files:
            raw_files = [f for f in FILES if re.search(r"\\.(tsv)$", f["path"], re.I)]
        stimulus_files = [f for f in FILES if re.search(r"stimuli/|aoi",f["path"],re.I)]
        status["raw"]={"file_count":len(raw_files),"extensions":dict(Counter(Path(f["path"]).suffix.lower() for f in raw_files))}
        status["stimuli"]={"file_count":len(stimulus_files),"extensions":dict(Counter(Path(f["path"]).suffix.lower() for f in stimulus_files)),
                           "sample_paths":[f["path"] for f in sorted(stimulus_files,key=lambda x:x["path"])[:15]]}

        if meta and meta.get("download_url"):
            data=prefix_text(meta["download_url"],max_bytes=256_000)
            data=data.lstrip("\ufeff")
            rows=csv.reader(io.StringIO(data))
            try:
                header=next(rows)
            except StopIteration:
                header=[]
            status["metadata"]={"present":True,"columns":header,"first_download_prefix_bytes":len(data.encode("utf-8")),
                                "has_expertise":any("expert" in c.lower() for c in header),
                                "has_order":any("order" in c.lower() for c in header),
                                "has_correctness_or_response":any(re.search(r"correct|answer|response",c,re.I) for c in header)}
        else:
            status["metadata"]={"present":False}
            WARN.append("Cannot locate emip_metadata.csv in accessible file tree")

        if raw_files and raw_files[0].get("download_url"):
            sample=sorted(raw_files,key=lambda x:x["path"])[0]
            prefix=prefix_text(sample["download_url"],max_bytes=256_000)
            rows=[r for r in prefix.splitlines() if r and not r.startswith("#")]
            head=next((r for r in rows if "Time" in r and "\t" in r), rows[0] if rows else "")
            types=Counter()
            for row in rows[1:100]:
                fields=row.split("\t")
                if len(fields)>1:
                    types[fields[1].strip()]+=1
            status["raw"].update({"inspected_file":sample["path"],"header_columns":head.split("\t"),
                                  "first_100_row_types":dict(types),
                                  "message_name_present":any("MSG" in r for r in rows),
                                  "timestamp_column_present":"Time" in head.split("\t")})
        else:
            WARN.append("No downloadable raw TSV file located")
        status["status"]="schema_audited" if status["metadata"].get("present") and status["raw"].get("timestamp_column_present") and stimulus_files else "partial_audit"
    except Exception as exc:
        status["status"]="source_access_blocked"
        status["error"]=type(exc).__name__+": "+str(exc)
        print(status["error"],file=sys.stderr)

    (OUT/"feasibility.json").write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in status.items() if k not in ("stimuli","metadata")},indent=2))
    print("files written:", OUT)
    # An inaccessible source must fail the CI audit gate transparently.
    return 0 if status["status"]=="schema_audited" else 2

if __name__=="__main__":
    raise SystemExit(main())
