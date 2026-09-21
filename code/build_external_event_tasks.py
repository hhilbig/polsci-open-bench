#!/usr/bin/env python3
"""Build four frozen external event tasks without inspecting model outcomes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
SEED_TEXT = "event-progress-20260823"

def rank(value: str) -> str:
    return hashlib.sha256(f"{SEED_TEXT}|{value}".encode()).hexdigest()

def take(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    return frame.assign(_rank=frame.source_id.map(rank)).sort_values("_rank").head(n).drop(columns="_rank")

def write_task(name: str, frame: pd.DataFrame, labels: list[str], label_key: str,
               source: str, prompt: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{name}.csv"; prompt_path = output / f"{name}.txt"; yaml_path = output / f"{name}.yaml"
    frame.to_csv(data_path, index=False); prompt_path.write_text(prompt)
    spec = {"order": 1, "name": name, "family": "External event validation", "source": source,
            "data_file": data_path.name, "prompt_file": prompt_path.name, "label_kind": "categorical",
            "label_key": label_key, "labels": labels, "id": {"kind": "column", "column": "source_id"},
            "text": {"template": "{text}"}, "ground_truth": {"column": "gold"}}
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

def build(maven: Path, rams: Path, arabic: Path, output: Path) -> pd.DataFrame:
    records = [json.loads(line) for line in maven.read_text().splitlines()]
    rows=[]
    for i,row in enumerate(records):
        structures=row["output"]["json_structures"]
        rows.append({"source_id":f"maven-valid-{i}","text":row["input"],"gold":"EVENT" if structures else "NO_EVENT"})
    md=pd.DataFrame(rows); md=pd.concat([take(md[md.gold.eq(x)],250) for x in ["EVENT","NO_EVENT"]])
    write_task("maven_event_presence",md,["EVENT","NO_EVENT"],"event_presence","MAVEN validation split (Wang et al. 2020)",
               "Does the sentence describe an event? Return EVENT or NO_EVENT.",output)
    rows=[]
    for line in (rams/"test.jsonlines").read_text().splitlines():
        row=json.loads(line); start,end,types=row["evt_triggers"][0]; tokens=sum(row["sentences"],[])
        event_type=types[0][0]; trigger=" ".join(tokens[start:end+1]); text=" ".join(tokens)
        rows.append({"source_id":row["doc_key"],"text":f"Trigger: {trigger}\nDocument: {text}","gold":event_type})
    rd=take(pd.DataFrame(rows),500); labels=sorted(pd.DataFrame(rows).gold.unique())
    write_task("rams_event_type",rd,labels,"event_type","RAMS 1.0b test split (Ebner et al. 2020)",
               "Classify the highlighted trigger into one permitted event type.",output)
    for kind in ["assault","protest"]:
        raw=pd.read_csv(arabic/f"{kind}_gsr.csv"); before=len(raw); raw=raw[raw.label.ne("ambiguous")].copy()
        raw["source_id"]=raw.id.astype(str); raw["gold"]=raw.label.eq("yes").map({True:"EVENT",False:"NO_EVENT"})
        positives=raw[raw.gold.eq("EVENT")]; negatives=raw[raw.gold.eq("NO_EVENT")]
        pos_n=min(250,len(positives)); ad=pd.concat([take(positives,pos_n),take(negatives,500-pos_n)])[["source_id","text","gold"]]
        write_task(f"arabic_gsr_{kind}_presence",ad,["EVENT","NO_EVENT"],"event_presence",
                   f"Arabic Event GSR {kind.upper()} (Open Event Data)",
                   f"Does the Arabic sentence describe a {kind.upper()} event? Return EVENT or NO_EVENT.",output)
        if (before-len(raw))/before > .20: raise ValueError("ambiguous-label exclusion exceeds 20%")
    audit=[]
    for path in sorted(output.glob("*.csv")):
        frame=pd.read_csv(path); audit.append({"task":path.stem,"rows":len(frame),"unique_ids":frame.source_id.nunique(),
                                               "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.DataFrame(audit)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--maven",type=Path,required=True); p.add_argument("--rams",type=Path,required=True)
    p.add_argument("--arabic",type=Path,required=True); p.add_argument("--output",type=Path,default=REPO/"experiments/external_event_tasks")
    a=p.parse_args(); audit=build(a.maven,a.rams,a.arabic,a.output); audit.to_csv(a.output/"build_audit.csv",index=False)
if __name__=="__main__": main()
