import sys,tempfile,unittest
from pathlib import Path
import pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"code"))
from build_external_event_tasks import build

class ExternalEventTaskTests(unittest.TestCase):
    def test_frozen_outputs(self):
        source=ROOT/"experiments/external_event_tasks"
        audit=pd.read_csv(source/"build_audit.csv")
        self.assertEqual(set(audit.task),{"maven_event_presence","rams_event_type","arabic_gsr_assault_presence","arabic_gsr_protest_presence"})
        self.assertEqual(audit.rows.tolist(),[500,500,500,500])
        for task in audit.task:
            frame=pd.read_csv(source/f"{task}.csv"); spec=yaml.safe_load((source/f"{task}.yaml").read_text())
            self.assertEqual(len(frame),frame.source_id.nunique()); self.assertTrue(set(frame.gold).issubset(set(spec["labels"])))
        config=yaml.safe_load((ROOT/"experiments/external_event_hive_20260823.yaml").read_text())
        self.assertEqual(config["task_scope"]["expected_items"],2000)
        self.assertEqual(set(config["models"]),{"llama3_1_70b_external_event","qwen3_6_27b_external_event"})
if __name__=="__main__": unittest.main()
