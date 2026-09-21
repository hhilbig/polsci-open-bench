"""Independent release guard regressions found during completion audit."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from build_refresh_release import validate_panel, class_metrics


class ReleaseAuditTests(unittest.TestCase):
    def test_fractional_binary_prediction_is_not_a_valid_label(self):
        panel={'rows':[{'task':'t','item':{'item_id':'1','gt':{'a':0}}}]}
        task={'name':'t','label_kind':'binary','label_key':'a','labels':['a']}
        for prediction in [.9,1.9,-.1]:
            with self.subTest(prediction=prediction):
                rows=pd.DataFrame({'task':['t'],'item_id':['1'],'gt_a':[0],'pred_a':[prediction],'parse_error':[None]})
                with self.assertRaises(ValueError):
                    validate_panel(rows,panel,{'t':task})

    def test_all_malformed_categorical_predictions_still_score(self):
        rows=pd.DataFrame({'gt_a':['x','y'],'pred_a':[np.nan,np.nan],'parse_error':['invalid','invalid']})
        task={'label_kind':'categorical','label_key':'a','labels':['x','y']}
        result=class_metrics(rows,task)
        self.assertTrue(all(row['f1']==0 for row in result))
        self.assertTrue(rows.pred_a.isna().all())


if __name__=='__main__':
    unittest.main()
