import sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from analyze_external_disagreement import estimate

class TestExternalDisagreement(unittest.TestCase):
    def test_deterministic_positive_slope(self):
        rows=[]
        for d in ['a','b','c']:
            for i in range(100):
                agreement=i/99; rows.append({'dataset':d,'agreement':agreement,'delta':agreement-.5})
        frame=pd.DataFrame(rows); one=estimate(frame,draws=200,seed=20260822); two=estimate(frame,draws=200,seed=20260822)
        pd.testing.assert_frame_equal(one,two); pooled=one.loc[one.dataset.eq('Equal-dataset pooled')].iloc[0]
        self.assertGreater(pooled.slope_per_agreement_sd,0); self.assertGreater(pooled.ci_low,0)

if __name__=='__main__': unittest.main()
