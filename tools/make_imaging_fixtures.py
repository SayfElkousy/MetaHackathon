"""Write the synthetic imaging fixtures (fixtures/imaging/). Not patient data."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imaging.synthetic import make_fixtures  # noqa: E402

if __name__ == "__main__":
    print(make_fixtures(os.path.join(ROOT, "fixtures", "imaging")))
