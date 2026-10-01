"""Entry point for XWBC direct PPO and six-route evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_adaptation.learning.xwbc import main

if __name__ == '__main__':
    main()
