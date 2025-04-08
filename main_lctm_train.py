from lctm.pipeline_lctm import pipeline_lctm
import argparse
from dotenv import load_dotenv

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Train a Regression LCTM to cluster LLM uncertainty.",
        epilog="Enjoy the program! :)",
    )
    
    parser.add_argument('--csv_path', type=str, default="", metavar='PF',
                        help='File containing the model training data')
    
    load_dotenv()
    args = parser.parse_args()
    pipeline_lctm(args=args)