from lctm.pipeline_lctm import pipeline_lctm
import argparse
from dotenv import load_dotenv

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Train a Regression LCTM to cluster LLM uncertainty.",
        epilog="Enjoy the program! :)",
    )
    
    parser.add_argument('--model', type=str, default="", metavar='PF',
                        help='Model name to find it in model directory.')
    parser.add_argument('--binarizer', type=str, default="default", metavar='PF',)
    parser.add_argument('--hash_batch_ids', action='store_true', default=False)
    parser.add_argument('--document', type=str, default="accumulation", metavar='PF',
                        help='Document to use for training. Default: fetched_batch_data.csv')
    load_dotenv()
    args = parser.parse_args()
    pipeline_lctm(args=args)