import sys
# sys.path.append("labelcritic_tm/tmu")
# from labelcritic_tm.tmu.tsetlin_machine import LCTM
from tmu.tsetlin_machine import LCTM

if __name__=="__main__":
    lctm_config = {
        "number_of_clauses": 10,
        "T": 2,
        "s": .8,
    }
    lctm = LCTM(**lctm_config)
    print("LCTM correctly loaded!")

