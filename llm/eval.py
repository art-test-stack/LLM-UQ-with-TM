class Eval:
    def __init__(self, metrics):
        self.metrics = metrics
        self.results = {}
        self.best = None
        self.best_metric = None
    
    def _bleu(self, pred, tgt):
        pass
    
    def _rouge(self, pred, tgt):
        pass
    
    def update(self, pred, tgt):
        pass
