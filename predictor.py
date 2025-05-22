import json

class Predictor:
    def __init__(self):
        print("Predictor initialized.")

    def infer(self, stats):
        print (json.dumps(stats))
        return -1
