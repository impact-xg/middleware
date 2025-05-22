class Predictor:
    def __init__(self):
        print("Predictor initialized.")

    def infer(self, stats):
        print("throughput:", stats.get("throughput"))
        print("packets_lost:", stats.get("packets_lost"))
        print("packet_loss_rate:", stats.get("packet_loss_rate"))
        print("jitter:", stats.get("jitter"))
        print("speed:", stats.get("speed"))
        return -1
