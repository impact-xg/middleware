from predictor import Predictor
trace_data = {
    "throughput": 994.2,
    "packets_lost": 0.0,
    "packet_loss_rate": 0.0,
    "jitter": 21.442,
    "speed": "0"
}

qoe_predictor = Predictor()
result = qoe_predictor.infer(trace_data)
print(result)
