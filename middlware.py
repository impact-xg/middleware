from predictor import Predictor
trace_data = [
    {
        "timestamp": 20250510124644,
        "20250510124636": {
            "throughput": 994.2,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 21.442,
            "speed": "0"
        },
        "20250510124638": {
            "throughput": 1150.4,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.4,
            "jitter": 21.398,
            "speed": "0"
        },
        "20250510124640": {
            "throughput": 1351.5,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 16.687,
            "speed": "0"
        },
        "20250510124642": {
            "throughput": 1384.5,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.3,
            "jitter": 20.939,
            "speed": "0"
        },
        "20250510124644": {
            "throughput": 1169.0,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 20.542,
            "speed": "0"
        }
    },
     {
        "timestamp": 20250510124744,
        "20250510124636": {
            "throughput": 994.2,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 21.442,
            "speed": "0"
        },
        "20250510124638": {
            "throughput": 1150.4,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.4,
            "jitter": 21.398,
            "speed": "0"
        },
        "20250510124640": {
            "throughput": 1351.5,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 16.687,
            "speed": "0"
        },
        "20250510124642": {
            "throughput": 1384.5,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.3,
            "jitter": 20.939,
            "speed": "0"
        },
        "20250510124644": {
            "throughput": 1169.0,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 20.542,
            "speed": "0"
        }
    },
     {
        "timestamp": 20250510124844,
        "20250510124636": {
            "throughput": 994.2,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 21.442,
            "speed": "0"
        },
        "20250510124638": {
            "throughput": 1150.4,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.4,
            "jitter": 21.398,
            "speed": "0"
        },
        "20250510124640": {
            "throughput": 1351.5,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 16.687,
            "speed": "0"
        },
        "20250510124642": {
            "throughput": 1384.5,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.3,
            "jitter": 20.939,
            "speed": "0"
        },
        "20250510124644": {
            "throughput": 1169.0,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 20.542,
            "speed": "0"
        }
    },
     {
        "timestamp": 20250510124944,
        "20250510124636": {
            "throughput": 994.2,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 21.442,
            "speed": "0"
        },
        "20250510124638": {
            "throughput": 1150.4,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.4,
            "jitter": 21.398,
            "speed": "0"
        },
        "20250510124640": {
            "throughput": 1351.5,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 16.687,
            "speed": "0"
        },
        "20250510124642": {
            "throughput": 1384.5,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.3,
            "jitter": 20.939,
            "speed": "0"
        },
        "20250510124644": {
            "throughput": 1169.0,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 20.542,
            "speed": "0"
        }
    },
    {
        "timestamp": 20250510125044,
        "20250510124636": {
            "throughput": 994.2,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 21.442,
            "speed": "0"
        },
        "20250510124638": {
            "throughput": 1150.4,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.4,
            "jitter": 21.398,
            "speed": "0"
        },
        "20250510124640": {
            "throughput": 1351.5,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 16.687,
            "speed": "0"
        },
        "20250510124642": {
            "throughput": 1384.5,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.3,
            "jitter": 20.939,
            "speed": "0"
        },
        "20250510124644": {
            "throughput": 1169.0,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 20.542,
            "speed": "0"
        }
    }
]

qoe_predictor = Predictor()
result = qoe_predictor.infer(trace_data)
print(result)
