"""Vast.ai PyWorker for LTX 2.3 ComfyUI Serverless.

Monitors ComfyUI startup via log file, runs a minimal benchmark via
the ComfyUI API Wrapper, and reports readiness to the Vast.ai
Serverless controller.
"""

from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig

CHECKPOINT = "ltx-2.3-22b-distilled-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"


def _build_benchmark_request(seed: int = 42) -> dict:
    """Minimal LTX 2.3 text-to-video workflow for benchmarking.

    9 frames at 512x320, 4 steps — fast validation that GPU + model work.
    Wrapped in the API Wrapper's expected {"input": {"workflow_json": ...}} format.
    """
    return {
        "input": {
            "workflow_json": {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": CHECKPOINT},
                },
                "2": {
                    "class_type": "LTXAVTextEncoderLoader",
                    "inputs": {
                        "text_encoder": TEXT_ENCODER,
                        "ckpt_name": CHECKPOINT,
                        "device": "default",
                    },
                },
                "3": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": "A lighthouse on a cliff at sunset",
                        "clip": ["2", 0],
                    },
                },
                "4": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": "blurry, low quality",
                        "clip": ["2", 0],
                    },
                },
                "5": {
                    "class_type": "LTXVConditioning",
                    "inputs": {
                        "positive": ["3", 0],
                        "negative": ["4", 0],
                        "frame_rate": 25.0,
                    },
                },
                "6": {
                    "class_type": "EmptyLTXVLatentVideo",
                    "inputs": {
                        "width": 512,
                        "height": 320,
                        "length": 9,
                        "batch_size": 1,
                    },
                },
                "9": {
                    "class_type": "CFGGuider",
                    "inputs": {
                        "model": ["1", 0],
                        "positive": ["5", 0],
                        "negative": ["5", 1],
                        "cfg": 1.0,
                    },
                },
                "10": {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                },
                "11": {
                    "class_type": "LTXVScheduler",
                    "inputs": {
                        "steps": 4,
                        "max_shift": 2.05,
                        "base_shift": 0.95,
                        "stretch": True,
                        "terminal": 0.1,
                        "latent": ["6", 0],
                    },
                },
                "12": {
                    "class_type": "RandomNoise",
                    "inputs": {"noise_seed": seed},
                },
                "13": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["12", 0],
                        "guider": ["9", 0],
                        "sampler": ["10", 0],
                        "sigmas": ["11", 0],
                        "latent_image": ["6", 0],
                    },
                },
                "14": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["13", 0], "vae": ["1", 2]},
                },
                "15": {
                    "class_type": "VHS_VideoCombine",
                    "inputs": {
                        "images": ["14", 0],
                        "frame_rate": 25,
                        "loop_count": 0,
                        "filename_prefix": "benchmark_ltx",
                        "format": "video/h264-mp4",
                        "pix_fmt": "yuv420p",
                        "crf": 23,
                        "save_metadata": False,
                        "trim_to_audio": False,
                        "pingpong": False,
                        "save_output": False,
                    },
                },
            }
        }
    }


worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=18288,
    model_log_file="/var/log/portal/comfyui.log",
    model_healthcheck_endpoint="/health",
    on_load=["To see the GUI go to: "],
    on_error=[
        "MetadataIncompleteBuffer",
        "Value not in list: ",
        "[ERROR] Provisioning Script failed",
    ],
    handler_config=HandlerConfig(
        route="/generate/sync",
        allow_parallel_requests=False,
        max_queue_time=120.0,
    ),
    benchmark_config=BenchmarkConfig(
        dataset=[_build_benchmark_request(seed=42)],
        runs=1,
        workload_calculator=lambda _: 10000.0,
    ),
)

Worker(worker_config).run()
