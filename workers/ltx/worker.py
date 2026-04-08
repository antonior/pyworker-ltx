"""Vast.ai PyWorker for LTX 2.3 ComfyUI Serverless.

Monitors ComfyUI startup via log file and reports readiness to the
Vast.ai Serverless controller. No benchmark — LTX is a video model,
not compatible with the default Text2Image benchmark format.
"""

from vastai import Worker, WorkerConfig, HandlerConfig

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
        max_queue_time=60.0,
    ),
)

Worker(worker_config).run()
