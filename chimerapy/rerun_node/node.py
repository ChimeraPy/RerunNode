from typing import Any, Dict, Optional

import cv2
import rerun as rr

from chimerapy.engine import DataChunk, Node
from chimerapy.orchestrator import sink_node


@sink_node(name="ChimeraPyRerunNode")
class RerunNode(Node):
    def __init__(
        self,
        application_id: str,
        log_chunk_keys: Dict[str, Any] = None,
        recording_id: Optional[str] = None,
        server_addr: str = "127.0.0.1:9876",
        name: str = "ChimeraPyRerunNode",
    ) -> None:
        super().__init__(name=name)
        self.application_id = application_id
        self.recording_id = recording_id
        self.server_addr = server_addr
        self.log_chunk_keys = log_chunk_keys or {}
        self.recording: Optional[rr.RecordingStream] = None
        self.is_first_step: bool = False

    def setup(self) -> None:

        self.recording = rr.new_recording(
            application_id=self.application_id, recording_id=self.recording_id
        )

        self.is_first_step = False
        rr.connect(addr=self.server_addr, recording=self.recording)
        rr.set_global_data_recording(recording=self.recording)

    def step(self, data_chunks: Dict[str, DataChunk]) -> None:
        if not self.is_first_step:
            print("First step")
            self.is_first_step = True

        for name, chunk in data_chunks.items():
            image_data = chunk.get("frame")["value"]
            if image_data is not None:
                image_data_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                rr.log(f"{name}/frame", rr.Image(image_data_rgb))

    def teardown(self) -> None:
        rr.disconnect(self.recording)
