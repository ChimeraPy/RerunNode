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
        rerun_mappings: Dict[str, Any] = None,
        recording_id: Optional[str] = None,
        server_addr: str = "127.0.0.1:9876",
        name: str = "ChimeraPyRerunNode",
    ) -> None:
        super().__init__(name=name)
        self.application_id = application_id
        self.recording_id = recording_id
        self.server_addr = server_addr
        self.recording: Optional[rr.RecordingStream] = None
        self.is_first_step: bool = False
        self.rerun_mappings: Dict[str, Any] = rerun_mappings or {}

    def setup(self) -> None:
        self.recording = rr.new_recording(
            application_id=self.application_id, recording_id=self.recording_id
        )

        self.is_first_step = False
        rr.connect(addr=self.server_addr, recording=self.recording)
        rr.set_global_data_recording(recording=self.recording)

    def step(self, data_chunks: Dict[str, DataChunk]) -> None:
        if not self.is_first_step:
            self.is_first_step = True

        for name, chunk in data_chunks.items():
            for key in self.rerun_mappings:
                try:
                    value = chunk.get(key)["value"]
                    self._log_archetype(name, value, key)
                except Exception as e:
                    self.logger.error(str(e))

    def teardown(self) -> None:
        rr.disconnect(self.recording)

    def _log_archetype(self, name, value, key):
        archetype_info = self.rerun_mappings[key]
        if archetype_info["primitive"]:
            if archetype_info["type"] == "Image":
                if len(value.shape) == 3:
                    data = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                else:
                    data = value
                rr.log(f"{name}/{key}", rr.Image(data=data))
        else:
            for attr, archetype_type in archetype_info["attributes"].items():
                attribute_value = getattr(value, attr)
                if archetype_type == "Image":
                    data = rr.Image(
                        data=cv2.cvtColor(attribute_value, cv2.COLOR_BGR2RGB)
                        if len(attribute_value.shape) == 3
                        else attribute_value,
                    )
                    rr.log(f"{name}/{key}/{attr}", data)
