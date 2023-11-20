import logging
from pathlib import Path

import torchaudio

from src.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            entry["ref"] = str(Path(entry["ref"]).absolute().resolve())
            ref_info = torchaudio.info(entry["ref"])
            entry["ref_length"] = ref_info.num_frames / ref_info.sample_rate

            entry["mix"] = str(Path(entry["mix"]).absolute().resolve())
            mix_info = torchaudio.info(entry["mix"])
            entry["mix_length"] = mix_info.num_frames / mix_info.sample_rate

            entry["target"] = str(Path(entry["target"]).absolute().resolve())
            target_info = torchaudio.info(entry["target"])
            entry["target_length"] = target_info.num_frames / target_info.sample_rate

            entry["target_id"] = 0
            entry["total_speakers"] = 1

        super().__init__(index, *args, **kwargs)
