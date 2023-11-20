import logging
from pathlib import Path

from src.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, mix_dir, refs_dir, targets_dir=None, *args, **kwargs):
        data = []
        for path in Path(mix_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["mix"] = str(path)

                name = path.name
                ref_name = name.replace("mixed", "ref")
                target_name = name.replace("mixed", "target")
                entry["ref"] = str(Path(refs_dir) / ref_name)
                entry["target"] = str(Path(targets_dir) / target_name)

            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
