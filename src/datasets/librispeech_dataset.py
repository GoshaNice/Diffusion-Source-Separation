import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
from glob import glob

from src.base.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from src.base.base_mixer import MixtureGenerator, LibriSpeechSpeakerFiles

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, nfiles=1000, test=False, *args, **kwargs):
        assert part in URL_LINKS or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.nfiles = nfiles
        self.test = test

        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS
                    if "train" in part
                ],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_ss_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        speakers_dirs = [el.name for el in os.scandir(split_dir)]
        speakers_files = [
            LibriSpeechSpeakerFiles(i, split_dir, audioTemplate="*.flac")
            for i in speakers_dirs
        ]
        out_folder = self._data_dir / f"{part}_ss"
        mixer = MixtureGenerator(
            speakers_files=speakers_files,
            out_folder=out_folder,
            nfiles=self.nfiles,
            test=self.test,
        )

        index = []
        if self.test:
            mixer.generate_mixes(
                snr_levels=[0, 0],
                num_workers=2,
                trim_db=None,
                vad_db=None,
                update_steps=100,
                audioLen=None,
            )
        else:
            mixer.generate_mixes(
                snr_levels=[0, 0],
                num_workers=2,
                trim_db=None,
                vad_db=None,
                update_steps=100,
                audioLen=3,
            )

        ref = sorted(glob(os.path.join(out_folder, "*-ref.wav")))
        mix = sorted(glob(os.path.join(out_folder, "*-mixed.wav")))
        noise = sorted(glob(os.path.join(out_folder, "*-noise.wav")))
        target = sorted(glob(os.path.join(out_folder, "*-target.wav")))

        map_target_speaker = {}
        i = 0

        for path in ref:
            speaker_id = path.split("/")[-1].split("_")[0]
            if speaker_id in map_target_speaker.keys():
                continue
            map_target_speaker[speaker_id] = i
            i += 1

        total_speakers = i

        for i in range(len(ref)):
            ref_path = ref[i]
            ref_info = torchaudio.info(ref_path)
            ref_length = ref_info.num_frames / ref_info.sample_rate

            mix_path = mix[i]
            mix_info = torchaudio.info(mix_path)
            mix_length = mix_info.num_frames / mix_info.sample_rate

            target_path = target[i]
            target_info = torchaudio.info(target_path)
            target_length = target_info.num_frames / target_info.sample_rate

            noise_path = noise[i]
            noise_info = torchaudio.info(noise_path)
            noise_length = noise_info.num_frames / noise_info.sample_rate

            target_id = ref_path.split("/")[-1].split("_")[0]
            index.append(
                {
                    "ref": ref_path,
                    "ref_length": ref_length,
                    "mix": mix_path,
                    "mix_length": mix_length,
                    "target": target_path,
                    "target_length": target_length,
                    "noise": noise_path,
                    "noise_length": noise_length,
                    "target_id": map_target_speaker[target_id],
                    "total_speakers": total_speakers,
                }
            )
        return index
