import os
import shutil
from hugdown import HugDown


download_output_dir = "./tests/download_temp"


def test_get_files_from_repo():
    downloader = HugDown(num_proc=1)
    downloader.preload_files(repo='knkarthick/samsum', data_files='train.csv')

    assert len(downloader.files) == 1


def test_download_files_from_repo():
    downloader = HugDown(num_proc=1)
    downloader.preload_files(repo='knkarthick/samsum', data_files='*.csv')
    downloader.start(output_dir=download_output_dir)

    assert len(os.listdir(download_output_dir)) == 3

    shutil.rmtree(download_output_dir)
