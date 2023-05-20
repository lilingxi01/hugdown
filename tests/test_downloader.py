import os
import shutil
from hugdown import HuggingfaceDL


download_output_dir = "./tests/download_temp"


def test_get_files_from_repo():
    huggingface_dl = HuggingfaceDL(num_proc=1)
    huggingface_dl.preload_files(repo='knkarthick/samsum', data_files='train.csv')

    assert len(huggingface_dl.files) == 1


def test_download_files_from_repo():
    huggingface_dl = HuggingfaceDL(num_proc=1)
    huggingface_dl.preload_files(repo='knkarthick/samsum', data_files='*.csv')
    huggingface_dl.start(output_dir=download_output_dir)

    assert len(os.listdir(download_output_dir)) == 3

    shutil.rmtree(download_output_dir)
