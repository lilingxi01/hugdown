import logging
import os
import time
from typing import List, Union, Optional

import multiprocessing as mp

import urllib3
from huggingface_hub import HfFileSystem, hf_hub_url
from tqdm.auto import tqdm

from .logger_init import _init_logger_sub_process, _init_logger_main_process, _init_logger_multiprocessing
from .utils import get_curr_version, prepare_output_dir


def _avoid_duplicate_download_filenames(url_batches: List[str]):
    filenames = set()
    new_url_batches = list(set(url_batches))  # Remove duplicate URLs.
    for url in new_url_batches:
        curr_filename = url.split('/')[-1]
        if curr_filename in filenames:
            new_url_batches.remove(url)
        else:
            filenames.add(curr_filename)
    return new_url_batches


def _avoid_download_existing_files(url_batches: List[str], output_dir: str):
    new_url_batches = []
    for url in url_batches:
        target_path = os.path.join(output_dir, url.split('/')[-1])
        if not os.path.exists(target_path):
            new_url_batches.append(url)
    return new_url_batches


_DEFAULT_NUM_PROC = 1
_DEFAULT_LOG_LEVEL = logging.INFO


class HuggingfaceDL:
    """
    A downloader for downloading files from the web.

    Attributes:
        is_started: Whether the downloader is started.
        is_completed: Whether the downloader is completed.
    """
    def __init__(self,
                 num_proc: Union[int, None] = _DEFAULT_NUM_PROC,
                 token: Union[str, None] = None,
                 log_level: int = _DEFAULT_LOG_LEVEL):
        """
        :param num_proc: The number of processes to use for downloading. (default: number of CPUs)
        :param token: Huggingface Access Token (default: None)
        :param log_level: The log level. (default: logging.INFO)
        """
        self.num_proc = num_proc
        self.token = token
        self.log_level = log_level

        self.files: List[str] = []

        self.is_started = False
        self.is_completed = False

        _init_logger_main_process(log_level=self.log_level)

    def _worker_initializer(self, q):
        """
        Initialize the worker process.
        """
        # Initialize the logger within the sub-process.
        _init_logger_sub_process(q, log_level=self.log_level)

    def preload_files(self, repo: str, data_files: Optional[str] = None):
        if self.token is not None:
            fs = HfFileSystem(token=self.token)
        else:
            fs = HfFileSystem()

        grabbed_files = fs.glob(os.path.join('datasets', repo, data_files or '**'))
        for file in grabbed_files:
            split_filepath = file.split('/')
            repo_type = 'dataset'
            repo_id = split_filepath[1] + '/' + split_filepath[2]
            filename = '/'.join(split_filepath[3:])
            print(repo_type, repo_id, filename)
            downloadable_link = hf_hub_url(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
            )
            self.files.append(downloadable_link)

        logging.info(f'Preloaded {len(self.files)} files from {repo}.')

    def add_download_link(self, link: str):
        """
        Add a download link to the downloader.
        :param link: The download link.
        """
        self.files.append(link)

    def download_executor(self, url: str, target_dir: str):
        """
        Downloads a file from the given URL to the target directory.
        This function will be used for multiprocessing.
        :param url: The URL to download.
        :param target_dir: The target directory to save the downloaded file.
        :return: The local path to the downloaded file.
        """
        os.makedirs(target_dir, exist_ok=True)

        response = None
        filename = url.split('/')[-1]
        total_size = 1

        while response is None:
            http = urllib3.PoolManager()
            if self.token is not None:
                response = http.request('GET', url, preload_content=False,
                                        headers={"Authorization": f"Bearer {self.token}"})
            else:
                response = http.request('GET', url, preload_content=False)
            total_size = int(response.headers.get('Content-Length', 0))

            # If the size does not look right, retry it.
            if total_size < 2048:
                print('Miss. Retry in 3 seconds.')
                response = None
                time.sleep(3)

        progress = tqdm(total=total_size, unit='B', unit_scale=True)

        with open(os.path.join(target_dir, filename), 'wb') as file:
            for chunk in response.stream(4096):
                file.write(chunk)
                progress.update(len(chunk))

        progress.close()
        response.release_conn()
        return os.path.join(target_dir, filename)

    def start(self, output_dir: str):
        """
        Start the downloader.
        :param output_dir: The output directory.
        """
        # Log the version.
        logging.info(f'HugDown Version: {get_curr_version()}')

        if self.is_completed:
            raise Exception("The downloader is already completed.")
        if self.is_started:
            raise Exception("The downloader is already started.")

        # Get the batches of URLs to download.
        url_batches = _avoid_duplicate_download_filenames(url_batches=self.files)
        url_batches = _avoid_download_existing_files(url_batches=url_batches, output_dir=output_dir)

        # Set the number of processes and limit it by the number of CPUs.
        num_proc = min(self.num_proc, os.cpu_count())

        # Mark the downloader as started.
        self.is_started = True

        if len(url_batches) == 0:
            logging.warning('No files are needed to download. Directly exit.')
            self.is_completed = True
            return

        # Prepare the output directory.
        prepare_output_dir(output_dir, exist_ok=True)

        # Log the number of files to download.
        logging.info(f'Will download {len(url_batches)} files.')

        # Download the files.
        logging.info(f'Start downloading. (# of assigned processes: {num_proc})')

        ql, q = _init_logger_multiprocessing(log_level=self.log_level)

        pool = mp.Pool(
            processes=num_proc,
            initializer=self._worker_initializer,
            initargs=(q,),
        )

        total_files = len(url_batches)
        main_progress = tqdm(total=total_files, unit='files', unit_scale=True)
        downloaded_files = []

        # Callback function for the normal return status of a process.
        def _success_callback(downloaded_file_path):
            nonlocal downloaded_files, main_progress
            downloaded_files.append(downloaded_file_path)
            main_progress.update(1)

        # Callback function for the error status of a process.
        def _error_callback(e):
            nonlocal main_progress
            main_progress.update(1)
            main_progress.colour = 'red'

            # Log the error.
            logging.error(f'Error occurred when downloading: {e}')

        # Start processes.
        for url in url_batches:
            pool.apply_async(
                func=self.download_executor,
                args=(url, output_dir),
                callback=_success_callback,
                error_callback=_error_callback,
            )

        # Wait for all processes to finish.
        pool.close()
        pool.join()
        ql.stop()
        main_progress.close()

        # Log the completion.
        logging.info(f'Downloading completed. {len(downloaded_files)} files are downloaded.')
        self.is_completed = True

        return downloaded_files



