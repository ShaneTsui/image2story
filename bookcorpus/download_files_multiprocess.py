#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import time
import threading, multiprocessing
from multiprocessing import Pool, TimeoutError

from urllib.request import urlopen, urlretrieve

import epub2txt
import os
from progressbar import ProgressBar
from glob import glob
import sys
import json

SLEEP_SEC = 0.05
SUCCESS_SLEEP_SEC = 0.001
RETRY_SLEEP_SEC = 0.5
MAX_OPEN_COUNT = 10

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', '--out', type=str, required=True)
parser.add_argument('--list-path', '--list', type=str, required=True)
parser.add_argument('--trash-bad-count', action='store_true', default=False)
args = parser.parse_args()


def write_txt(txt, out_path, num_words=None):
    # occasionally, some epubs text are decoded with errors
    # e.g. repeated bib lines
    # filter out them by comparing number of words
    counted_num_words = len(txt.split())
    if not txt.strip():
        pass
    elif num_words is None or \
            (num_words * 0.5 < counted_num_words < num_words * 1.5):
        with open(out_path, "w") as txt_out:  # convert epub2txt and save
            txt_out.write(txt)

dataset = []
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
filelist_path = args.list_path

lines = list(open(filelist_path).readlines())

done_files = set([os.path.split(path)[-1]
                    for path in glob(os.path.join(out_dir, '*.txt'))])
sys.stderr.write('{} files had already been saved in {}.\n'.format(
    len(done_files), out_dir))

def download(i, line):
    # read data
    try:
        data = json.loads(line.strip())
        _, book_id = os.path.split(data['page'])
        _, file_name = os.path.split(data['epub'])

        out_file_name = '{}__{}'.format(
            book_id, file_name.replace('.epub', '.txt'))
        out_path = os.path.join(out_dir, out_file_name)
        # if out_file_name in done_files:
        #     return
        if data['txt']:
            # try to download .txt file
            for try_count in range(MAX_OPEN_COUNT):
                try:
                    response = urlopen(data['txt'])
                    if try_count >= 1:
                        sys.stderr.write(
                            'Succeeded in opening {}\n'.format(data['txt']))
                    time.sleep(SUCCESS_SLEEP_SEC)
                    break # success
                except Exception as e:
                    sys.stderr.write(
                        'Failed to open {}\n'.format(data['txt']))
                    sys.stderr.write(
                        '{}: {}\n'.format(type(e).__name__, str(e)))
                    time.sleep(RETRY_SLEEP_SEC)
            else:
                sys.stderr.write(
                    ' Gave up to open {}\n'.format(data['txt']))
            txt = response.read().decode('utf-8', 'ignore')
            write_txt(txt, out_path, None)
        else:
            # revenge by converting .epub to .txt
            tmp_path = os.path.join(out_dir, f'{book_id}__{file_name}')
            for try_count in range(MAX_OPEN_COUNT):
                try:
                    urlretrieve(data['epub'], tmp_path)  # download epub
                    if try_count >= 1:
                        sys.stderr.write(
                            'Succeeded in opening {}\n'.format(data['epub']))
                    time.sleep(SUCCESS_SLEEP_SEC)
                    break  # success
                except Exception as e:
                    sys.stderr.write(
                        'Failed to open {}\n'.format(data['epub']))
                    sys.stderr.write(
                        '{}: {}\n'.format(type(e).__name__, str(e)))
                    time.sleep(RETRY_SLEEP_SEC)
            else:
                sys.stderr.write(
                    ' Gave up to open {}\n'.format(data['epub']))
            txt = epub2txt.epub2txt(tmp_path).convert()
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        if os.path.exists(out_path):
            os.remove(out_path)

def main():
    num_workers = multiprocessing.cpu_count() 
    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(download, [(i, line) for i, line in enumerate(lines)])

if __name__ == '__main__':
    main()