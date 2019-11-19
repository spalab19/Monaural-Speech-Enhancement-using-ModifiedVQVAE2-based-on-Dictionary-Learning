#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITMIT data corpus I/O

Module (Third-party library)
======
- Numpy
- Pandas

Update Contents
---------------
 (19.11.19) Initialize

Last update: Tue Nov 19 2019
@author t-take
"""

import os
import pathlib
import re

import numpy as np
import pandas as pd


class TIMIT(object):
    """
    TIMIT Acoustic-Phonetic Continuous Speech Corpus

    Parameter
    ---------
    root : str or pathlib.Path
        TIMIT data's root path

    Attributes
    ----------
    sample_rate : int
        sampling rate (Hz)
    path_fmt : str
        .wav file path format
    speaker_information : pandas.DataFrame
        TIMIT speaker's information
    """
    def __init__(self, root):
        super(TIMIT, self).__init__()
        self.root = pathlib.Path(root).expanduser()
        self.sample_rate = 16000

        self.path_fmt = os.path.join(
            '{usage}', 'DR{dialect}', '{sex}{speaker_id}',
            '{sentence_id}.{file_type}'
        )

        if not self.root.exists():
            raise RuntimeError(
                f'The root ({self.root}) directory does not exist.')

        # read the speakers information data
        spinfo_path = self.root / 'DOC' / 'SPKRINFO.TXT'
        with open(spinfo_path, 'r') as fp:
            txt = fp.read().split('\n')
            names = re.split(r'\s+', txt[37][1:])
            fp.seek(0)  # seek to top for pd.read_csv
            self.speakers_information = pd.read_csv(
                fp, sep=r'\s+', names=names,
                skiprows=lambda l: txt[l].startswith(';'),)

    def spkrinfo(self, code):
        """
        Return a speaker information

        Parameter
        ---------
        code : str
            speaker's ID (e.g. ABC0, ABW0, ...)
            The all speaker's ID are in timit.speakers_information['ID']

        Return
        ------
        dict
            speaker information
        """

        try:
            info = self.speakers_information.loc[
                self.speakers_information['ID'] == code].values[0]
        except IndexError:
            raise ValueError('speaker code is not exist: "%s"' % code)

        return {k: v for k, v in zip(self.speakers_information.columns, info)}

    def spkrloads(self, code):
        """
        Load the waveform data of a speaker

        Parameters
        ----------
        code : str
            speaker ID (e.g. ABC0, ABW0, ...)
            The all ID are in timit.speakers_information['ID']

        Yields
        ------
        numpy.ndarray, shape=(nsamples), dtype=np.float64
            The wavefom of a speaker
        """

        spinfo = self.spkrinfo(code)
        pattern = self.path_fmt.format(
            usage='TEST' if spinfo['Use'] == 'TST' else 'TRAIN',
            dialect=spinfo['DR'],
            sex=spinfo['Sex'],
            speaker_id=spinfo['ID'],
            sentence_id='*',
            file_type='WAV',
        )

        for path in self.root.glob(pattern):
            wave, info = sphread(path)
            assert info['sample_rate'] == self.sample_rate,\
                (f'The sampling rate `{info["sample_rate"]}` of loaded file '
                 f'"{path}" is not required.')
            yield wave.squeeze()


def sphread(filepath):
    """
    Read sph data

    Parameter
    ---------
    filepath : str or pathlib.Path
        data file path (.sph)

    Return
    ------
    numpy.ndarray, shape=(`nframes`, `nchannels`)
        wave data
    dict
        header information
    """

    with open(filepath, 'rb') as fp:
        # Read header info.
        if fp.read(8) != b'NIST_1A\n':
            raise OSError('file does not start with `NIST_1A` id')
        headsize = int(fp.read(8).strip())

        end = b'end_head'
        headinfo = {}
        # Read `headsize` without 16 bits already read
        for line in fp.read(headsize - 16).splitlines():
            if line.startswith(end):
                break

            line = line.decode('latin-1')
            key, size, contents = line.split(' ')
            if size[:2] == '-i':
                contents = int(contents)
            headinfo.update({key: contents})

        # Read wave data
        datasize = headinfo['sample_count'] * headinfo['sample_n_bytes']
        if headinfo['sample_n_bytes'] == 1:
            npformat = np.uint8
        elif headinfo['sample_n_bytes'] == 2:
            npformat = np.int16
        elif headinfo['sample_n_bytes'] == 4:
            npformat = np.int32
        else:
            raise RuntimeError('Unrecognized bytes count: {}'
                               .format(headinfo['sample_n_bytes']))

        data = np.frombuffer(fp.read(datasize), dtype=npformat)
        data = (data.reshape((-1, headinfo['channel_count']))
                / 2 ** (headinfo['sample_sig_bits'] - 1))

        return data, headinfo
