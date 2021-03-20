import os
import glob
import csv
import argparse
import numpy as np

import data.storage as data_storage
import data.preprocess as data_preproc
import miapy.data.transformation as miapy_tfm


def assert_exists(file: str):
    if not os.path.exists(file):
        print('Error: {} not found'.format(file))
        exit(1)


class DTypeTransform(miapy_tfm.Transform):

    def __init__(self, dtype, entries=('images',)) -> None:
        self.dtype = dtype
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            np_entry = sample[entry]
            sample[entry] = np_entry.astype(self.dtype)

        return sample


# Assume subject name is in first column. All remaining columns will be added as meta data
def read_meta(csv_file: str) -> dict:
    params = None
    meta = dict()

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            if params is None:
                params = row
            else:
                meta[row[0]] = {params[i]: float(row[i]) for i in range(1, len(params))}

    return meta


def main(hdf_file: str, intensity_rescale_max: int, center: bool, clip_intensity: bool, brain_pattern: str, csv_file: str, directories: list):
    transformers = []

    if clip_intensity:
        print('Applying clip negative intensities')
        transformers.append(data_preproc.ClipNegativeTransform())

    if center:
        print('Applying center centroid')
        transformers.append(data_preproc.CenterCentroidTransform())

    if intensity_rescale_max:
        print('Applying intensity rescale {}'.format(intensity_rescale_max))
        transformers.append(miapy_tfm.IntensityRescale(0, intensity_rescale_max))
        if intensity_rescale_max <= 255:
            transformers.append(DTypeTransform(np.uint8))
        else:
            transformers.append(DTypeTransform(np.float16))

    transformer = miapy_tfm.ComposeTransform(transformers)

    meta_dict = read_meta(csv_file) if csv_file else None
    subjects = []
    for directory in directories:
        assert_exists(directory)
        for subject_dir in glob.glob(os.path.join(directory, '*')):
            subject_id = os.path.basename(subject_dir)
            if meta_dict and subject_id not in meta_dict:
                continue
                
            brain_file = os.path.join(subject_dir, brain_pattern)
            stats_dir = os.path.join(subject_dir, 'stats')
            assert_exists(brain_file)
            assert_exists(stats_dir)

            subjects.append(data_storage.Subject(subject_id,
                                    {data_storage.FileTypes.BRAIN_MRI: brain_file,
                                     data_storage.FileTypes.MORPHOMETRY_STATS: stats_dir}))

    if os.path.exists(hdf_file):
        print('Overriding existing {}'.format(hdf_file))
        os.remove(hdf_file)

    store = data_storage.DataStore(hdf_file)
    store.import_data(subjects, intensity_rescale_max, transformer, meta_dict)

    print('{} subjects imported to {}'.format(len(subjects), hdf_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset (.h5) from one or more directories with subjects.')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='dataset.h5',
        help='Path to hd5 file.'
    )

    parser.add_argument(
        '--intensity_rescale_max',
        type=int,
        help='Apply intensity rescale(0, <max>) to input data and convert to uint8 (<=255) or float16, e.g. 255 or 4095'
    )

    parser.add_argument(
        '--center',
        type=bool,
        help='Center the centroid.'
    )

    parser.add_argument(
        '--clip_intensity',
        type=bool,
        default=True,
        help='Clip negative voxel intensities to zero (e.g. ADNI).'
    )

    parser.add_argument(
        '--brain_file',
        type=str,
        required=True,
        help='Path to brain image relative to subject dir (e.g. mri/brain.nii.gz or mri/T1w_noskull_normsize.nii.gz)'
    )

    parser.add_argument(
        '--meta_csv',
        type=str,
        help='Optional csv file with metadata.'
    )

    parser.add_argument(
        'directories',
        type=str,
        metavar='dir',
        nargs='+',
        help='Directories to import subjects from.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.intensity_rescale_max, args.center, args.clip_intensity, args.brain_file, args.meta_csv, args.directories)
