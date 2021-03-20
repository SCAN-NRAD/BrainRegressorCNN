import enum
import typing
import re
import os

import SimpleITK as sitk
import numpy as np

import miapy.data as miapy_data
import miapy.data.conversion as miapy_conv
import miapy.data.creation as miapy_crt
import miapy.data.extraction as miapy_extr
import miapy.data.indexexpression as miapy_expr
import miapy.data.transformation as miapy_tfm
import miapy.data.creation.fileloader as miapy_fileload
import miapy.data.creation.writer as miapy_writer
import miapy.data.creation.callback as miapy_callback


STORE_META_MORPHOMETRY_COLUMNS = 'meta/MORPHOMETRY_COLUMNS'
STORE_META_INTENSITY_RESCALE_MAX = 'meta/INTENSITY_RESCALE_MAX'
STORE_IMAGES = 'images'
STORE_MORPHOMETRICS = 'morphometrics'
STORE_META_CENTROID_TRANSFORM = 'meta/transform/centroids'
STORE_DEMOGRAPHIC_AGE = 'demographics/age'
STORE_DEMOGRAPHIC_SEX = 'demographics/sex'


class FileTypes(enum.Enum):
    BRAIN_MRI = 1  # mri/brain.nii.gz
    MORPHOMETRY_STATS = 2  # stats/{aseg.stats, lh.aparc.stats, rh.aparc.stats}


class Subject(miapy_data.SubjectFile):
    def __init__(self, subject: str, files: dict):
        super().__init__(subject,
                         images={FileTypes.BRAIN_MRI.name: files[FileTypes.BRAIN_MRI]},
                         morphometrics={FileTypes.MORPHOMETRY_STATS.name: files[FileTypes.MORPHOMETRY_STATS]})


class FreeSurferDataLoader(miapy_fileload.Load):
    def __init__(self, meta_dict: dict = None):
        self.morphometry_column_names = []
        self.meta_dict = meta_dict

    def __call__(self, file_path: str, id_: str, category: str) -> typing.Tuple[
                    np.ndarray, typing.Union[miapy_conv.ImageProperties, None]]:

        if id_ == FileTypes.BRAIN_MRI.name:
            img = sitk.ReadImage(file_path)
            return sitk.GetArrayFromImage(img), miapy_conv.ImageProperties(img)

        if id_ == FileTypes.MORPHOMETRY_STATS.name:
            metrics = {}
            subject_name = file_path.split('/')[-2]
            print(subject_name)
            # aseg.stats
            with open(os.path.join(file_path, 'aseg.stats'), 'r') as f:
                for line in f.readlines():
                    # Parse header lines
                    # Example: '# Measure TotalGray, TotalGrayVol, Total gray matter volume, 666685.704205, mm^3'
                    matcher = re.match('# Measure (\w+), \w+, [\w\s]+, ([0-9.]+),.*', line)
                    if matcher:
                        key = matcher.group(1)
                        value = float(matcher.group(2))
                        metrics[key] = value

                    # Parse body lines
                    # Example: ' 13  18  1442   1442.0  Left-Amygdala   74.0193  6.1854  48.0000  94.0000  46.0000'
                    matcher = re.match('\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+([\w\-_]+).*', line)
                    if matcher:
                        key = matcher.group(2)
                        value = float(matcher.group(1))
                        metrics[key] = value

            # sum up Corpus Callosum
            metrics['CC_sum'] = sum([v for k, v in metrics.items() if k.startswith('CC_')])

            # old freesurfer versions only
            if 'IntraCranialVol' in metrics:
                metrics['EstimatedTotalIntraCranialVol'] = metrics.pop('IntraCranialVol')
                print('WARN: ' + file_path + '. Using IntraCranialVol')

            # lh.aparc.stats, rh.aparc.stats
            for hemi in ['lh', 'rh']:
                with open(os.path.join(file_path, '{}.aparc.stats'.format(hemi)), 'r') as f:
                    for line in f.readlines():
                        # Parse body lines
                        # StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
                        # Example: 'bankssts  1507  1003  2841  2.494 0.553  0.146  0.076  29  4.7'
                        matcher = re.match(
                            '(\w+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+\s+([0-9.]+).*', line)
                        if matcher:
                            key = matcher.group(1)
                            thick_avg = float(matcher.group(2))
                            mean_curv = float(matcher.group(3))
                            metrics['{}-{}-ThickAvg'.format(hemi, key)] = thick_avg
                            metrics['{}-{}-MeanCurv'.format(hemi, key)] = mean_curv

            # add additional meta data from dict (read from csv)
            if self.meta_dict: #and subject_name in self.meta_dict:
                meta = self.meta_dict[subject_name]
                for key in meta:
                    metrics[key] = meta[key]

            if len(self.morphometry_column_names) == 0:
                self.morphometry_column_names = np.array([key for key in metrics.keys()])

            # if a structure has no voxels for a particular subject, entry is not available
            # in aparc.stats (e.g. entorhinal can be 0)
            for col in self.morphometry_column_names:
                if col not in metrics:
                    print('No {} found for {}. Assuming 0.0'.format(col, subject_name))
                    metrics[col] = 0.0

            return np.array([metrics[value] for value in self.morphometry_column_names]), None


# extract centroid transformation and write to metadata
class WriteImageCentroidTransformCallback(miapy_callback.Callback):

    def __init__(self, writer: miapy_writer.Writer) -> None:
        self.writer = writer
        self.new_subject = False

    def on_start(self, params: dict):
        subject_count = len(params['subject_files'])
        self.writer.reserve(STORE_META_CENTROID_TRANSFORM, (subject_count, 3), dtype=np.int16)

    def on_subject(self, params: dict):
        subject_index = params['subject_index']
        centroid_transform = params['centroid_transform']
        self.writer.fill(STORE_META_CENTROID_TRANSFORM, centroid_transform, miapy_expr.IndexExpression(subject_index))


class CentroidTransformExtractor(miapy_extr.Extractor):

    def __init__(self) -> None:
        super().__init__()

    def extract(self, reader: miapy_extr.reader.Reader, params: dict, extracted: dict) -> None:
        subject_index_expr = miapy_expr.IndexExpression(params['subject_index'])

        extracted['centroid_transform'] = reader.read(STORE_META_CENTROID_TRANSFORM, subject_index_expr)


# write demographics metadata (age/sex)
class WriteDemographicsCallback(miapy_callback.Callback):

    def __init__(self, writer: miapy_writer.Writer, meta_dict: dict = None) -> None:
        self.writer = writer
        self.new_subject = False
        self.meta_dict = meta_dict

    def on_start(self, params: dict):
        subject_count = len(params['subject_files'])
        self.writer.reserve(STORE_DEMOGRAPHIC_AGE, (subject_count, 1), dtype=np.uint8)
        self.writer.reserve(STORE_DEMOGRAPHIC_SEX, (subject_count, 1), dtype=np.uint8)

    def on_subject(self, params: dict):
        subject_files = params['subject_files']
        subject_index = params['subject_index']
        subject = subject_files[subject_index].subject

        age = 0
        sex = 0
        if self.meta_dict:
            meta = self.meta_dict[subject]
            age = meta['AGE']
            sex = meta['SEX']

        self.writer.fill(STORE_DEMOGRAPHIC_AGE, age, miapy_expr.IndexExpression(subject_index))
        self.writer.fill(STORE_DEMOGRAPHIC_SEX, sex, miapy_expr.IndexExpression(subject_index))


class DemographicsExtractor(miapy_extr.Extractor):

    def extract(self, reader: miapy_extr.reader.Reader, params: dict, extracted: dict) -> None:
        subject_index_expr = miapy_expr.IndexExpression(params['subject_index'])

        extracted[STORE_DEMOGRAPHIC_AGE] = reader.read(STORE_DEMOGRAPHIC_AGE, subject_index_expr)
        extracted[STORE_DEMOGRAPHIC_SEX] = reader.read(STORE_DEMOGRAPHIC_SEX, subject_index_expr)


class DataStore:
    def __init__(self, hdf_file: str, data_transform: miapy_tfm.Transform = None):
        self.hdf_file = hdf_file
        self._dataset = None
        self._data_transform = data_transform

    def __del__(self):
        if self._dataset is not None:
            self._dataset.close_reader()

    def import_data(self, subjects: typing.List[Subject], intensity_max: int, input_transform: miapy_tfm = None, meta_dict: dict = None):
        with miapy_crt.get_writer(self.hdf_file) as writer:
            callbacks = miapy_crt.get_default_callbacks(writer)
            callbacks.callbacks.append(WriteImageCentroidTransformCallback(writer))
            callbacks.callbacks.append(WriteDemographicsCallback(writer, meta_dict))
            traverser = miapy_crt.SubjectFileTraverser()
            loader = FreeSurferDataLoader(meta_dict)
            traverser.traverse(subjects, callback=callbacks, load=loader, transform=input_transform)
            writer.write(STORE_META_MORPHOMETRY_COLUMNS, loader.morphometry_column_names, dtype='str')
            writer.write(STORE_META_INTENSITY_RESCALE_MAX, intensity_max, dtype='int')

    @property
    def dataset(self) -> miapy_extr.ParameterizableDataset:
        if self._dataset is None:
            self._dataset = miapy_extr.ParameterizableDataset(
                self.hdf_file,
                None,
                miapy_extr.SubjectExtractor(),
                self._data_transform)

        return self._dataset

    def set_transforms_enabled(self, enabled: bool):
        if enabled:
            self._dataset.set_transform(self._data_transform)
        else:
            self._dataset.set_transform(None)

    @staticmethod
    def collate_batch(batch) -> dict:
        # batch is a list of dicts -> change to dict of lists
        return dict(zip(batch[0], zip(*[d.values() for d in batch])))

    def get_all_metrics(self) -> typing.Tuple[np.ndarray, list]:
        """
        Get the metrics from all subjects and the corresponding column names
        
        :return: (metrics, column_names)
        """

        metrics = [self.dataset.direct_extract(
            miapy_extr.SelectiveDataExtractor(category=STORE_MORPHOMETRICS), idx) for idx in range(len(self.dataset))]
        column_names = self.dataset.reader.read(STORE_META_MORPHOMETRY_COLUMNS)

        return np.stack(self.collate_batch(metrics)[STORE_MORPHOMETRICS], axis=0), column_names

    def get_intensity_scale_max(self) -> int:
        return self.dataset.reader.read(STORE_META_INTENSITY_RESCALE_MAX)

    def get_loader(self, batch_size: int, subject_ids: typing.List[str], num_workers: int):
        sampler = miapy_extr.SubsetRandomSampler(
            miapy_extr.select_indices(self.dataset, miapy_extr.SubjectSelection(subject_ids)))

        return miapy_extr.DataLoader(self.dataset,
                                     batch_size,
                                     sampler=sampler,
                                     collate_fn=self.collate_batch,
                                     num_workers=num_workers)



def demographics2numbers(study_str: str, age_str: str, sex_str: str) -> (str, int, int):
    age = 0
    sex = 0

    if age_str != 'n/a':
        age = int(age_str)
    if sex_str is None or sex_str == 'n/a':
        sex = 0
    elif sex_str.lower() == 'm':
        sex = 1
    elif sex_str.lower() == 'f':
        sex = 2

    return study_str, age, sex
