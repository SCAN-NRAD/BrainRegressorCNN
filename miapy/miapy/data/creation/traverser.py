import abc
import typing as t

import numpy as np

from miapy.data import subjectfile as subj
import miapy.data.transformation as tfm
import miapy.data.conversion as conv
from . import callback as cb
from . import fileloader as load


class Traverser(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def traverse(self, files: list, loader: load.Load, callbacks: t.List[cb.Callback]=None,
                 transform: tfm.Transform=None):
        pass


def default_concat(data: t.List[np.ndarray]) -> np.ndarray:
    if len(data) == 1:
        return data[0]
    return np.stack(data, axis=-1)


class SubjectFileTraverser(Traverser):
    def traverse(self, subject_files: t.List[subj.SubjectFile], load=load.LoadDefault(), callback: cb.Callback=None,
                 transform: tfm.Transform=None, concat_fn=default_concat):
        if len(subject_files) == 0:
            raise ValueError('No files')
        if not isinstance(subject_files[0], subj.SubjectFile):
            raise ValueError('files must be of type {}'.format(subj.SubjectFile.__class__.__name__))
        if callback is None:
            raise ValueError('callback can not be None')

        callback_params = {'subject_files': subject_files}
        for category in subject_files[0].categories:
            callback_params.setdefault('categories', []).append(category)
            callback_params['{}_names'.format(category)] = self._get_names(subject_files, category)
        callback.on_start(callback_params)

        # looping over the subject files and calling callbacks
        for subject_index, subject_file in enumerate(subject_files):
            transform_params = {'subject_index': subject_index}
            for category in subject_file.categories:

                category_list = []
                category_property = None  # type: conv.ImageProperties
                for id_, file_path in subject_file.categories[category].entries.items():
                    np_data, data_property = load(file_path, id_, category)
                    category_list.append(np_data)
                    if category_property is None:  # only required once
                        category_property = data_property

                category_data = concat_fn(category_list)
                transform_params[category] = category_data
                transform_params['{}_properties'.format(category)] = category_property

            if transform:
                transform_params = transform(transform_params)

            callback.on_subject({**transform_params, **callback_params})

        callback.on_end(callback_params)




        # # get the image, label, and supplementary names
        # image_names = self._get_image_names(subject_files)
        #
        # label_image_names = self._get_label_image_names(subject_files)
        # has_labels = len(label_image_names) > 0
        #
        # supplementary_names = self._get_supplementary_names(subject_files)
        # has_supplemenatries = len(supplementary_names) > 0
        #
        # callback_params = { 'has_labels': has_labels,
        #                    'has_supplementaries': has_supplemenatries, 'image_names': image_names}
        # if has_labels:
        #     callback_params['label_names'] = label_image_names
        # if has_supplemenatries:
        #     callback_params['supplementary_names'] = supplementary_names

        # callback.on_start(callback_params)
        #
        # # looping over the subject files and calling callbacks
        # for subject_index, subject_file in enumerate(subject_files):
        #
        #     images_list = []
        #     image_property = None  # type: conv.ImageProperties
        #     for id_, image_path in subject_file.images.items():
        #         np_img, img_property = load(image_path, id_, subj.FileType.IMAGE)
        #         images_list.append(np_img)
        #         if image_property is None:  # only required once
        #             image_property = img_property
        #
        #     images = concat_fn(images_list)
        #     transform_params = {'subject_index': subject_index, 'images': images, 'image_properties': image_property}
        #
        #     if has_labels:
        #         label_image_list = []
        #         label_image_property = None  # type: conv.ImageProperties
        #         for id_, label_image_path in subject_file.label_images.items():
        #             np_label_img, label_property = load(label_image_path, id_, subj.FileType.LABEL)
        #             label_image_list.append(np_label_img)
        #             if label_image_property is None:
        #                 label_image_property = label_property
        #
        #         label_images = concat_fn(label_image_list)
        #         transform_params['labels'] = label_images
        #         transform_params['label_properties'] = label_image_property
        #
        #     if has_supplemenatries:
        #         supplementaries = {}
        #         supplementary_properties = {}
        #         for id_, supplementary_path in subject_file.supplementaries.items():
        #             props_key = '{}_properties'.format(id_)
        #             supplementaries[id_], supplementary_properties[props_key] = load(supplementary_path, id_,
        #                                                                              subj.FileType.SUPPLEMENTARY)
        #
        #         transform_params = {**transform_params, 'supplementaries': supplementaries,
        #                             'supplementary_properties': supplementary_properties}
        #
        #     if transform:
        #         transform_params = transform(transform_params)
        #
        #     callback.on_subject({**transform_params, **callback_params})
        #
        # callback.on_end(callback_params)

    @staticmethod
    def _get_names(subject_files: t.List[subj.SubjectFile], category: str) -> list:
        names = subject_files[0].categories[category].entries.keys()
        if not all(s.categories[category].entries.keys() == names for s in subject_files):
            raise ValueError('Inconsistent {} identifiers in the subject list'.format(category))
        return list(names)


    # @staticmethod
    # def _get_image_names(subject_files: t.List[subj.SubjectFile]) -> list:
    #     """Gets the image names from a list of subjects and checks the consistency of the image names.
    #
    #     For consistency, each subject must have the same image identifiers.
    #
    #     Args:
    #         subject_files (list of SubjectFile): The subject files.
    #
    #     Returns:
    #         list(str): List of image names.
    #     """
    #     images_names = subject_files[0].images.keys()
    #     if not all(s.images.keys() == images_names for s in subject_files):
    #         raise ValueError('Inconsistent image identifiers in the subject list')
    #     return list(images_names)
    #
    # @staticmethod
    # def _get_label_image_names(subject_files: t.List[subj.SubjectFile]) -> list:
    #     """Gets the label names from a list of subjects and checks the consistency of the label names.
    #
    #     For consistency, each subject must have the same label identifiers.
    #
    #     Args:
    #         subject_files (list of SubjectFile): The subject files.
    #
    #     Returns:
    #         list(str): List of label names (can be empty).
    #     """
    #     if subject_files[0].label_images is None:
    #         if not all(s.label_images is None for s in subject_files):
    #             raise ValueError('Inconsistent label image identifiers in the subject list')
    #         return []
    #     label_image_names = subject_files[0].label_images.keys()
    #     if not all(s.label_images.keys() == label_image_names for s in subject_files):
    #         raise ValueError('Inconsistent label image identifiers in the subject list')
    #     return list(label_image_names)
    #
    # @staticmethod
    # def _get_supplementary_names(subject_files: t.List[subj.SubjectFile]) -> list:
    #     """Gets the supplementary names from a list of subjects and checks the consistency of the supplementary names.
    #
    #     For consistency, each subject must have the same supplementary identifiers.
    #
    #     Args:
    #         subject_files (list of SubjectFile): The subject files.
    #
    #     Returns:
    #         list(str): List of supplementary names (can be empty).
    #     """
    #     if subject_files[0].supplementaries is None:
    #         if not all(s.supplementaries is None for s in subject_files):
    #             raise ValueError('Inconsistent supplementary identifiers in the subject list')
    #         return []
    #     supplementary_names = subject_files[0].supplementaries.keys()
    #     if not all(s.supplementaries.keys() == supplementary_names for s in subject_files):
    #         raise ValueError('Inconsistent supplementary identifiers in the subject list')
    #     return list(supplementary_names)
