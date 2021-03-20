#!/bin/bash

if [ $# -lt 1 ] ; then
	echo "Usage: $0 FS_RESULTS_DIR"
	exit 1
fi

# subjects directory with FreeSurfer results to import into dataset
FS_RESULTS_DIR=$1

export PYTHONPATH=../miapy:../Python

python main_create_dataset.py \
	--intensity_rescale_max 4095 \
	--center 1 \
	--brain_file mri/T1w_noskull_normsize.nii.gz \
	--hdf_file ../dataset.h5 \
	${FS_RESULTS_DIR}

