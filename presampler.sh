#python presampler.py --mixing=ON --n_files=10
datatype=$1

if [[ $datatype=data ]]; then
	echo Presampling data
	python presampler.py --merging=ON
elif [[ $datatype=mc ]]; then
	echo presampling mc
	python presampler.py --merging=ON --data_path='../hdf5/MC/electron/423300' 
else 
	echo datatype must be either 'data' or 'mc'
fi
