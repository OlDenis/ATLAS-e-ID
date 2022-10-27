#python presampler.py --mixing=ON --n_files=10
echo Reading presampler.sh
python presampler.py --n_e=500e6 --merging='ON' --data_path="/scratch/odenis/JF17/test/e-ID"
#python presampler.py --n_e=500e6 --n_tasks=20 --sampling='OFF' --merging='ON' --data_path="/scratch/odenis/JF17/test/e-ID"
#--eta_region=0-2.5
#python presampler.py --n_e=500e6 --n_tasks=20 --eta_region=1.6-2.5 --sampling=ON --merging=OFF
#python presampler_Olivier.py --n_e=500e6 --n_tasks=20 --sampling=ON --merging=ON --eta_region=0-2.5 --indir=/lcg/storage20/atlas/denis/hdf5/JF17/11151/MC/electron/e-ID
