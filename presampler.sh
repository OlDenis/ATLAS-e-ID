#python presampler.py --mixing=ON --n_files=10
echo Reading presampler.sh
python presampler.py --n_e=500e6 --merging='ON' --data_path="/scratch/odenis/JF17/test/e-ID"
#python presampler.py --n_e=500e6 --n_tasks=20 --sampling='OFF' --merging='ON' --data_path="/scratch/odenis/JF17/test/e-ID"
#--eta_region=0-2.5
