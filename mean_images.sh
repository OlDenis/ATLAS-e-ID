python mean_images.py --host_name=beluga --n_classes=0 --n_train=0 --valid_cut='(abs(sample["eta"]) > 0.0) & (abs(sample["eta"]) < 1.37) & (sample["pt"] > 5) & (sample["pt"] < 10)' --pt_region='5-10' --eta_region='0.0-1.37'
python mean_images.py --host_name=beluga --n_classes=0 --n_train=0 --valid_cut='(abs(sample["eta"]) > 0.0) & (abs(sample["eta"]) < 1.37) & (sample["pt"] > 30) & (sample["pt"] < 35)' --pt_region='30-35' --eta_region='0.0-1.37'
python mean_images.py --host_name=beluga --n_classes=0 --n_train=0 --valid_cut='(abs(sample["eta"]) > 0.0) & (abs(sample["eta"]) < 1.37) & (sample["pt"] > 90) & (sample["pt"] < 130)' --pt_region='90-130' --eta_region='0.0-1.37'
python mean_images.py --host_name=beluga --n_classes=0 --n_train=0 --valid_cut='(abs(sample["eta"]) > 1.6) & (abs(sample["eta"]) < 2.5) & (sample["pt"] > 5) & (sample["pt"] < 10)' --pt_region='5-10' --eta_region='1.6-2.5'
python mean_images.py --host_name=beluga --n_classes=0 --n_train=0 --valid_cut='(abs(sample["eta"]) > 1.6) & (abs(sample["eta"]) < 2.5) & (sample["pt"] > 30) & (sample["pt"] < 35)' --pt_region='30-35' --eta_region='1.6-2.5'
python mean_images.py --host_name=beluga --n_classes=0 --n_train=0 --valid_cut='(abs(sample["eta"]) > 1.6) & (abs(sample["eta"]) < 2.5) & (sample["pt"] > 90) & (sample["pt"] < 130)' --pt_region='90-130' --eta_region='1.6-2.5'
exit