"""
Write the commands to run the mean_images productions for LF CR.
jan 20 2022 : first commit

Olivier Denis
"""

eta_regions = [("0.0","1.37"), ("1.6","2.5")]
pt_regions = [("5","20"), ("30","35"), ("90","130")]

command_list = []
for eta in eta_regions:
    for pt in pt_regions:
        eta_region = eta[0] + "-" + eta[1]
        CUTS = '(abs(sample["eta"]) > {0}) & (abs(sample["eta"]) < {1}) \\
                & (sample["pt"] > {2}) & (sample["pt"] < {3})'.format(*eta, *pt)
        COMMAND = """python classifier.py --host_name=beluga --n_classes=0\\
                --n_train=0 --valid_cut={} --eta_region={}""".format(CUTS, eta_region)
        command_list.append(COMMAND)

command_list.append("exit")
TO_WRITE = "\n".join(command_list)
#print(TO_WRITE)

with open("mean_images.sh", 'w') as f:
    f.write(TO_WRITE)
