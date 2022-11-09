import numpy as np
import pandas as pd
import ipdb

nr3d_file = "/mnt/petrelfs/zhaolin/files/Asrclin/data/scannet/infos/nr3d.csv"
nr3d_data = pd.read_csv(nr3d_file)

sr3d_file = "/mnt/petrelfs/zhaolin/files/Asrclin/data/scannet/infos/Sr3D/sr3d.csv"
sr3d_data = pd.read_csv(sr3d_file)

sr3d_data['instance_type'].value_counts()

def count_nr3d(nr3d):
    target_count = nr3d['target_id'].value_counts()

    print()


if __name__ == '__main__':
    count_nr3d(nr3d=nr3d_data)
    print()
