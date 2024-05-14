import os
import numpy as np

def assemble_compare_measures(results_folder, file_prefix, prefix_len = 3, gap_normalizer = 1.0, filter_out = None, filter_by = None):
    exp_files = []
    for f in os.listdir(results_folder):
        if not filter_out is None and filter_out in f:
            continue
        if not filter_by is None and not filter_by in f:
            continue
        if ".txt" in f and file_prefix in f:
            exp_files.append(f)

    tracial_measures = []
    meig_measures = []
    trace = []
    fisher_rao = []
    norm = []
    gaps = []
    pacbayes_flat = []
    pacbayes_orig = []
    labels = []
    for f in exp_files:
        labels.append("_".join(f.split("_")[prefix_len:]))
        for l in open(os.path.join(results_folder, f), "r").readlines():
            if l[0] != "#" and len(l.split("\t")) > 3:
                row = l.replace("\n", "").replace("\r", "").split("\t")
                norm.append(float(row[2]))
                trace.append(float(row[4]))
                fisher_rao.append(float(row[5]))
                pacbayes_flat.append(float(row[6]))
                pacbayes_orig.append(float(row[7]))
                tracial_measures.append(float(row[8]))
                meig_measures.append(float(row[9]))
                gaps.append((float(row[1]) - float(row[0]))/gap_normalizer)
    print(len(exp_files))
    ind = np.array(gaps).argsort()[3:]
    tracial_measures = np.array(tracial_measures)[ind]
    meig_measures = np.array(meig_measures)[ind]
    trace = np.array(trace)[ind]
    fisher_rao = np.array(fisher_rao)[ind]
    pacbayes_flat = np.array(pacbayes_flat)[ind]
    pacbayes_orig = np.array(pacbayes_orig)[ind]
    norm = np.array(norm)[ind]
    gaps = np.array(gaps)[ind]
    labels = np.array(labels)[ind]
    return tracial_measures, meig_measures, trace, fisher_rao, pacbayes_flat, pacbayes_orig, norm, gaps, labels
