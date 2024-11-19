
def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '/media/BD_4t/zjh_all/zjhdata/iss/cityscapes'
    if name == 'cityscapes16':
        return '/media/BD_4t/zjh_all/zjhdata/iss/cityscapes'
    if name == 'gta':
        return '/media/BD_4t/zjh_all/zjhdata/iss/gta'
    if name == 'synthia':
        return '/media/BD_4t/zjh_all/zjhdata/iss/synthia'

def save_class_stats(out_dir, sample_class_stats, dataset_name):
    import os.path as osp
    import json
    if dataset_name == 'CityscapesDataset':
        sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)
