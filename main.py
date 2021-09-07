from configs import ArcfaceConfig, BaseConfig, CenterlossConfig, SpherefaceConfig, DefaultDataConfig, VggFace2Config
from data import Balancedface, BalancedfaceSingle, BalancedfacePair, BalancedfaceMulti, RFW, getRFWList
from torch.utils import data
from trainers import ArcfaceTrainer, CenterlossTrainer, VggFace2Trainer

import argparse
import json
import numpy as np
import os
import time
import torch

methods_lookup_dict = {
    'arcface': {
        'config': ArcfaceConfig,
        'trainer': ArcfaceTrainer,
    },
    'centerloss': {
        'config': CenterlossConfig,
        'trainer': CenterlossTrainer,
    },
    'sphereface': {
        'config': SpherefaceConfig,
        'trainer': ArcfaceTrainer,
    },
    'vggface2': {
        'config': VggFace2Config,
        'trainer': VggFace2Trainer,
    },
}

usr_config_lookup_dict = {
    'default': DefaultDataConfig
}

methods_list_str = ', '.join(list(methods_lookup_dict.keys()))
usr_list_str = ', '.join(list(usr_config_lookup_dict.keys()))


def get_method_trainer(method_str):
    for method, method_items in methods_lookup_dict.items():
        if method_str == method:
            return method_items['trainer']
    raise ValueError(f'Unsupported method: {method_str}. Please use one of: {methods_list_str}')


def get_method_config(method_str):
    for method, method_items in methods_lookup_dict.items():
        if method_str == method:
            return method_items['config']().get_config()
    raise ValueError(f'Unsupported method: {method_str}. Please use one of: {methods_list_str}')


def get_data_config(usr_str):
    for usr, DataConfig in usr_config_lookup_dict.items():
        if usr_str == usr:
            return DataConfig().get_config()
    raise ValueError(f'Unsupported user: {usr_str}. Please use one of: {usr_list_str}')


def get_base_config():
    return BaseConfig().get_config()


def merge_configs(configs):
    merged_config = {}
    for config in configs:
        for key, value in config.items():
            merged_config[key] = value

    return merged_config


def get_config(method, usr_config):
    base_config = get_base_config()
    method_config = get_method_config(method)
    data_config = get_data_config(usr_config)

    # IMPORTANT: this is not intended usage, but items in later configs have priority over 
    #            items in earlier configs, basically if there are key conflicts the rightmost
    #            config in the list with the key will determine the value in the merged config
    config = merge_configs([base_config, method_config, data_config])

    return config


def get_trainset(opt):
    if opt['experiment_name'] == 'single_race':
        if opt['noise_mix']:
            return BalancedfaceSingle(opt['train_root'], opt['train_race'], input_shape=opt['input_shape'],
                    noise_mixing=opt['noise_mix'], noise=opt['data_noise'])
        return BalancedfaceSingle(opt['train_root'], opt['train_race'], input_shape=opt['input_shape'])
    elif opt['experiment_name'] == 'balanced_face':
        return Balancedface(opt['train_root'], input_shape=opt['input_shape'])
    elif opt['experiment_name'] == 'race_pairs':
        return BalancedfacePair(opt['train_root'], opt['train_race'], opt['train_race_b'], opt['percent_primary'])
    elif 'all_races' in opt['experiment_name']:
        file_name = opt['identities_per_race_path'] + '_'.join([str(opt['num_classes']), str(opt['trial_num'])]) + '.pth.tar'
        if 'baseline' in opt['experiment_name'] or opt['num_classes'] == 10000 or opt['num_classes'] == 4000:
            opt['identities_per_race'] = {
                'african': int(opt['num_classes'] / 4),
                'asian': int(opt['num_classes'] / 4),
                'caucasian': int(opt['num_classes'] / 4),
                'indian': int(opt['num_classes'] / 4)
            }
        elif not os.path.isfile(file_name):
            raise ValueError(f'Unsupported experiment, no identities_per_race found: {opt["identities_per_race_path"]}')
        else:
            opt['identities_per_race'] = torch.load(file_name)
        return BalancedfaceMulti(opt['train_root'], opt['identities_per_race'])
    elif opt['experiment_name'] == 'images_vs_identities':
        opt['identities_per_race'] = {
            'african': 2500,
            'asian': 2500,
            'caucasian': 2500,
            'indian': 2500
        }
        opt['identities_per_race'][opt['experiment_race'].lower()] += (opt['num_classes'] - 10000)
        opt['images_per_identity'] = {
            'african': 10,
            'asian': 10,
            'caucasian': 10,
            'indian': 10
        }
        opt['images_per_identity'][opt['experiment_race'].lower()] += (opt['num_extra_images'])
        return BalancedfaceMulti(opt['train_root'], opt['identities_per_race'], images_per_identity=opt['images_per_identity'])
    elif opt['experiment_name'] == 'race_simplex':
        with open(opt['simplex_file_path'], 'r') as json_file:
            json_obj = json.load(json_file)
            races = json_obj['order']
            cur_simplex_point = json_obj['distributions'][str(opt['simplex_max'])][opt['simplex_point_id']]
            opt['identities_per_race'] = {}
            for i, race in enumerate(races):
                opt['identities_per_race'][race.lower()] = int(cur_simplex_point[i])
            print('identities_per_race', opt['identities_per_race'])
            if sum(opt['identities_per_race'].values()) != opt['num_classes']:
                raise ValueError(f'Something really bad, check code')
            return BalancedfaceMulti(opt['train_root'], opt['identities_per_race'])
    raise ValueError(f'Unsupported experiment: {opt["experiment_name"]}')


def get_testset(opt):
    return RFW(opt['test_root'], opt['test_race'], input_shape=opt['input_shape'])


def compute_weighted_neighbors(test_features):
    SAMPLES_PER_RACE = 1000
    NUM_NEIGHBORS = 20

    num_images = 0
    features_samples = []
    features_races = []
    last_idxs = []
    for index, features_map in enumerate(test_features):
        features_array = np.zeros((len(features_map.keys()), 512))
        for i, features in enumerate(features_map.values()):
            features_array[i] = features

        potential_indices = np.arange(features_array.shape[0])
        indices = np.random.choice(potential_indices, SAMPLES_PER_RACE, replace=False)
        features_sample = features_array[indices]

        num_images += features_sample.shape[0]
        last_idxs.append(num_images)
        features_samples.extend(features_sample)

        features_races.extend([index] * SAMPLES_PER_RACE)

    features_matrix = np.array(features_samples)
    features_races = np.array(features_races)

    start_time = time.time()
    d = features_matrix @ features_matrix.T
    norm = (features_matrix * features_matrix).sum(1, keepdims=True) ** .5
    distances_matrix = 1 - d / norm / norm.T
    print('Distances matrix computed in {} seconds'.format(str(time.time() - start_time)))

    race_weights_list = []
    for i in range(distances_matrix.shape[0]):  
        row_without_self = np.concatenate((distances_matrix[i,:i], distances_matrix[i,i+1:]))
        nearest_neighbors = np.argpartition(row_without_self, NUM_NEIGHBORS)[:NUM_NEIGHBORS]
        race_votes = features_races[nearest_neighbors]
        vote_weights = 1 - row_without_self[nearest_neighbors]
        race_weights = np.array([0.0, 0.0, 0.0, 0.0])
        for j in range(NUM_NEIGHBORS):
            race_weights[race_votes[j]] += vote_weights[j]
        race_weights_list.append(race_weights)

    print('----------')
    print('Cluster membership by race')
    print('African Asian Caucasian Indian')
    race_weights_matrix = np.array(race_weights_list)
    race_clusters = np.argmax(race_weights_matrix, axis=1)
    first_idx = 0
    all_cluster_counts = np.array([0, 0, 0, 0])
    for last_idx in last_idxs:
        clusters = race_clusters[first_idx:last_idx]
        race_cluster_counts = np.array([0, 0, 0, 0])
        for cluster in clusters:
            race_cluster_counts[cluster] += 1
        print(race_cluster_counts)
        all_cluster_counts += race_cluster_counts
        first_idx = last_idx
    print(all_cluster_counts)
    print('----------')
    print('\n')

    return all_cluster_counts
    

def save_new_distribution(opt, weighted_race_neighbors):
    print(opt['identities_per_race'])
    races = ['african', 'asian', 'caucasian', 'indian']
    budget = opt['step_size']

    persons_to_add_list = weighted_race_neighbors / weighted_race_neighbors.sum() * budget
    persons_added = 0
    counter = 0
    new_images_per_race = {}
    for race, cur_num_persons in opt['identities_per_race'].items():
        persons_to_add = int(persons_to_add_list[races.index(race)])
        if persons_added + persons_to_add > budget:
            persons_to_add -= 1
        elif counter == 3 and persons_added + persons_to_add < budget:
            persons_to_add += (budget - (persons_added + persons_to_add))
        new_images_per_race[race] = cur_num_persons + persons_to_add
        persons_added += persons_to_add
        counter += 1

    print(new_images_per_race)
    if not os.path.isdir(opt['identities_per_race_path']):
        os.makedirs(opt['identities_per_race_path'])
    torch.save(new_images_per_race, os.path.join(opt['identities_per_race_path'], '_'.join([str(opt['num_classes'] + persons_added), str(opt['trial_num'])]) + '.pth.tar'))


if __name__ == '__main__':
    # TODO designate a 'default' experiment and set defaults accordingly
    parser = argparse.ArgumentParser(description='Command line arguments for running experiments')
    parser.add_argument('--method', default='arcface', type=str, help=f'Training method; one of: {methods_list_str}')
    parser.add_argument('--usr-config', type=str, required=True,  help=f'User specific data configurations; one of: {usr_list_str}')
    parser.add_argument('--train-race', default='African', type=str, help='set race to train')
    parser.add_argument('--train-race-b', default='Caucasian', type=str, help='set other race to train (for race pairs experiment)')
    parser.add_argument('--test-race', default='African', type=str, help='set race to test')
    parser.add_argument('--trial-num', default=0, type=int, help='which trial is this')
    parser.add_argument('--percent-primary', default=0, type=int, help='set percentage of race a (for race pairs)')
    parser.add_argument('--data-noise', default=0.25, type=float, help='noise level in training data')
    parser.add_argument('--experiment-name', default='single_race', type=str, help='set experiment to run')
    parser.add_argument('--train', default=False, dest='train', action='store_true', help='flag to include training')
    parser.add_argument('--test', default=False, dest='test', action='store_true', help='flag to include evaluation')
    parser.add_argument('--gen-ims-per-race', default=False, dest='gen_ims_per_race', action='store_true', help='flag to include generating next balance')
    parser.add_argument('--num-classes', default=7000, type=int, help='set number of classes for train')
    parser.add_argument('--step-size', default=500, type=int, help='set number of classes to add each iteration')
    parser.add_argument('--num-extra-images', default=0, type=int, help='set number of extra images for selected race in identity vs. image experiment')
    parser.add_argument('--experiment-race', default='African', type=str, help='set selected race to receive additional images in identity vs. image experiment')
    parser.add_argument('--simplex-max', default=100, type=int, help='sum of entries for given point on 4-d race simplex')
    parser.add_argument('--simplex-point-id', default=0, type=int, help='id of point within given simplex, which provides race distribution')

    args = parser.parse_args()
    print(f'Args {args}')

    opt = get_config(args.method, args.usr_config)
    opt['trial_num'] = args.trial_num
    opt['train_race'] = args.train_race
    opt['train_race_b'] = args.train_race_b
    opt['percent_primary'] = args.percent_primary
    opt['experiment_name'] = args.experiment_name
    opt['data_noise'] = args.data_noise
    opt['train'] = args.train
    opt['test'] = args.test
    opt['gen_ims_per_race'] = args.gen_ims_per_race
    opt['num_classes'] = args.num_classes
    opt['step_size'] = args.step_size
    opt['num_extra_images'] = args.num_extra_images
    opt['experiment_race'] = args.experiment_race
    opt['simplex_max'] = args.simplex_max
    opt['simplex_point_id'] = args.simplex_point_id
    if opt['noise_mix']:
        opt['lr'] = opt['lr'] * 5

    print(f'Experiment is {opt["experiment_name"]}.')

    train_loader = None
    test_loader = None
    test_pair_list = None
    Trainer = get_method_trainer(opt['method'])

    if opt['train']:
        print('\'Train\' is set.')
        print(f'Train race is {opt["train_race"]}.')

        train_dataset = get_trainset(opt)
        train_loader = data.DataLoader(train_dataset,
                                    batch_size=opt['train_batch_size'],
                                    shuffle=True,
                                    num_workers=opt['num_workers'])

        print(f'train iters per epoch: {len(train_loader)}')

        trainer = Trainer(opt, train_loader, test_loader, test_pair_list)
        trainer.train()
        trainer.validate(train_loader)

    if opt['test']:
        print('\'Test\' is set.')
        opt['test_race'] = args.test_race
        print(f'Test race is {opt["test_race"]}.')

        test_dataset = get_testset(opt)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=opt['test_batch_size'],
                                      shuffle=False,
                                      num_workers=opt['num_workers'])
        test_pair_list = getRFWList(os.path.join(opt['test_root'], 'txts', opt['test_race'], opt['test_race'] + opt['rfw_test_list']))

        trainer = Trainer(opt, train_loader, test_loader, test_pair_list)
        features_dict, acc = trainer.evaluate()
        print('test acc: {}'.format(acc))

        if not os.path.isdir(os.path.join(opt['features_dir'], opt['experiment_name'], opt['method'], opt['test_race'])):
            os.makedirs(os.path.join(opt['features_dir'], opt['experiment_name'], opt['method'], opt['test_race']))
        if opt['experiment_name'] == 'single_race':
            save_name = os.path.join(opt['features_dir'], opt['experiment_name'], opt['method'], opt['test_race'], '_'.join([opt['train_race'], opt['backbone'], str(opt['trial_num'])]) + '.pth.tar')
        elif opt['experiment_name'] == 'race_pairs':
            save_name = os.path.join(opt['features_dir'], opt['experiment_name'], opt['method'], opt['test_race'], '_'.join([opt['train_race'], opt['train_race_b'], str(opt['percent_primary']), opt['backbone'], str(opt['trial_num'])]) + '.pth.tar')
        elif 'all_races' in opt['experiment_name']:
            save_name = os.path.join(opt['features_dir'], opt['experiment_name'], opt['method'], opt['test_race'], '_'.join([opt['backbone'], str(opt['num_classes']), str(opt['trial_num'])]) + '.pth.tar')
        elif opt['experiment_name'] == 'images_vs_identities':
            save_name = os.path.join(opt['features_dir'], opt['experiment_name'], opt['method'], opt['test_race'], '_'.join([opt['backbone'], str(opt['num_classes']), opt['experiment_race'], 
                str(opt['num_extra_images']), str(opt['trial_num'])]) + '.pth.tar')
        elif opt['experiment_name'] == 'race_simplex':
            save_name = os.path.join(opt['features_dir'], opt['experiment_name'], opt['method'], opt['test_race'], '_'.join([str(opt['simplex_max']), str(opt['simplex_point_id']), opt['backbone'], str(opt['trial_num'])]) + '.pth.tar')
        elif opt['experiment_name'] == 'balanced_face':
            save_name = os.path.join(opt['features_dir'], opt['experiment_name'], opt['method'], opt['test_race'], '_'.join([opt['backbone'], str(opt['trial_num'])]) + '.pth.tar')
        torch.save(features_dict, save_name)

    if opt['gen_ims_per_race'] and opt['experiment_name'] == 'all_races':
        print(f'Generating new balance file with {opt["num_classes"] + opt["step_size"]} classes')

        test_features = []
        for race in ['African', 'Asian', 'Caucasian', 'Indian']:
            opt['test_race'] = race
            test_dataset = get_testset(opt)
            test_loader = data.DataLoader(test_dataset,
                                        batch_size=opt['test_batch_size'],
                                        shuffle=False,
                                        num_workers=opt['num_workers'])

            trainer = Trainer(opt, train_loader, test_loader, test_pair_list)
            features_dict = trainer.get_features_dict()
            test_features.append(features_dict)
        weighted_race_neighbors = compute_weighted_neighbors(test_features)
        save_new_distribution(opt, weighted_race_neighbors)
