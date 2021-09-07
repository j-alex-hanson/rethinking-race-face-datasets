from models import FocalLoss
from models import resnet_face18, resnet_face50, senet50_scratch_dag

import numpy as np
import os
import random
import time
import torch
from torch.nn import DataParallel

class Trainer(object):
    
    def __init__(self, opt, train_loader, test_loader, pair_list):

        self.opt = opt
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pair_list = pair_list

        if opt['loss'] == 'focal_loss':
            self.criterion = FocalLoss(gamma=2)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if opt['backbone'] == 'resnet18':
            self.model = resnet_face18(opt['num_classes'], opt['save_center'], use_se=opt['use_se'])
        elif opt['backbone'] == 'resnet50':
            self.model = resnet_face50(opt['num_classes'], opt['save_center'], use_se=opt['use_se'])
        elif opt['backbone'] == 'senet50':
            self.model = senet50_scratch_dag()

        self.model.cuda()
        self.model = DataParallel(self.model)

        self.epoch = 0
        self.max_epoch = opt['max_epoch']

        random.seed(42)


    def load_model(self):
        if self.opt['experiment_name'] == 'single_race':
            if self.opt['noise_mix']:
                checkpoint_file = '_'.join([self.opt['train_race'], self.opt['backbone'], str(self.opt['data_noise']), str(self.opt['trial_num']) + '.pth'])
            else:
                checkpoint_file = '_'.join([self.opt['train_race'], self.opt['backbone'], str(self.opt['trial_num']) + '.pth'])
        elif self.opt['experiment_name'] == 'balanced_face':
            checkpoint_file = '_'.join([self.opt['backbone'], str(self.opt['trial_num']) + '.pth'])
        elif self.opt['experiment_name'] == 'race_pairs':
            checkpoint_file = '_'.join([self.opt['train_race'], self.opt['train_race_b'], str(self.opt['percent_primary']),
                self.opt['backbone'], str(self.opt['trial_num']) + '.pth'])
        elif 'all_races' in self.opt['experiment_name']:
            checkpoint_file = '_'.join([self.opt['backbone'], str(self.opt['num_classes']), str(self.opt['trial_num']) + '.pth'])
        elif self.opt['experiment_name'] == 'images_vs_identities':
            checkpoint_file = '_'.join([self.opt['backbone'], str(self.opt['num_classes']), self.opt['experiment_race'], 
                str(self.opt['num_extra_images']), str(self.opt['trial_num']) + '.pth'])
        elif self.opt['experiment_name'] == 'race_simplex':
            checkpoint_file = '_'.join([str(self.opt['simplex_max']), str(self.opt['simplex_point_id']), self.opt['backbone'], str(self.opt['trial_num']) + '.pth'])

        checkpointPath = os.path.join(self.opt['checkpoints_path'], self.opt['experiment_name'], self.opt['method'], checkpoint_file)
        if os.path.isfile(checkpointPath):
            print(f'Loading from checkpoint: {checkpointPath}')
            checkpoint = torch.load(checkpointPath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.metric_fc.load_state_dict(checkpoint['metric_fc_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    def train_iteration(self, epoch):
        pass


    def train(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.train_iteration(epoch)

            if epoch % self.opt['save_interval'] == 0:
                self.save_model(epoch)

            if self.opt['eval_each_epoch']:
                self.model.eval()
                _, acc = self.evaluate()
                print('test acc: {}'.format(acc))

        self.save_model(self.max_epoch)


    def validate(self, data_loader):
        self.model.eval()

        start = time.time()
        accs = []
        with torch.no_grad():
            for ii, batch in enumerate(data_loader):
                images, label = batch
                images = images.cuda()
                label = label.cuda().long()
                feature = self.model(images)
                output = self.metric_fc(feature, label)

                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                accs.append(acc)
                if ii % self.opt['print_freq'] == 0:
                    speed = self.opt['print_freq'] / (time.time() - start)
                    mean_accuracy = np.array(accs).mean()
                    print(f'validate iter {ii} {speed} iters/s acc {mean_accuracy}')

                    start = time.time()

                torch.cuda.empty_cache()


    def cosine_metric(self, x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


    def calc_threshold(self, y_score, y_true):
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        best_acc = 0
        best_th = 0
        for i in range(len(y_score)):
            th = y_score[i]
            y_test = (y_score >= th)
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th

        print('best acc, best threshold', str(best_acc), str(best_th))
        return best_th


    def get_pairwise_similarities(self, features_dict):
        sims = []
        labels = []
        for pair in self.pair_list:
            features1 = features_dict[pair[0]]
            features2 = features_dict[pair[1]]
            label = pair[2]
            sim = self.cosine_metric(features1, features2)

            sims.append(sim)
            labels.append(label)
        return sims, labels


    def calc_accuracy(self, pairwise_similarities, pair_labels, threshold):
        predictions = (np.asarray(pairwise_similarities) >= threshold)
        accuracy = np.mean((predictions == np.asarray(pair_labels)).astype(int))
        return accuracy

    
    def compute_accuracy_from_folds(self, pairwise_similarities, pair_labels):
        idxs = list(range(len(pair_labels)))
        random.shuffle(idxs)
        partitions = [idxs[i::10] for i in range(10)]

        pairwise_similarities = np.asarray(pairwise_similarities)
        pair_labels = np.asarray(pair_labels)

        accuracies = []
        for i in range(10):
            train_pairwise_similarities = []
            train_pair_labels = []
            train_partitions = partitions[:i] + partitions[i+1:]
            for partition in train_partitions:
                train_pairwise_similarities.extend(pairwise_similarities[partition].tolist())
                train_pair_labels.extend(pair_labels[partition].tolist())
            
            val_partition = partitions[i]
            val_pairwise_similarities = pairwise_similarities[val_partition].tolist()
            val_pair_labels = pair_labels[val_partition].tolist()

            threshold = self.calc_threshold(train_pairwise_similarities, train_pair_labels)
            accuracy = self.calc_accuracy(val_pairwise_similarities, val_pair_labels, threshold)
            accuracies.append(accuracy)

        return np.mean(accuracies)


    def make_features_dict(self, identities, features):
        features_dict = {}
        for i, img_iden in enumerate(identities):
            features_dict[img_iden] = features[i]
        return features_dict


    def get_features(self, data_loader):
        self.model.eval()
        features = None
        img_idens = []
        cnt = 0

        with torch.no_grad():
            for batch in data_loader:
                img, img_iden = batch
                img_idens.extend(list(img_iden))
                img = img.cuda()
                feature = self.model(img)
                feature = feature.data.cpu().numpy()
                if features is None:
                    features = feature
                else:
                    features = np.vstack((features, feature))

                cnt += len(batch)

                torch.cuda.empty_cache()

        return features, img_idens, cnt


    def get_features_dict(self):
        start_time = time.time()
        features, identities, count = self.get_features(self.test_loader)
        print(features.shape)
        elapsed_time = time.time() - start_time
        print('total time to get all test features is {}, average time is {}'.format(elapsed_time, elapsed_time / count))
        features_dict = self.make_features_dict(identities, features)
        return features_dict


    def evaluate(self):
        features_dict = self.get_features_dict()
        pairwise_similarities, pair_labels = self.get_pairwise_similarities(features_dict)
        accuracy = self.compute_accuracy_from_folds(pairwise_similarities, pair_labels)
        print('test verification accuracy: ', accuracy)
        return features_dict, accuracy


    def save_model(self, epoch):
        opt = self.opt

        if not os.path.isdir(os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method'])):
            os.makedirs(os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method']))

        if opt['experiment_name'] == 'single_race':
            if opt['noise_mix']:
                save_name = os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method'], '_'.join([opt['train_race'], opt['backbone'], str(opt['data_noise']), str(opt['trial_num'])]) + '.pth')
            else:
                save_name = os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method'], '_'.join([opt['train_race'], opt['backbone'], str(opt['trial_num'])]) + '.pth')
        elif opt['experiment_name'] == 'balanced_face':
            save_name = os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method'], '_'.join([opt['backbone'], str(opt['trial_num'])]) + '.pth')
        elif opt['experiment_name'] == 'race_pairs':
            save_name = os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method'], '_'.join([opt['train_race'], opt['train_race_b'], str(opt['percent_primary']), opt['backbone'], str(opt['trial_num'])]) + '.pth')
        elif 'all_races' in self.opt['experiment_name']:
            save_name = os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method'], '_'.join([opt['backbone'], str(opt['num_classes']), str(opt['trial_num'])]) + '.pth')
        elif opt['experiment_name'] == 'images_vs_identities':
            save_name = os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method'], '_'.join([self.opt['backbone'], str(self.opt['num_classes']), self.opt['experiment_race'], 
                str(self.opt['num_extra_images']), str(self.opt['trial_num'])]) + '.pth')
        elif opt['experiment_name'] == 'race_simplex':
            save_name = os.path.join(opt['checkpoints_path'], opt['experiment_name'], opt['method'], '_'.join([str(self.opt['simplex_max']), str(self.opt['simplex_point_id']), self.opt['backbone'], str(self.opt['trial_num'])]) + '.pth')
        torch.save({
            'epoch': epoch + 1,
            'trial_num': opt['trial_num'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric_fc_state_dict': self.metric_fc.state_dict()
        }, save_name)
