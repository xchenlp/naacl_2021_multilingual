import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--label', type=str, default='soft')  # 'soft' or 'hard'
args = parser.parse_args()


if __name__ == '__main__':
    cls2ind = torch.load(os.path.join(args.data_path, 'cls2ind_batch4.dict'))
    ind2cls = torch.load(os.path.join(args.data_path, 'ind2cls_batch4.dict'))
    for set_name in ('va', 'tr'):

        data = torch.load(os.path.join(args.data_path, '{}_hard_batch4_marina.pkl'.format(set_name)))

        label_counter = np.zeros((len(cls2ind),), dtype=np.float32)
        if args.label == 'hard':
            for _, _, label in data:
                label_counter[label] += 1
        elif args.label == 'soft':
            for _, label_prob in data:
                label_counter[np.argmax(label_prob)] += 1
        label_counter = label_counter / len(data) * 100

        np.set_printoptions(precision=1)
        print(label_counter)

        fig, ax = plt.subplots()
        plt.bar(np.arange(len(cls2ind)), label_counter)
        print("{} size: {}".format(set_name, len(data)))
        fig.savefig("{}_class_occurrence.png".format(set_name))

        class_counter = [(ind2cls[i], v) for i, v in enumerate(label_counter)]
        class_counter.sort(key=lambda x: x[1], reverse=True)
        print('Most frequent 8 classes for {} set: {}'.format(set_name, class_counter[:8]))
