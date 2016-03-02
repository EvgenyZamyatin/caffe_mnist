import numpy
import os
import pickle
import caffe
import lmdb
import struct
from caffe import layers as L, params as P
from matplotlib import pyplot as plt
import xgboost as xgb
import argparse as argparse
import random

caffe.set_device(0)
caffe.set_mode_gpu()


def next_label(label_data):
    return struct.unpack('>B', label_data.read(1))[0]


def next_image(image_data, rows, cols):
    return numpy.frombuffer(image_data.read(rows * cols), dtype=numpy.uint8)


def create_lmdb(name, f_images, f_labels, drop_labels=()):
    if os.path.exists(name):
        return
    lmdb.open(name)
    image_data = open(f_images, 'rb')
    label_data = open(f_labels, 'rb')
    image_magic_number = struct.unpack('>i', image_data.read(4))[0]
    label_magic_number = struct.unpack('>i', label_data.read(4))[0]
    images = struct.unpack('>i', image_data.read(4))[0]
    labels = struct.unpack('>i', label_data.read(4))[0]
    assert images == labels
    n = images
    rows = struct.unpack('>i', image_data.read(4))[0]
    cols = struct.unpack('>i', image_data.read(4))[0]
    db = lmdb.open(name, map_size=n * (rows * cols + 1) * 2)
    gen = [(next_image(image_data, rows, cols), next_label(label_data)) for i in range(n)]
    gen = filter(lambda x: x[1] not in drop_labels, gen)
    with db.begin(write=True) as txn:
        for i, (image, label) in enumerate(gen):
            datum = caffe.io.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = rows
            datum.width = cols
            datum.data = image.tobytes()
            datum.label = label
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
    return


def create_network(name, db, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, source=db, backend=P.Data.LMDB,
                             transform_param=dict(scale=1. / 255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    with open(name, 'w+') as out:
        out.write(str(n.to_proto()))


def get_feature_vector(net, image):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    out = net.forward()
    return out['score'][0]


def get_feature_vector_from_mtr(net, mtr):
    net.blobs['data'].data[...] = mtr
    out = net.forward()
    return out['score'][0]


def build_without_36():
    create_lmdb('train_without_36.lmdb', 'data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte', (3, 6))
    create_lmdb('test_without_36.lmdb', 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte', (3, 6))
    create_network('lenet_train_without_36.prototxt', 'train_without_36.lmdb', 64)
    create_network('lenet_test_without_36.prototxt', 'test_without_36.lmdb', 100)
    solver = caffe.SGDSolver('lenet_solver.prototxt')
    solver.solve()
    net = solver.test_nets[0]
    results = None
    true_results = None
    for i in range(100):
        out = net.forward(blobs=['score', 'label'])
        results = numpy.vstack((results, out['score'])) if results is not None else out['score']
        true_results = numpy.hstack((true_results, out['label'])) if true_results is not None else out['label']
    print sum(results.argmax(1) == true_results) * 1.0 / numpy.size(results, 0)


def create_train_test_pairs():
    create_lmdb('train_only_36.lmdb', 'data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte',
                (0, 1, 2, 4, 5, 7, 8, 9))
    env = lmdb.open('train_only_36.lmdb', readonly=True)
    arr = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            raw_datum = value
            datum = caffe.io.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = numpy.fromstring(datum.data, dtype=numpy.uint8)
            x = flat_x.reshape(datum.channels, datum.height, datum.width)
            y = datum.label
            arr.append((x, y))
    random.shuffle(arr)
    with open('digit_desc_36.bin', 'w+b') as out:
        pickle.dump(arr, out)
    train_samples = []
    for i in range(5000):
        a = random.randint(0, len(arr) / 2 - 1)
        b = random.randint(0, len(arr) / 2 - 1)
        if arr[a][1] == arr[b][1]:
            c = 1
        else:
            c = 0
        train_samples.append((a, b, c))
    test_samples = []
    for i in range(5000):
        a = random.randint(len(arr) / 2 - 1, len(arr) - 1)
        b = random.randint(len(arr) / 2 - 1, len(arr) - 1)
        if arr[a][1] == arr[b][1]:
            c = 1
        else:
            c = 0
        test_samples.append((a, b, c))
    with open('train_test_36samples.bin', 'w+b') as out:
        pickle.dump((train_samples, test_samples), out)


def main():
    # build_without_36()
    # create_train_test_pairs()
    net = caffe.Net('lenet_deploy.prototxt', 'mnist/lenet_iter_10000.caffemodel', caffe.TEST)
    with open('digit_desc_36.bin', 'rb') as data:
        digits = pickle.load(data)
    with open('train_test_36samples.bin', 'rb') as data:
        train_samples, test_samples = pickle.load(data)

    vectors = []
    for mtr, label in digits:
        v = list(get_feature_vector_from_mtr(net, mtr * 1. / 255.))
        vectors.append(v)

    # data = []
    # for i in random.sample(list(range(len(vectors))), 5):
    #    get_feature_vector_from_mtr(net, digits[i][0]*1./255)
    #    data.append(net.blobs['pool2'].data[0, :36].copy())
    # data = numpy.vstack(tuple(data))
    # vis_square('tmp4.png', data)
    # exit()

    vectors = numpy.array(vectors)
    feature = lambda x, y: (numpy.hstack((x,y)))
    train_X = numpy.array([feature(vectors[a], vectors[b]) for a, b, _ in train_samples])
    train_Y = numpy.array([c for a, b, c in train_samples])

    test_X = numpy.array([feature(vectors[a], vectors[b]) for a, b, _ in test_samples])
    test_Y = numpy.array([c for a, b, c in test_samples])

    # for a, b, c in train_samples[:10]:
    #    print(list(vectors[a]))
    #    print(list(vectors[b]))
    #    print c
    # exit()

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(test_X, label=test_Y)

    three_mean = numpy.sum(vectors[i] for i, (_, label) in enumerate(digits) if label == 3) / sum(
        1 for i, (_, label) in enumerate(digits) if label == 3)
    six_mean = numpy.sum(vectors[i] for i, (_, label) in enumerate(digits) if label == 6) / sum(
        1 for i, (_, label) in enumerate(digits) if label == 6)

    print three_mean
    print six_mean

    param = {'objective': 'binary:logistic',
             'max_depth': 10,
             'eta': 0.05,
             'subsample': 0.1,
             'lambda': 100}

    eval = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 30000
    bst = xgb.train(param, dtrain, num_round, eval)
    bst.save_model('xgboost.model_v1')
    f = lambda x: 1 if x >= 0.5 else 0
    labels = [f(i) for i in bst.predict(dtest)]
    print sum(labels == test_Y) * 1. / len(labels)
    labels = [f(i) for i in bst.predict(dtrain)]
    print sum(labels == train_Y) * 1. / len(labels)
    return


def vis_square(name, data):
    data = (data - data.min()) / (data.max() - data.min())
    n = int(numpy.ceil(numpy.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = numpy.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data, 'gray')
    plt.axis('off')
    plt.savefig(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()
