import os.path
import itertools
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
import gudhi as gd

from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.linalg import eigh
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import random_uniform_initializer as rui
from perslay import PerslayModel


dataset = "ORBIT5K"


def get_parameters():
    dataset_parameters = {"data_type": "orbit", "filt_names": ["Alpha0", "Alpha1"]}
    return dataset_parameters


def get_model(nf, hparams={}):
    plp = {}
    plp["pweight"] = "grid"
    plp["pweight_init"] = rui(1., 1.)
    plp["pweight_size"] = (10, 10)
    plp["pweight_bnds"] = ((-0.001, 1.001), (-0.001, 1.001))
    plp["pweight_train"] = True
    plp["layer"] = "PermutationEquivariant"
    plp["lpeq"] = [(25, None), (25, "max")]
    plp["lweight_init"] = rui(0.,1.)
    plp["lbias_init"] = rui(0.,1.)
    plp["lgamma_init"] = rui(0.,1.)
    plp["layer_train"] = True
    plp["perm_op"] = "topk"
    plp["keep"] = 5
    plp["final_model"] = "identity"
    plp.update(hparams)
    perslay_parameters = [plp for _ in range(2)]

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        rho = tf.keras.Sequential([tf.keras.layers.Dense(5, activation="sigmoid", input_shape=(250+nf,))])
        model = PerslayModel(name="PersLay", diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=20, decay_rate=1., staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-4)
        # optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.9) 
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

    return model, optimizer, loss, metrics


def hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def apply_graph_extended_persistence(A, filtration_val):
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):        
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmOrd0 if p[0] == 0]) if len(dgmOrd0) else np.empty([0,2])
    dgmRel1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmRel1 if p[0] == 1]) if len(dgmRel1) else np.empty([0,2])
    dgmExt0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmExt0 if p[0] == 0]) if len(dgmExt0) else np.empty([0,2])
    dgmExt1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmExt1 if p[0] == 1]) if len(dgmExt1) else np.empty([0,2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1


def generate_diagrams_and_features(X_orbit, path_dataset=""):
    dataset_parameters = get_parameters()

    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    if os.path.isfile(path_dataset + dataset + ".hdf5"):
        os.remove(path_dataset + dataset + ".hdf5")
    diag_file = h5py.File(path_dataset + dataset + ".hdf5", "w")
    [diag_file.create_group(str(filtration)) for filtration in dataset_parameters["filt_names"]]

    labs = []
    count = 0
    num_diag_per_param = 1000 if "5K" in dataset else 20000
    for lab, r in enumerate([2.5, 3.5, 4.0, 4.1, 4.3]):
        for dg in range(num_diag_per_param):
            X = X_orbit[lab * 1000 + dg, :, :]
            alpha_complex = gd.AlphaComplex(points=X)
            st = alpha_complex.create_simplex_tree(max_alpha_square=1e50)
            st.persistence()
            diag_file["Alpha0"].create_dataset(name=str(count), data=np.array(st.persistence_intervals_in_dimension(0)))
            diag_file["Alpha1"].create_dataset(name=str(count), data=np.array(st.persistence_intervals_in_dimension(1)))
            orbit_label = {"label": lab, "pcid": count}
            labs.append(orbit_label)
            count += 1
    labels = pd.DataFrame(labs)
    labels.set_index("pcid")
    features = labels[["label"]]

    features.to_csv(path_dataset + dataset + ".csv")

    return diag_file.close()


def load_data(path_dataset="", filtrations=[], verbose=False):

    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    diagfile = h5py.File(path_dataset + dataset + ".hdf5", "r")
    filts = list(diagfile.keys()) if len(filtrations) == 0 else filtrations

    diags_dict = dict()
    if len(filts) == 0:
        filts = diagfile.keys()
    for filtration in filts:
        list_dgm, num_diag = [], len(diagfile[filtration].keys())
        for diag in range(num_diag):
            list_dgm.append(np.array(diagfile[filtration][str(diag)]))
        diags_dict[filtration] = list_dgm

    # Extract features and encode labels with integers
    feat = pd.read_csv(path_dataset + dataset + ".csv", index_col=0, header=0)
    F = np.array(feat)[:, 1:]  # 1: removes the labels
    L = np.array(LabelEncoder().fit_transform(np.array(feat["label"])))
    L = OneHotEncoder(sparse=False, categories="auto").fit_transform(L[:, np.newaxis])

    if verbose:
        print("Dataset:", dataset)
        print("Number of observations:", L.shape[0])
        print("Number of classes:", L.shape[1])

    return diags_dict, F, L


def visualize_diagrams(diags_dict, ilist=(0, 10, 20, 30, 40, 50)):
    filts = diags_dict.keys()
    n, m = len(filts), len(ilist)
    _, axs = plt.subplots(n, m, figsize=(m*n / 2, n*m / 2))
    for (i, filtration) in enumerate(filts):
        for (j, idx) in enumerate(ilist):
            xs, ys = diags_dict[filtration][idx][:, 0], diags_dict[filtration][idx][:, 1]
            axs[i, j].scatter(xs, ys)
            axs[i, j].plot([0, 1], [0, 1])
            axs[i, j].axis([0, 1, 0, 1])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    # axis plot
    cols = ["idx = " + str(i) for i in ilist]
    rows = filts
    for ax, col in zip(axs[0], cols):
        ax.set_tiX_orbit5ktle(col)
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size="large")
    plt.show()


def evaluate_model(L, F, D, train_sub, test_sub, model, optimizer, loss, metrics, num_epochs, batch_size=128, verbose=1, plots=False):

    _, _, _, num_filt = L.shape[0], L.shape[1], F.shape[1], len(D)

    label_train, label_test = L[train_sub, :], L[test_sub, :]
    feats_train, feats_test = F[train_sub, :], F[test_sub, :]
    diags_train, diags_test = [D[dt][train_sub, :] for dt in range(num_filt)], [D[dt][test_sub, :] for dt in range(num_filt)]



    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(x=[diags_train, feats_train], y=label_train, validation_data=([diags_test, feats_test], label_test), epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose=verbose)
    train_results = model.evaluate([diags_train, feats_train], label_train, verbose=verbose)
    test_results = model.evaluate([diags_test,  feats_test],  label_test, verbose=verbose)
    
    if plots:
        ltrain, ltest = history.history["categorical_accuracy"], history.history["val_categorical_accuracy"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.array(ltrain), color="blue", label="train acc")
        ax.plot(np.array(ltest),  color="red",  label="test acc")
        ax.set_ylim(top=1.)
        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel("classif. accuracy")
        ax.set_title("Evolution of train/test accuracy")
        plt.show()

    return history.history, train_results, test_results
