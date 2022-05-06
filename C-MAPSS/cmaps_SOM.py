# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:40:29 2022

@author: alan-
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, pdist
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
import numpy as np
import streamlit as st
# %%


def filtro_ng(treino, tol):
    filtered = [treino[0]]
    for i, z in enumerate(treino):
        if (pdist(np.vstack([np.array(filtered), z])) > tol).all():
            filtered.append(z)
    return np.array(filtered)


def filtro_ngbatch(treino, tol, n_splits):
    splits = np.array_split(treino, n_splits)
    batchs = []
    for k in splits:
        batchs.append(filtro_ng(k, tol))
    return np.concatenate(batchs)


@st.cache
def load_data():
    rul_data = pd.read_csv('RUL_FD001.csv', sep=';')
    train = pd.read_csv('train_FD001.csv', sep=';', index_col='Cycle')
    train = train.apply(pd.to_numeric, errors='coerce')
    train.drop(columns=['OPSet1', 'OPSet2', 'OPSet3'], inplace=True)

    test = pd.read_csv('test_FD001.csv', sep=';', index_col='Cycle')
    test = test.apply(pd.to_numeric, errors='coerce')

    testUnNumber, trainUnNumber = train_test_split(np.arange(1, len(rul_data)+1), random_state=0)
    train_split = train[train['Unit'].isin(trainUnNumber)]
    test_split = train[train['Unit'].isin(testUnNumber)]

    return train_split, test_split


@st.cache
def scale_filter(data, n_splits, ng_tol):
    scaler = MinMaxScaler()
    scaler.fit(data)
    train_split_scaled = scaler.transform(data)
    train_split_filtered = filtro_ngbatch(train_split_scaled, ng_tol, n_splits)
    return train_split_filtered, scaler


@st.cache(allow_output_mutation=True)
def som_train(n_neurons, m_neurons, data_train, sigma, neighborhood_function, activation_distance, learning_rate, do_pca=False):
    if do_pca:
        som = MiniSom(n_neurons, m_neurons, data_train.shape[1], sigma=sigma,
                      neighborhood_function=neighborhood_function, activation_distance=activation_distance,
                      learning_rate=learning_rate, random_seed=0)
        som.pca_weights_init(data_train)
        som.train(data_train, iterations, verbose=True)  # random training
    else:
        som = MiniSom(n_neurons, m_neurons, data_train.shape[1], sigma=sigma,
                      neighborhood_function=neighborhood_function, activation_distance=activation_distance,
                      learning_rate=learning_rate, random_seed=0)
        som.train(data_train, iterations, verbose=True)  # random training

    return som


# %%
load_data_state = st.text('Loading data...')
train_split, test_split = load_data()
load_data_state.text("Ready")

n_splits = st.sidebar.slider('n of splits NG filter', 1, 20, 1, step=1)

ng_tol = st.sidebar.slider('tolerance for NG filter', 0.0, 1.0, 0.1, step=0.01)

filter_data_state = st.text('Filtering data...')
train_split_filtered, scaler = scale_filter(train_split.drop(columns=['Unit']), n_splits, ng_tol)
filter_data_state.text("Ready")
# %%
st.sidebar.markdown('## Parameters')
neighborhood_function = st.sidebar.selectbox('neighborhood_function',
                                             ['gaussian', 'mexican_hat', 'bubble', 'triangle'])
activation_distance = st.sidebar.selectbox('activation_distance',
                                           ['euclidean', 'cosine', 'manhattan', 'chebyshev'])


n_neurons = st.sidebar.number_input('n_neurons', 1, 150, 9, step=1)
m_neurons = n_neurons
sigma = st.sidebar.number_input('sigma', 0.1, 50.0, 1.5, step=0.1)
learning_rate = st.sidebar.number_input('learning rate', 0.1, 10.0, 0.5, step=0.1)
iterations = st.sidebar.number_input('training iterations', 0, 2000, 500, step=1)

som_train_state = st.text('Training data...')
som = som_train(n_neurons, m_neurons, train_split_filtered, sigma,
                neighborhood_function, activation_distance, learning_rate)
if st.sidebar.checkbox('PCA initialization'):
    som = som_train(n_neurons, m_neurons, train_split_filtered, sigma,
                    neighborhood_function, activation_distance, learning_rate, do_pca=True)
som_train_state.text("Ready")

all_ends = []
for i, k in enumerate(train_split['Unit'].unique()):
    all_ends.append(train_split[train_split['Unit'] == k].iloc[-1, 1:].values)
all_ends = scaler.transform(np.array(all_ends))

all_inis = []
for i, k in enumerate(train_split['Unit'].unique()):
    all_inis.append(train_split[train_split['Unit'] == k].iloc[0, 1:].values)
all_inis = scaler.transform(np.array(all_inis))

all_mids = []
for i, k in enumerate(train_split['Unit'].unique()):
    # half = int(len(train_split[train_split['Unit'] == k])/2)
    # print(half)
    all_mids.append(train_split[train_split['Unit'] == k].iloc[-30, 1:].values)
all_mids = scaler.transform(np.array(all_mids))


bmus_end = []
for xx in all_ends:
    bmus_end.append(som.winner(xx))
bmus_end = np.array(bmus_end)

bmus_ini = []
for xx in all_inis:
    bmus_ini.append(som.winner(xx))
bmus_ini = np.array(bmus_ini)

bmus_mid = []
for xx in all_mids:
    bmus_mid.append(som.winner(xx))
bmus_mid = np.array(bmus_mid)

fig = plt.figure(figsize=(10, 9))
plt.pcolormesh(som.distance_map().T, cmap='bone_r')
plt.scatter(bmus_ini[:, 0]+0.5, bmus_ini[:, 1]+0.5, color='red', marker='+', s=150, label='Initial state')
plt.scatter(bmus_mid[:, 0]+0.5, bmus_mid[:, 1]+0.5, color='orange', marker='3', s=150, label='Middle state')
plt.scatter(bmus_end[:, 0]+0.5, bmus_end[:, 1]+0.5, color='blue', marker='*', s=150, label='End state')
plt.legend(fontsize="x-large")
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
st.pyplot(fig)
# st.write('erro convergencia: ', map_gtm.log_err)
st.write('len train_y_before: ', len(train_split))
st.write('len train_y_after: ', len(train_split_filtered))


# %%
if st.sidebar.checkbox('INSTANCIA 1'):
    st.header('INSTANCIA 1')
    unit = st.select_slider('Unidade:', options=test_split['Unit'].unique(), key='1')

    teste = test_split[test_split['Unit'] == unit]
    teste.drop(columns=['Unit'], inplace=True)

    teste_scaled = scaler.transform(teste)
    bmus = []
    for xxx in teste_scaled:
        bmus.append(som.winner(xxx))
    bmus = np.array(bmus)
    qu = np.arange(len(bmus))
    bmus = bmus[0::10]
    qu = qu[0::10]

    q = st.number_input('to point: ', min_value=0, max_value=len(bmus)-1, value=0, step=1, key='11')

    fig_padrao, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(bmus_ini[:, 0]+0.5, bmus_ini[:, 1]+0.5, color='red', marker='+', s=150, label='Initial state')
    ax1.scatter(bmus_mid[:, 0]+0.5, bmus_mid[:, 1]+0.5, color='orange', marker='3', s=150, label='Middle state')
    ax1.scatter(bmus_end[:, 0]+0.5, bmus_end[:, 1]+0.5, color='blue', marker='*', s=150, label='End state')
    ax1.scatter(bmus[q, 0]+0.5, bmus[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(bmus[0:q+1, 0]+0.5, bmus[0:q+1, 1]+0.5, '*-', zorder=3)
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax2.plot(teste_scaled[0:qu[q]])
    ax2.set_xlabel('Cycles', fontsize='x-large')
    ax2.set_ylabel('Scaled', fontsize='x-large')
    ax2.set_xlim(0, len(teste_scaled))
    st.pyplot(fig=fig_padrao)
# %%
if st.sidebar.checkbox('INSTANCIA 2'):
    st.header('INSTANCIA 2')
    unit = st.select_slider('Unidade:', options=test_split['Unit'].unique(), key='2')

    teste = test_split[test_split['Unit'] == unit]
    teste.drop(columns=['Unit'], inplace=True)

    teste_scaled = scaler.transform(teste)
    bmus = []
    for xxx in teste_scaled:
        bmus.append(som.winner(xxx))
    bmus = np.array(bmus)

    q = st.number_input('to point: ', min_value=0, max_value=len(bmus)-1, value=0, step=1, key='22')

    fig_padrao, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(bmus[q, 0]+0.5, bmus[q, 1]+0.5, color='red', zorder=5)
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax1.plot(bmus[0:q+1, 0]+0.5, bmus[0:q+1, 1]+0.5, '*-', zorder=3)
    ax2.plot(teste_scaled[0:q])
    ax2.set_xlim(0, len(teste_scaled))
    ax2.set_xlabel('Cycles', fontsize='x-large')
    ax2.set_ylabel('Scaled', fontsize='x-large')
    st.pyplot(fig=fig_padrao)
# %%
if st.sidebar.checkbox('INSTANCIA 3'):
    st.header('INSTANCIA 3')
    unit = st.select_slider('Unidade:', options=test_split['Unit'].unique(), key='3')

    teste = test_split[test_split['Unit'] == unit]
    teste.drop(columns=['Unit'], inplace=True)

    teste_scaled = scaler.transform(teste)
    bmus = []
    for xxx in teste_scaled:
        bmus.append(som.winner(xxx))
    bmus = np.array(bmus)

    q = st.number_input('to point: ', min_value=0, max_value=len(bmus)-1, value=0, step=1, key='33')

    fig_padrao, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax1.scatter(bmus[q, 0]+0.5, bmus[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(bmus[0:q+1, 0]+0.5, bmus[0:q+1, 1]+0.5, '*-', zorder=3)
    ax2.plot(teste_scaled[0:q])
    ax2.set_xlabel('Cycles', fontsize='x-large')
    ax2.set_ylabel('Scaled', fontsize='x-large')
    ax2.set_xlim(0, len(teste_scaled))
    st.pyplot(fig=fig_padrao)
