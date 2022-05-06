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
from ugtm import eGTM
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
def gtm_train(k, s, m, niter, regul, data_train):

    map_gtm = eGTM(k=k, s=s, m=m, niter=500, regul=regul, verbose=True, model='modes').fit(data_train)

    responsibilities = map_gtm.optimizedModel.matR
    nodes_cumulated_responsibilities = sum(map_gtm.optimizedModel.matR, 0)
    nodes_coordinates = map_gtm.optimizedModel.matX

    gtm_heatmap, xedges, yedges = np.histogram2d(nodes_coordinates[:, 0], nodes_coordinates[:, 1], bins=k,
                                                 range=([-1, 1], [-1, 1]), weights=nodes_cumulated_responsibilities)
    centers = [-1, 1, -1, 1]
    dx, = np.diff(centers[:2]) / (gtm_heatmap.shape[1] - 1)
    dy, = -np.diff(centers[2:]) / (gtm_heatmap.shape[0] - 1)
    extent = [centers[0] - dx / 2, centers[1] + dx / 2, centers[2] + dy / 2, centers[3] - dy / 2]

    return map_gtm, gtm_heatmap, extent


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
k = st.sidebar.number_input('k', 1, 50, 10, step=1)
s = st.sidebar.number_input('s', 0.0, 10.0, 0.5, step=0.1)
m = st.sidebar.number_input('m', 0, 10, 2, step=1)
regul = st.sidebar.number_input('regularization', 0.0, 100.0, 0.5, step=0.1)
niter = st.sidebar.number_input('number iterations', 0, 5000, 500, step=1)

gtm_train_state = st.text('Training data...')
map_gtm, gtm_heatmap, extent = gtm_train(k, s, m, niter, regul, train_split_filtered)
gtm_train_state.text("Ready")

all_ends = []
for i, k in enumerate(train_split['Unit'].unique()):
    all_ends.append(train_split[train_split['Unit'] == k].iloc[-1, 1:].values)
all_ends = np.array(all_ends)

all_inis = []
for i, k in enumerate(train_split['Unit'].unique()):
    all_inis.append(train_split[train_split['Unit'] == k].iloc[0, 1:].values)
all_inis = np.array(all_inis)

all_mids = []
for i, k in enumerate(train_split['Unit'].unique()):
    # half = int(len(train_split[train_split['Unit'] == k])/2)
    # print(half)
    all_mids.append(train_split[train_split['Unit'] == k].iloc[-30, 1:].values)
all_mids = np.array(all_mids)


bmus_end = map_gtm.transform(scaler.transform(all_ends))
bmus_ini = map_gtm.transform(scaler.transform(all_inis))
bmus_mid = map_gtm.transform(scaler.transform(all_mids))

fig = plt.figure(figsize=(10, 9))
plt.imshow(gtm_heatmap.T, extent=extent, origin='lower', cmap='bone_r', aspect='auto')
plt.scatter(bmus_ini[:, 0], bmus_ini[:, 1], color='red', marker='+', s=150, label='Initial state')
plt.scatter(bmus_mid[:, 0], bmus_mid[:, 1], color='orange', marker='3', s=150, label='Middle state')
plt.scatter(bmus_end[:, 0], bmus_end[:, 1], color='blue', marker='*', s=150, label='End state')
plt.legend(fontsize="x-large")
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
st.pyplot(fig)
st.write('erro convergencia: ', map_gtm.log_err)
st.write('len train_y: ', len(train_split_filtered))

# %%
if st.sidebar.checkbox('INSTANCIA 1'):
    st.header('INSTANCIA 1')
    unit = st.select_slider('Unidade:', options=test_split['Unit'].unique(), key='1')

    teste = test_split[test_split['Unit'] == unit]
    teste.drop(columns=['Unit'], inplace=True)

    teste_scaled = scaler.transform(teste)
    bmus = map_gtm.transform(teste_scaled)
    qu = np.arange(len(bmus))
    bmus = bmus[0::10]
    qu = qu[0::10]

    q = st.number_input('to point: ', min_value=0, max_value=len(bmus)-1, value=0, step=1, key='11')

    fig_padrao, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.imshow(gtm_heatmap.T, extent=extent, origin='lower', cmap='bone_r', aspect='auto')
    ax1.scatter(bmus_ini[:, 0], bmus_ini[:, 1], color='red', marker='+', s=150, label='Initial state')
    ax1.scatter(bmus_mid[:, 0], bmus_mid[:, 1], color='orange', marker='3', s=150, label='Middle state')
    ax1.scatter(bmus_end[:, 0], bmus_end[:, 1], color='blue', marker='*', s=150, label='End state')
    ax1.scatter(bmus[q, 0], bmus[q, 1], color='red', zorder=5)
    ax1.plot(bmus[0:q+1, 0], bmus[0:q+1, 1], '*-', zorder=3)
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
    bmus = np.array(bmus)
    qu = np.arange(len(bmus))
    bmus = bmus[0::10]
    qu = qu[0::10]

    q = st.number_input('to point: ', min_value=0, max_value=len(bmus)-1, value=0, step=1, key='22')

    fig_padrao, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.imshow(gtm_heatmap.T, extent=extent, origin='lower', cmap='bone_r', aspect='auto')
    ax1.scatter(bmus[q, 0], bmus[q, 1], color='red', zorder=5)
    ax1.plot(bmus[0:q+1, 0], bmus[0:q+1, 1], '*-', zorder=3)
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax2.plot(teste_scaled[0:qu[q]])
    ax2.set_xlabel('Cycles', fontsize='x-large')
    ax2.set_ylabel('Scaled', fontsize='x-large')
    ax2.set_xlim(0, len(teste_scaled))
    ax2.set_xlim(0, len(teste_scaled))
    st.pyplot(fig=fig_padrao)
# %%
if st.sidebar.checkbox('INSTANCIA 3'):
    st.header('INSTANCIA 3')
    unit = st.select_slider('Unidade:', options=test_split['Unit'].unique(), key='3')

    teste = test_split[test_split['Unit'] == unit]
    teste.drop(columns=['Unit'], inplace=True)

    teste_scaled = scaler.transform(teste)
    bmus = map_gtm.transform(teste_scaled)

    q = st.number_input('to point: ', min_value=0, max_value=len(bmus)-1, value=0, step=1, key='33')

    fig_padrao, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.imshow(gtm_heatmap.T, extent=extent, origin='lower', cmap='bone_r', aspect='auto')
    ax1.scatter(bmus[q, 0], bmus[q, 1], color='red', zorder=5)
    ax1.plot(bmus[0:q+1, 0], bmus[0:q+1, 1], '*-', zorder=3)
    ax2.plot(teste_scaled[0:q])
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax2.set_xlim(0, len(teste_scaled))
    st.pyplot(fig=fig_padrao)
