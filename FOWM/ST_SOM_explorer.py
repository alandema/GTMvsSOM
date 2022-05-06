# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:09:16 2022

@author: alan-
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
import scipy.io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import FILTROS

st.sidebar.markdown('# MiniSom playground')


@st.cache
def load_data_train(tipo_poco, used_vars):

    if tipo_poco == 'dinamico':
        load_path1 = 'mats/treino_pocodinamico_comlabel.mat'
        load_path2 = 'mats/treino_pocodinamico_comlabel_part2.mat'
    else:
        load_path1 = 'mats/treino_pocoestatico_comlabel.mat'
        load_path2 = 'mats/treino_pocoestatico_comlabel_part2.mat'

    used_vars = [0 if element == 'Ppdg' else element for element in used_vars]
    used_vars = [1 if element == 'Ptt' else element for element in used_vars]
    used_vars = [2 if element == 'Prb' else element for element in used_vars]
    used_vars = [3 if element == 'Prt' else element for element in used_vars]
    used_vars = [4 if element == 'Ttub' else element for element in used_vars]
    used_vars = [5 if element == 'Trai' else element for element in used_vars]
    used_vars = [6 if element == 'Choke' else element for element in used_vars]
    used_vars.append(7)

    data1 = scipy.io.loadmat(load_path1)
    data2 = scipy.io.loadmat(load_path2)
    data_treino1 = data1['dados'][1000:, used_vars, :]
    data_treino2 = data2['dados'][1000:, used_vars, :]
    data_treino = np.dstack([data_treino1, data_treino2])
    treino_stack = data_treino.transpose(2, 0, 1).reshape(-1, data_treino.shape[1])

    treino_y = treino_stack[np.where(treino_stack[:, -1] == 1)][:, :-1]
    treino_y_label = treino_stack[np.where(treino_stack[:, -1] == 1)][:, -1]

    treino_n = treino_stack[np.where(treino_stack[:, -1] == 0)][:, :-1]
    treino_n_label = treino_stack[np.where(treino_stack[:, -1] == 0)][:, -1]

    return treino_y, treino_y_label, treino_n, treino_n_label


@st.cache
def load_data_testepadrao(used_vars):

    used_vars = [0 if element == 'Ppdg' else element for element in used_vars]
    used_vars = [1 if element == 'Ptt' else element for element in used_vars]
    used_vars = [2 if element == 'Prb' else element for element in used_vars]
    used_vars = [3 if element == 'Prt' else element for element in used_vars]
    used_vars = [4 if element == 'Ttub' else element for element in used_vars]
    used_vars = [5 if element == 'Trai' else element for element in used_vars]
    used_vars = [6 if element == 'Choke' else element for element in used_vars]

    data = scipy.io.loadmat('mats/novos_testes/dados_padrao.mat')
    teste_padrao = data['dados_padrao'][1000:, used_vars]
    teste_padrao_label = data['dados_padrao'][1000:, 7]
    teste_padrao_scaled = scaler.transform(teste_padrao)

    golfa_padrao = np.where(teste_padrao_label == 1)[0][0]
    chk_padrao_true = teste_padrao_scaled[golfa_padrao][-1]

    return teste_padrao_scaled, teste_padrao_label, golfa_padrao, chk_padrao_true


@st.cache
def load_data_testeps(used_vars):

    used_vars = [0 if element == 'Ppdg' else element for element in used_vars]
    used_vars = [1 if element == 'Ptt' else element for element in used_vars]
    used_vars = [2 if element == 'Prb' else element for element in used_vars]
    used_vars = [3 if element == 'Prt' else element for element in used_vars]
    used_vars = [4 if element == 'Ttub' else element for element in used_vars]
    used_vars = [5 if element == 'Trai' else element for element in used_vars]
    used_vars = [6 if element == 'Choke' else element for element in used_vars]

    data = scipy.io.loadmat('mats/novos_testes/dados_ps.mat')
    teste_ps = data['dados_ps'][1000:, used_vars]
    teste_ps_label = data['dados_ps'][1000:, 7]
    teste_ps_scaled = scaler.transform(teste_ps)

    golfa_ps = np.where(teste_ps_label == 1)[0][0]
    chk_ps_true = teste_ps_scaled[golfa_ps][-1]

    return teste_ps_scaled, teste_ps_label, golfa_ps, chk_ps_true


@st.cache
def load_data_testepr(used_vars):

    used_vars = [0 if element == 'Ppdg' else element for element in used_vars]
    used_vars = [1 if element == 'Ptt' else element for element in used_vars]
    used_vars = [2 if element == 'Prb' else element for element in used_vars]
    used_vars = [3 if element == 'Prt' else element for element in used_vars]
    used_vars = [4 if element == 'Ttub' else element for element in used_vars]
    used_vars = [5 if element == 'Trai' else element for element in used_vars]
    used_vars = [6 if element == 'Choke' else element for element in used_vars]

    data = scipy.io.loadmat('mats/novos_testes/dados_pr.mat')
    teste_pr = data['dados_pr'][1000:, used_vars]
    teste_pr_label = data['dados_pr'][1000:, 7]
    teste_pr_scaled = scaler.transform(teste_pr)

    golfa_pr = np.where(teste_pr_label == 1)[0][0]
    chk_pr_true = teste_pr_scaled[golfa_pr][-1]

    return teste_pr_scaled, teste_pr_label, golfa_pr, chk_pr_true


@st.cache
def load_data_testeglm(used_vars):

    used_vars = [0 if element == 'Ppdg' else element for element in used_vars]
    used_vars = [1 if element == 'Ptt' else element for element in used_vars]
    used_vars = [2 if element == 'Prb' else element for element in used_vars]
    used_vars = [3 if element == 'Prt' else element for element in used_vars]
    used_vars = [4 if element == 'Ttub' else element for element in used_vars]
    used_vars = [5 if element == 'Trai' else element for element in used_vars]
    used_vars = [6 if element == 'Choke' else element for element in used_vars]

    data = scipy.io.loadmat('mats/novos_testes/dados_glm.mat')
    teste_glm = data['dados_glm'][1000:, used_vars]
    teste_glm_label = data['dados_glm'][1000:, 7]
    teste_glm_scaled = scaler.transform(teste_glm)

    golfa_glm = np.where(teste_glm_label == 1)[0][0]
    chk_glm_true = teste_glm_scaled[golfa_glm][-1]

    return teste_glm_scaled, teste_glm_label, golfa_glm, chk_glm_true


@st.cache
def load_data_testectrlheavy(used_vars):

    used_vars = [0 if element == 'Ppdg' else element for element in used_vars]
    used_vars = [1 if element == 'Ptt' else element for element in used_vars]
    used_vars = [2 if element == 'Prb' else element for element in used_vars]
    used_vars = [3 if element == 'Prt' else element for element in used_vars]
    used_vars = [4 if element == 'Ttub' else element for element in used_vars]
    used_vars = [5 if element == 'Trai' else element for element in used_vars]
    used_vars = [6 if element == 'Choke' else element for element in used_vars]

    data = scipy.io.loadmat('mats/novos_testes/dados_ctrl.mat')
    teste_ctrl = data['dados_ctrl'][1000:, used_vars]
    teste_ctrl_label = data['dados_ctrl'][1000:, 7]
    teste_ctrl_scaled = scaler.transform(teste_ctrl)

    golfa_ctrl = np.where(teste_ctrl_label == 1)[0][0]
    chk_ctrl_true = teste_ctrl_scaled[golfa_ctrl][-1]

    return teste_ctrl_scaled, teste_ctrl_label, golfa_ctrl, chk_ctrl_true


@st.cache
def load_data_testectrllight(used_vars):

    used_vars = [0 if element == 'Ppdg' else element for element in used_vars]
    used_vars = [1 if element == 'Ptt' else element for element in used_vars]
    used_vars = [2 if element == 'Prb' else element for element in used_vars]
    used_vars = [3 if element == 'Prt' else element for element in used_vars]
    used_vars = [4 if element == 'Ttub' else element for element in used_vars]
    used_vars = [5 if element == 'Trai' else element for element in used_vars]
    used_vars = [6 if element == 'Choke' else element for element in used_vars]

    data = scipy.io.loadmat('mats/novos_testes/dados_ctrl_light.mat')
    teste_ctrl_light = data['dados_ctrl_light'][1000:, used_vars]
    teste_ctrl_light_label = data['dados_ctrl_light'][1000:, 7]
    teste_ctrl_light_scaled = scaler.transform(teste_ctrl_light)

    golfa_ctrl_light = np.where(teste_ctrl_light_label == 1)[0][0]
    chk_ctrl_light_true = teste_ctrl_light_scaled[golfa_ctrl_light][-1]

    return teste_ctrl_light_scaled, teste_ctrl_light_label, golfa_ctrl_light, chk_ctrl_light_true


@st.cache
def load_data_testevarglm(used_vars):

    used_vars = [0 if element == 'Ppdg' else element for element in used_vars]
    used_vars = [1 if element == 'Ptt' else element for element in used_vars]
    used_vars = [2 if element == 'Prb' else element for element in used_vars]
    used_vars = [3 if element == 'Prt' else element for element in used_vars]
    used_vars = [4 if element == 'Ttub' else element for element in used_vars]
    used_vars = [5 if element == 'Trai' else element for element in used_vars]
    used_vars = [6 if element == 'Choke' else element for element in used_vars]

    data = scipy.io.loadmat('mats/novos_testes/dados_muda_glm.mat')
    teste_varglm = data['dados_muda_glm'][1000:, used_vars]
    teste_varglm_label = data['dados_muda_glm'][1000:, 7]
    teste_varglm_scaled = scaler.transform(teste_varglm)

    golfa_varglm = np.where(teste_varglm_label == 1)[0][0]
    chk_varglm_true = teste_varglm_scaled[golfa_varglm][-1]

    return teste_varglm_scaled, teste_varglm_label, golfa_varglm, chk_varglm_true


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


@st.cache
def filter_scaler(data):
    scaler.fit(data)
    treino_y_scaled = scaler.transform(treino_y)
    treino_n_scaled = scaler.transform(treino_n)
    treino_y_filtered = FILTROS.filtro_ngbatch(treino_y_scaled, ng_tol, n_splits)

    treino_n_filtered = FILTROS.filtro_ngbatch(treino_n_scaled, ng_tol, n_splits)

    return scaler, treino_y_filtered, treino_n_filtered


tipo_poco = st.sidebar.selectbox('tipo do poco',
                                 ['dinamico', 'estatico'])
used_vars = st.sidebar.multiselect('variaveis',
                                   ['Ppdg', 'Ptt', 'Prb', 'Prt', 'Ttub', 'Trai', 'Choke'], default=['Ppdg', 'Ptt', 'Prt', 'Trai'])

data_load_state = st.text('Loading data...')
treino_y, treino_y_label, treino_n, treino_n_label = load_data_train(tipo_poco, used_vars)
data_load_state.text("Ready")


st.sidebar.markdown('## Parameters')
neighborhood_function = st.sidebar.selectbox('neighborhood_function',
                                             ['gaussian', 'mexican_hat', 'bubble', 'triangle'])
activation_distance = st.sidebar.selectbox('activation_distance',
                                           ['euclidean', 'cosine', 'manhattan', 'chebyshev'])

normalization = st.sidebar.selectbox('normalization method',
                                     ['MinMaxScaler', 'StandardScaler', 'RobustScaler'])
scaler = globals()[normalization]()

n_splits = st.sidebar.slider('n of splits NG filter', 1, 20, 1, step=1)

ng_tol = st.sidebar.slider('tolerance for NG filter', 0.0, 1.0, 0.1, step=0.01)

n_neurons = st.sidebar.number_input('n_neurons', 1, 150, 9, step=1)
m_neurons = n_neurons
sigma = st.sidebar.number_input('sigma', 0.1, 50.0, 1.5, step=0.1)
learning_rate = st.sidebar.number_input('learning rate', 0.1, 5.0, 0.5, step=0.1)
iterations = st.sidebar.number_input('training iterations', 0, 2000, 500, step=1)

filter_scaler_state = st.text('Filtering and scaling data...')
scaler, treino_y_filtered, treino_n_filtered = filter_scaler(np.vstack([treino_y, treino_n]))
filter_scaler_state.text("Ready")
treino_y_filtered_label = np.ones(len(treino_y_filtered))
treino_n_filtered_label = np.zeros(len(treino_n_filtered))

som_train_state = st.text('Training data...')
som = som_train(n_neurons, m_neurons, np.vstack(
    [treino_y_filtered, treino_n_filtered]), sigma, neighborhood_function, activation_distance, learning_rate)
if st.sidebar.checkbox('PCA initialization'):
    som = som_train(n_neurons, m_neurons, np.vstack(
        [treino_y_filtered, treino_n_filtered]), sigma, neighborhood_function, activation_distance, learning_rate, do_pca=True)
som_train_state.text("Ready")

fig = plt.figure(figsize=(10, 9))


plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
w_y = []
w_n = []
for cnt, xx in enumerate(treino_y_filtered):
    w_y.append(som.winner(xx))  # getting the winner
for cnt, xx in enumerate(treino_n_filtered):
    w_n.append(som.winner(xx))  # getting the winner
w_y = np.array(w_y)
w_n = np.array(w_n)
plt.scatter(w_n[:, 0]+.5, w_n[:, 1]+.5, color='red', marker='+', s=150, label='Initial state')
plt.scatter(w_y[:, 0]+.5, w_y[:, 1]+.5, color='blue', marker='*', s=150, label='End state')
plt.legend(fontsize="x-large", loc=1)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
st.pyplot(fig)

st.write('`{e:2.6f}` Quantization error'.format(
    e=som.quantization_error(np.vstack([treino_y_filtered, treino_n_filtered]))))
st.write('`{e:2.6f}` Topographic error'.format(
    e=som.topographic_error(np.vstack([treino_y_filtered, treino_n_filtered]))))
st.write('len train_y: ', len(treino_y_filtered))
st.write('len train_n: ', len(treino_n_filtered))

# %%
if st.sidebar.checkbox('TESTE PADRAO'):
    st.header('TESTE PADRAO')

    teste_padrao_scaled, teste_padrao_label, golfa_padrao, chk_padrao_true = load_data_testepadrao(
        used_vars)

    bmus_padrao = []
    for xx in teste_padrao_scaled:
        bmus_padrao.append(som.winner(xx))
    bmus_padrao = np.array(bmus_padrao)

    sequencia_bmus_padrao_bool = list((np.diff(bmus_padrao, axis=0) != 0).any(axis=1))
    sequencia_bmus_padrao_bool.append(True)
    sequencia_bmus_padrao = bmus_padrao[sequencia_bmus_padrao_bool]
    indexes_padrao = np.where(sequencia_bmus_padrao_bool)

    q = st.number_input('to point: ', min_value=0, max_value=len(indexes_padrao[0])-1, value=0, step=1, key='_padrao')

    fig_padrao, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(sequencia_bmus_padrao[q, 0]+0.5, sequencia_bmus_padrao[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(sequencia_bmus_padrao[0:q+1, 0]+0.5, sequencia_bmus_padrao[0:q+1, 1]+0.5, '*-', zorder=3)
    q_plot = indexes_padrao[0][q]
    ax2.plot(teste_padrao_scaled[0:q_plot])
    ax2.vlines(indexes_padrao, np.min(teste_padrao_scaled), np.max(teste_padrao_scaled), linestyles='dotted')
    ax2.vlines(golfa_padrao, np.min(teste_padrao_scaled), np.max(teste_padrao_scaled), linestyles='solid', color='red')
    st.pyplot(fig=fig_padrao)
    st.write('golfa com choke: ', chk_padrao_true)
    st.write('choke atual: ', teste_padrao_scaled[indexes_padrao[0][q]][-1])

if st.sidebar.checkbox('TESTE PS'):
    st.header('TESTE PS')
    teste_ps_scaled, teste_ps_label, golfa_ps, chk_ps_true = load_data_testeps(used_vars)

    bmus_ps = []
    for xx in teste_ps_scaled:
        bmus_ps.append(som.winner(xx))
    bmus_ps = np.array(bmus_ps)

    sequencia_bmus_ps_bool = list((np.diff(bmus_ps, axis=0) != 0).any(axis=1))
    sequencia_bmus_ps_bool.append(True)
    sequencia_bmus_ps = bmus_ps[sequencia_bmus_ps_bool]
    indexes_ps = np.where(sequencia_bmus_ps_bool)

    q = st.number_input('to point: ', min_value=0, max_value=len(indexes_ps[0])-1, value=0, step=1, key='_ps')

    fig_ps, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(sequencia_bmus_ps[q, 0]+0.5, sequencia_bmus_ps[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(sequencia_bmus_ps[0:q+1, 0]+0.5, sequencia_bmus_ps[0:q+1, 1]+0.5, '*-', zorder=3)
    q_plot = indexes_ps[0][q]
    ax2.plot(teste_ps_scaled[0:q_plot])
    ax2.vlines(indexes_ps, np.min(teste_ps_scaled), np.max(teste_ps_scaled), linestyles='dotted')
    ax2.vlines(golfa_ps, np.min(teste_ps_scaled), np.max(teste_ps_scaled), linestyles='solid', color='red')
    st.pyplot(fig=fig_ps)
    st.write('golfa com choke: ', chk_ps_true)
    st.write('choke atual: ', teste_ps_scaled[indexes_ps[0][q]][-1])

if st.sidebar.checkbox('TESTE PR'):
    st.header('TESTE PR')
    teste_pr_scaled, teste_pr_label, golfa_pr, chk_pr_true = load_data_testepr(used_vars)

    bmus_pr = []
    for xx in teste_pr_scaled:
        bmus_pr.append(som.winner(xx))
    bmus_pr = np.array(bmus_pr)

    sequencia_bmus_pr_bool = list((np.diff(bmus_pr, axis=0) != 0).any(axis=1))
    sequencia_bmus_pr_bool.append(True)
    sequencia_bmus_pr = bmus_pr[sequencia_bmus_pr_bool]
    indexes_pr = np.where(sequencia_bmus_pr_bool)

    q = st.number_input('to point: ', min_value=0, max_value=len(indexes_pr[0])-1, value=0, step=1, key='_pr')

    fig_pr, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(sequencia_bmus_pr[q, 0]+0.5, sequencia_bmus_pr[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(sequencia_bmus_pr[0:q+1, 0]+0.5, sequencia_bmus_pr[0:q+1, 1]+0.5, '*-', zorder=3)
    q_plot = indexes_pr[0][q]
    ax2.plot(teste_pr_scaled[0:q_plot])
    ax2.vlines(indexes_pr, np.min(teste_pr_scaled), np.max(teste_pr_scaled), linestyles='dotted')
    ax2.vlines(golfa_pr, np.min(teste_pr_scaled), np.max(teste_pr_scaled), linestyles='solid', color='red')
    st.pyplot(fig=fig_pr)
    st.write('golfa com choke: ', chk_pr_true)
    st.write('choke atual: ', teste_pr_scaled[indexes_pr[0][q]][-1])

if st.sidebar.checkbox('TESTE GLM'):
    st.header('TESTE GLM')
    teste_glm_scaled, teste_glm_label, golfa_glm, chk_glm_true = load_data_testeglm(
        used_vars)

    bmus_glm = []
    for xx in teste_glm_scaled:
        bmus_glm.append(som.winner(xx))
    bmus_glm = np.array(bmus_glm)

    sequencia_bmus_glm_bool = list((np.diff(bmus_glm, axis=0) != 0).any(axis=1))
    sequencia_bmus_glm_bool.append(True)
    sequencia_bmus_glm = bmus_glm[sequencia_bmus_glm_bool]
    indexes_glm = np.where(sequencia_bmus_glm_bool)

    q = st.number_input('to point: ', min_value=0, max_value=len(indexes_glm[0])-1, value=0, step=1, key='_glm')

    fig_glm, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(sequencia_bmus_glm[q, 0]+0.5, sequencia_bmus_glm[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(sequencia_bmus_glm[0:q+1, 0]+0.5, sequencia_bmus_glm[0:q+1, 1]+0.5, '*-', zorder=3)
    q_plot = indexes_glm[0][q]
    ax2.plot(teste_glm_scaled[0:q_plot])
    ax2.vlines(indexes_glm, np.min(teste_glm_scaled), np.max(teste_glm_scaled), linestyles='dotted')
    ax2.vlines(golfa_glm, np.min(teste_glm_scaled), np.max(teste_glm_scaled), linestyles='solid', color='red')
    st.pyplot(fig=fig_glm)
    st.write('golfa com choke: ', chk_glm_true)
    st.write('choke atual: ', teste_glm_scaled[indexes_glm[0][q]][-1])

if st.sidebar.checkbox('CTRL HEAVY'):
    st.header('TESTE CTRL HEAVY')
    teste_ctrl_scaled, teste_ctrl_label, golfa_ctrl, chk_ctrl_true = load_data_testectrlheavy(
        used_vars)

    bmus_ctrl = []
    for xx in teste_ctrl_scaled:
        bmus_ctrl.append(som.winner(xx))
    bmus_ctrl = np.array(bmus_ctrl)

    sequencia_bmus_ctrl_bool = list((np.diff(bmus_ctrl, axis=0) != 0).any(axis=1))
    sequencia_bmus_ctrl_bool.append(True)
    sequencia_bmus_ctrl = bmus_ctrl[sequencia_bmus_ctrl_bool]
    indexes_ctrl = np.where(sequencia_bmus_ctrl_bool)

    q = st.number_input('to point: ', min_value=0, max_value=len(indexes_ctrl[0])-1, value=0, step=1, key='_ctrl')

    fig_ctrl, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(sequencia_bmus_ctrl[q, 0]+0.5, sequencia_bmus_ctrl[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(sequencia_bmus_ctrl[0:q+1, 0]+0.5, sequencia_bmus_ctrl[0:q+1, 1]+0.5, '*-', zorder=3)
    q_plot = indexes_ctrl[0][q]
    ax2.plot(teste_ctrl_scaled[0:q_plot])
    ax2.vlines(indexes_ctrl, np.min(teste_ctrl_scaled), np.max(teste_ctrl_scaled), linestyles='dotted')
    ax2.vlines(golfa_ctrl, np.min(teste_ctrl_scaled), np.max(teste_ctrl_scaled), linestyles='solid', color='red')
    st.pyplot(fig=fig_ctrl)
    st.write('golfa com choke: ', chk_ctrl_true)
    st.write('choke atual: ', teste_ctrl_scaled[indexes_ctrl[0][q]][-1])

if st.sidebar.checkbox('CTRL LIGHT'):
    st.header('TESTE CTRL LIGHT')
    teste_ctrl_light_scaled, teste_ctrl_light_label, golfa_ctrl_light, chk_ctrl_light_true = load_data_testectrllight(
        used_vars)

    bmus_ctrl_light = []
    for xx in teste_ctrl_light_scaled:
        bmus_ctrl_light.append(som.winner(xx))
    bmus_ctrl_light = np.array(bmus_ctrl_light)

    sequencia_bmus_ctrl_light_bool = list((np.diff(bmus_ctrl_light, axis=0) != 0).any(axis=1))
    sequencia_bmus_ctrl_light_bool.append(True)
    sequencia_bmus_ctrl_light = bmus_ctrl_light[sequencia_bmus_ctrl_light_bool]
    indexes_ctrl_light = np.where(sequencia_bmus_ctrl_light_bool)

    q = st.number_input('to point: ', min_value=0, max_value=len(
        indexes_ctrl_light[0])-1, value=0, step=1, key='_ctrl_light')

    fig_ctrl_light, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(sequencia_bmus_ctrl_light[q, 0]+0.5, sequencia_bmus_ctrl_light[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(sequencia_bmus_ctrl_light[0:q+1, 0]+0.5, sequencia_bmus_ctrl_light[0:q+1, 1]+0.5, '*-', zorder=3)
    q_plot = indexes_ctrl_light[0][q]
    ax2.plot(teste_ctrl_light_scaled[0:q_plot])
    ax2.vlines(indexes_ctrl_light, np.min(teste_ctrl_light_scaled),
               np.max(teste_ctrl_light_scaled), linestyles='dotted')
    ax2.vlines(golfa_ctrl_light, np.min(teste_ctrl_light_scaled), np.max(
        teste_ctrl_light_scaled), linestyles='solid', color='red')
    st.pyplot(fig=fig_ctrl_light)
    st.write('golfa com choke: ', chk_ctrl_light_true)
    st.write('choke atual: ', teste_ctrl_light_scaled[indexes_ctrl_light[0][q]][-1])

if st.sidebar.checkbox('VAI E VOLTA'):
    st.header('TESTE VAI E VOLTA')
    teste_varglm_scaled, teste_varglm_label, golfa_varglm, chk_varglm_true = load_data_testevarglm(used_vars)

    bmus_varglm = []
    for xx in teste_varglm_scaled:
        bmus_varglm.append(som.winner(xx))
    bmus_varglm = np.array(bmus_varglm)

    sequencia_bmus_varglm_bool = list((np.diff(bmus_varglm, axis=0) != 0).any(axis=1))
    sequencia_bmus_varglm_bool.append(True)
    sequencia_bmus_varglm = bmus_varglm[sequencia_bmus_varglm_bool]
    indexes_varglm = np.where(sequencia_bmus_varglm_bool)

    q = st.number_input('to point: ', min_value=0, max_value=len(
        indexes_varglm[0])-1, value=0, step=1, key='_varglm')

    fig_varglm, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.pcolormesh(som.distance_map().T, cmap='bone_r')
    ax1.scatter(sequencia_bmus_varglm[q, 0]+0.5, sequencia_bmus_varglm[q, 1]+0.5, color='red', zorder=5)
    ax1.plot(sequencia_bmus_varglm[0:q+1, 0]+0.5, sequencia_bmus_varglm[0:q+1, 1]+0.5, '*-', zorder=3)
    q_plot = indexes_varglm[0][q]
    ax2.plot(teste_varglm_scaled[0:q_plot])
    ax2.vlines(indexes_varglm, np.min(teste_varglm_scaled),
               np.max(teste_varglm_scaled), linestyles='dotted')
    ax2.vlines(golfa_varglm, np.min(teste_varglm_scaled), np.max(
        teste_varglm_scaled), linestyles='solid', color='red')
    st.pyplot(fig=fig_varglm)
    st.write('golfa com choke: ', chk_varglm_true)
    st.write('choke atual: ', teste_varglm_scaled[indexes_varglm[0][q]][-1])
