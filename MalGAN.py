#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path, argparse, subprocess, numpy as np, pandas as pd, tensorflow as tf
from keras.backend import tensorflow_backend
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, PReLU, BatchNormalization, Dropout, Input, Layer
from keras.optimizers import Adam
from keras import backend as K
import load_file_MalGAN
mal_feature = None


class CustomVariationalLayer(Layer):
    def custom_rmse_loss(self, x):
        global mal_feature
        return K.sqrt(K.mean(K.sum(K.square((x + 1.0) / 2.0 - mal_feature), axis=1))) * 0.01
    def call(self, inputs):
        x = inputs
        loss = self.custom_rmse_loss(x)
        self.add_loss(loss, inputs=inputs)
        return x
    
    
def set_trainable(model, trainable=False):
    model.trainable = trainable
    try:
        layers = model.layers
    except:
        return
    for layer in layers:
        set_trainable(layer, trainable)

        
def generator_model(feature_dim):
    input_layer = Input((feature_dim, ))
    x = Dense(1024)(input_layer)
    x = PReLU()(x)
    x = Dense(512)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(128)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(feature_dim, activation='tanh')(x)
    x = CustomVariationalLayer()(x) 
    return Model(input_layer, x, name='generator')
    

def substituteD_model(feature_dim):
    input_layer = Input((feature_dim, ))
    x = Dense(128)(input_layer)
    x = PReLU()(x)
    x = Dense(256)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dense(512)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dense(1024)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, x, name='discriminator')


def GAN(feature_dim):
    d = substituteD_model(feature_dim)
    d_opt = Adam(lr=2e-4, beta_1=0.5)
    d.compile(optimizer=d_opt, loss='binary_crossentropy', metrics=['acc'])
    
    g = generator_model(feature_dim)
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    g.compile(optimizer=g_opt, loss='binary_crossentropy', metrics=['acc'])
    
    set_trainable(d, False)
    d_on_g = Sequential([g, d])
    d_on_g.compile(optimizer=g_opt, loss='binary_crossentropy', metrics=['acc'])
    return g, d, d_on_g


def train(malware):
    # 30個の良性ファイルから抽出したAPIリストの読み込み
    with open('discriminator_api_list.txt') as f:
        api_list = [api.rstrip('\n') for api in f]
    
    # マルウェアからAPIを抽出
    X_malware = load_file_MalGAN.make_used_api_dataframe_with_malware_file(malware, api_list)
    X_malware_ndarray = X_malware.values
    feature_dim = len(X_malware.columns)
    
    # カスタム損失関数用に値を保持
    global mal_feature
    mal_feature = np.asarray(X_malware_ndarray[0], dtype=np.float32)
    
    # 学習モデルのビルド
    g, d, d_on_g = GAN(feature_dim)
   
    # トレーニング
    MAX_EPOCH = 200 
    CREATE_FILE_NUM = 200
    for epoch in range(MAX_EPOCH):
        ### Generate
        # ノイズの生成
        Z = np.random.uniform(-1, 1, (CREATE_FILE_NUM, feature_dim))
        # Generatorによる特徴量の生成と, 値の範囲の変更 [-1,1] -> [0,1]
        X_gen = (g.predict(Z) + 1.0) / 2.0
        # マルウェアの特徴量を追加
        for file_num in range(CREATE_FILE_NUM):
            for i in range(feature_dim):
                X_gen[file_num][i] = int(np.round(X_gen[file_num][i])) | X_malware_ndarray[0][i]

        ### Save
        # 対応するAPIをtxtに保存
        save_gen_data = 'gen/'
        if not os.path.exists(save_gen_data):
            os.mkdir(save_gen_data)
        save_gen_data += str(epoch).zfill(3)
        if not os.path.exists(save_gen_data):
            os.mkdir(save_gen_data)
            
        for file_num in range(CREATE_FILE_NUM):
            with open(save_gen_data+'/gen_data_'+str(file_num).zfill(3)+'.txt', 'w') as f:
                for i, api in enumerate(X_malware.columns):
                    if X_gen[file_num][i] == 1:
                        f.write(api+'\n')

        ### Labeling by blackboxD
        # subprocessによるコマンド実行
        cmd = 'python blackboxDetector.py --algo LR --folder '+save_gen_data
        cmd_resoult = subprocess.check_output(cmd.split()).decode('utf-8')
        print(cmd_resoult)
        
        # ラベルを保存
        y_gen = []
        X_gen_clean = []
        y_gen_clean = []
        for i, line in enumerate(cmd_resoult.split('\n')):
            if 'is cleanware' in line:
                y_gen.append(0)
                if len(X_gen_clean) == 0:
                    X_gen_clean = np.array([X_gen[i]])
                else:
                    X_gen_clean = np.concatenate([X_gen_clean, np.array([X_gen[i]])], axis=0)
                y_gen_clean.append(0)
            elif 'is malware' in line:
                y_gen.append(1)
                
        gen_loss = 0
        gen_acc = 0
        subD_loss = 0
        subD_acc = 0
        repet_num = 20
        for i in range(repet_num):
            ### Substitute Detector training
            set_trainable(d, True)
            subD_loss_now, subD_acc_now = d.train_on_batch(X_gen * 2.0 - 1.0, np.array(y_gen))
            set_trainable(d, False)
            subD_loss += subD_loss_now
            subD_acc  += subD_acc_now

            ### Generator training
            if len(y_gen_clean) > 0:
                gen_loss_now, gen_acc_now = d_on_g.train_on_batch(X_gen_clean * 2.0 - 1.0, np.array(y_gen_clean))
            gen_loss += gen_loss_now
            gen_acc  += gen_acc_now
            print('gen_loss = {0:.4f}, subD_loss = {1:.4f}, gen_acc = {2:.4f}, subD_acc = {3:.4f}'
                  .format(gen_loss_now, subD_loss_now, gen_acc_now, subD_acc_now))

        ### Print loss and accuracy
        print('Generator training loss:\t{0}'.format(gen_loss / repet_num))
        print('SubstituteD training loss:\t{0}'.format(subD_loss / repet_num))
        print('Generator training accuracy:\t{0}'.format(gen_acc / repet_num))
        print('SubstituteD training accuracy:\t{0}'.format(subD_acc / repet_num))

    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--malware", type=str, help="malware file_path")
    args = parser.parse_args()
    return args
               
    
if __name__ == '__main__':
    np.random.seed(1)
    tf.set_random_seed(1)
    args = get_args()
    train(args.malware)
    