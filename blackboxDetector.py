#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, pickle
import load_file_blackboxD


def predict(folder, classify_algo):
    # モデルの学習時に特徴量にとったAPIリストの読み込み
    with open('blackboxD_api_list.txt') as f:
        api_list = [api.rstrip('\n') for api in f]
    
    # 対象ファイルの特徴量の抽出
    X_test, file_name = load_file_blackboxD.make_used_api_dataframe(folder, api_list)
    
    # 学習済みモデルのロード
    model = pickle.load(open('blackboxD_model_'+classify_algo+'.pkl', 'rb'))
    
    # ラベルの予測
    y_pred = model.predict(X_test)
    
    # 結果の表示
    for i, resoult in enumerate(y_pred):
        if resoult == 0:
            print(file_name[i]+' is cleanware')
        elif resoult == 1:
            print(file_name[i]+' is malware')
            
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="classify algorithm")
    parser.add_argument("--folder", type=str, help="folder_path")
    args = parser.parse_args()
    return args
            
    
if __name__ == '__main__':
    args = get_args()
    predict(args.folder, args.algo)
    