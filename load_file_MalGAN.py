import pandas as pd

def make_used_api_dataframe_with_malware_file(file, api_list):
    used_api_dict = {api:[0] for api in api_list}
    
    with open(file) as f:
        for line in f.readlines():
            api = line.rstrip('\n')
            if api in used_api_dict.keys():
                used_api_dict[api][0] = 1
            else:
                # APIリストにないAPIは新たに追加
                used_api_dict[api] = [1]
    return pd.DataFrame.from_dict(used_api_dict)

