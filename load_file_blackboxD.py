import pandas as pd
from pathlib import Path

def make_used_api_dataframe(folder, api_list):
    used_api_dict = {api:[] for api in api_list}
    file_name = []
       
    for file in sorted(Path(folder).glob('*.txt')):
        with file.open() as f:
            for api in api_list:
                used_api_dict[api].append(0) 
            for line in f.readlines():
                api = line.rstrip('\n')
                if api in used_api_dict.keys():
                    used_api_dict[api][len(used_api_dict[api])-1] = 1
            file_name.append(str(file))
    return pd.DataFrame.from_dict(used_api_dict), file_name
