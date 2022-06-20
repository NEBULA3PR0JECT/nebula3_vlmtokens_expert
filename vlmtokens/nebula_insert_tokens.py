from nebula3_database.config import NEBULA_CONF
from nebula3_database.database.arangodb import DatabaseConnector
import json

def load_json(json_path):
    return json.load(open(json_path))

def open_json_file():
    msrvtt_json_path = []
    msrvtt_json = []
    #msrvtt_json_path.append(f'blip_output1/msrvtt_test/visual_tokenization_blip/visual_tokens.json')
    #msrvtt_json_path.append(f'blip_output2/msrvtt_test/visual_tokenization_blip/visual_tokens.json')
    #msrvtt_json_path.append(f'blip_output3/msrvtt_test/visual_tokenization_blip/visual_tokens.json')
    #msrvtt_json_path.append(f'blip_output4/msrvtt_test/visual_tokenization_blip/visual_tokens.json')
    #msrvtt_json_path.append(f'blip_output5/msrvtt_test/visual_tokenization_blip/visual_tokens.json')
    #msrvtt_json_path.append(f'blip_output6/msrvtt_test/visual_tokenization_blip/visual_tokens.json')
    for json_path in msrvtt_json_path:
        #print(json_path)
        json_ = load_json(json_path)
        for v in json_ :
            print(v)
            data = {
            'url': "http://74.82.28.99:9000/msrvtt/"+ v + ".mp4",
            'file': "/datasets/msrvtt/shared_datasets/MSRVTT_ret/videos/" + v + ".mp4",
            'movie_name': v,
            'split': 'test',
            'annotation': json_[v]['caption'],
            'frame_tokens': json_[v]['frame_tokens'],
            'aggregated_tokens': json_[v]['aggregated_tokens']
            }
            msrvtt_json.append(data)
    return(msrvtt_json)

dbc = DatabaseConnector()
db = dbc.connect_db('prodemo')
msrvtt_orig = db.collection('msrvtt_orig')
videos = open_json_file()
for v in videos:
    msrvtt_orig.insert(v)