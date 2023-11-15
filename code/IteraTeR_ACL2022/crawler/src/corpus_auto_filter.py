import os
import json
import argparse
from utils import read_data, remove_duplicates, clean_text, clean_abstract


def get_timestamp_list(revisions):
    timestamp_list = []
    for rev in revisions:
        timestamp_list.append(rev['timestamp'])
    return timestamp_list

def wiki_merge_all(domain):
    tmp_path = f'../data/{domain}/auto'
    if not os.path.isdir(f'{tmp_path}'):
        os.mkdir(f'{tmp_path}')
        
    if domain == 'wiki':
        cates = ['culture', 'geography', 'health', 'history', 'human', 'nature', 
                 'people', 'philosophy', ]
    else:
        cates = ['all']
        
    f = open(f'../data/{domain}/raw/problems.txt', 'a')
    for category in cates:
        print(category)
        all_dict = {}
        counts = 0
        files = os.listdir(f'../data/{domain}/raw/{category}')
        for fname in files:
            if fname=='.DS_Store': continue
            print(fname)
            lines = read_data(f'../data/{domain}/raw/{category}/{fname}')
            if type(lines) is str:
                f.write(lines+'\n')
            else:
                for line in lines:
                    before_revision = clean_text(line['parent_content'], domain)
                    after_revision = clean_text(line['cur_content'], domain)
                    
                    # filter documents which do not have enough edits
                    if 'Category:' in line['title']: continue
                    if 'List of ' in line['title']: continue
                    if before_revision == after_revision: continue
                
                    tmp = {'revid': line['revid'],
                            'timestamp': line['timestamp'],
                            'title': line['title'],
                            'before_revision': before_revision,
                            'after_revision': after_revision,
                            }
                    
                    if line['pageid'] not in all_dict.keys(): # new page
                        all_dict[line['pageid']] = [tmp]
                        counts += 1
                    else:
                        all_dict[line['pageid']] += [tmp]
        
        with open(f'{tmp_path}/{domain}_{category}_raw_{counts}.json', 'w') as json_file:
            json.dump(all_dict, json_file, indent=2)
    f.close()
    return tmp_path
            
def arxiv_merge_all(domain):
    tmp_path = f'../data/{domain}/auto'
    if not os.path.isdir(f'{tmp_path}'):
        os.mkdir(f'{tmp_path}')
        
    all_dict = {}
    counts = 0
    files = os.listdir(f'../data/{domain}/raw')
    for fname in files:
        if fname=='.DS_Store': continue
        print(fname)
        lines = read_data(f'../data/{domain}/raw/{fname}')
        for line in lines:
            tmp = {'arxiv_id': line['arxiv_id'],
                    'timestamp': line['before_version'],
                    'before_revision': clean_abstract(line['before_raw_txt']),
                    'after_revision': clean_abstract(line['after_raw_txt']),
                    }
            
            if line['arxiv_id'] not in all_dict.keys(): # new page
                all_dict[line['arxiv_id']] = []
                all_dict[line['arxiv_id']].append(tmp)
                counts += 1
            else:
                all_dict[line['arxiv_id']].append(tmp)
    
    with open(f'{tmp_path}/{domain}_raw_{counts}.json', 'w') as json_file:
        json.dump(all_dict, json_file, indent=2)
    return f'{tmp_path}/{domain}_raw_{counts}.json'

def extract_rev_history(domain, tmp_path):
    docs = []
    fpath = f'../data/{domain}/auto/'
    files = os.listdir(fpath)
    
    with open(f'{tmp_path}/raw_{args.domain}.json', 'a') as json_file:
        for fname in files:
            if fname == '.DS_Store': continue
            with open(f'{fpath}/{fname}', 'r') as f:
                all_dict = json.load(f)
            print(fname)
            
            counts = 0
            for doc_id, rev_list in all_dict.items():
                if doc_id in docs: continue
            
                if len(rev_list) > 1:
                    rev_list = remove_duplicates(rev_list)
                    
                for i, rev in enumerate(rev_list):
                    tmp = {'doc_id': doc_id,
                           'version_depth': i+1,
                           'before_revision': rev['before_revision'],
                           'after_revision': rev['after_revision'],
                           }
                    json_file.write(json.dumps(tmp)+'\n')
                    counts += 1
                docs.append(doc_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default='arxiv', type=str,
                        help="Specify document domain, please select from: arxiv, wiki, news")
    args = parser.parse_args()
    
    # merge all seperate raw revision files into a json file
    if args.domain in ['wiki', 'news']:
        fpath = wiki_merge_all(args.domain)
    else:
        fpath = arxiv_merge_all(args.domain)
    
    # sort revision history of each document by revision timestamps 
    tmp_path = '../data/31K_revisions'
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    extract_rev_history(args.domain, tmp_path)
    