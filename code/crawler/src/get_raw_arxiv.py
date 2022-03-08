import os
import time
import json
import arxiv


def get_paper_list(query="cat:cs.CL", domain='cscl',
                   latest_year=2021, end_year=2010, max_results=1000):
    outs = {}
    year = latest_year
    i = 0
    while year > end_year: 
        print(f"Results {i} - {i+max_results}:")
        result = arxiv.query(query = query,
                             start = i,
                             sort_by = 'submittedDate',
                             sort_order = 'descending',
                             max_results = max_results)
        new_papers = 0
        for paper in result:
            arxiv_id = paper.id.split('/')[-1]
            N = int(arxiv_id[-1])
            if '.' in arxiv_id and N > 1:
                arxiv_id = arxiv_id.replace(f'v{N}', '')
                print(arxiv_id)
                new_papers += 1
                year = int(paper.updated[:4])
                if arxiv_id not in outs.keys():
                    outs[arxiv_id] = [N]
                else:
                    outs[arxiv_id].append(N)
        i += max_results
        time.sleep(3)
        print(year)
        if new_papers == 0: break
        
    with open(f'../data/arxiv/list/{domain}_list_{len(outs)}.json', 'w') as json_file:
        json.dump(outs, json_file, indent=2)
    return outs

def generate_json_file(preprint_list, tmp_file_path, domain):
    with open(f'{tmp_file_path}/raw_revisions_{domain}.json', 'a') as json_file:
        for ID in preprint_list.keys():
            max_ver = max(preprint_list[ID])
            for i in range(1, max_ver):
                print(ID)
                preprint_v1 = ID+f'v{i}'
                preprint_v2 = ID+f'v{i+1}'
                papers = arxiv.query(query="",
                                     id_list=[preprint_v1,preprint_v2],
                                     max_results=2)
                try:
                    source_abs = papers[0].summary
                    target_abs = papers[1].summary
                except:
                    print(f'Fail to get paper {ID}!!!')
                    continue
                tmp = {
                        "arxiv_id": ID,
                        "before_version": i,
                        "after_version": i+1,
                        "before_raw_txt": source_abs, 
                        "after_raw_txt": target_abs, 
                        }
                time.sleep(3)
                json_file.write(json.dumps(tmp)+'\n')
            


if __name__ == '__main__':
    tmp_path = '../data/arxiv'
    tmp_list_path = '../data/arxiv/list'
    tmp_file_path = '../data/arxiv/raw'
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    if not os.path.isdir(tmp_list_path):
        os.mkdir(tmp_list_path)
    if not os.path.isdir(tmp_file_path):
        os.mkdir(tmp_file_path)
        
    # get raw paper id list (paper version >= 2)
    cates = ['econ.EM', 'econ.GN', 'econ.TH']
    cates += ['q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF', 'q-fin.PM', 'q-fin.PR',
            'q-fin.RM', 'q-fin.ST', 'q-fin.TR']
    cates += ['q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN', 'q-bio.NC', 'q-bio.OT',
              'q-bio.PE', 'q-bio.QM', 'q-bio.SC', 'q-bio.TO']
    cates += ['cs.AI', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.GT', 'cs.CV', 'cs.CY', 'cs.CR',
              'cs.DS', 'cs.DB', 'cs.DL', 'cs.DM', 'cs.DC', 'cs.ET', 'cs.FL', 'cs.GL',
              'cs.GR', 'cs.AR', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LO', 'cs.LG', 'cs.MS',
              'cs.MA', 'cs.MM', 'cs.NI', 'cs.NE', 'cs.NA', 'cs.OS', 'cs.OH', 'cs.PF',
              'cs.PL', 'cs.RO', 'cs.SI', 'cs.SE', 'cs.SD', 'cs.SC', 'cs.SY']
    for cate in cates:
        preprint_list = get_paper_list(query=f"cat:{cate}", domain=f'{cate}',
                                        latest_year=2021, end_year=1900, max_results=1000)
    
    # extract paper abstract by paper id 
    files = os.listdir(tmp_list_path)
    for fname in files:
        if fname == '.DS_Store': continue
        domain = fname.split('_')[0]
        print(domain)
        with open(f'{tmp_list_path}/{fname}', 'r') as f:
            preprint_list = json.load(f)
        outs = generate_json_file(preprint_list, tmp_file_path, domain)
        
