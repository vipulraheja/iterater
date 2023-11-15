import re
import json
import string
import difflib
from datetime import datetime


def read_data(file_path):
    lines = []
    try:
        with open(file_path, 'r') as f:
            lines = json.load(f)
    except:
        with open(file_path, 'r') as f:
            data = f.read().strip().split('\n')
        for line in data:
            lines.append(json.loads(line))
    return lines

def clean_unused(text):
    text_clean = text.translate(str.maketrans('', '', string.punctuation))
    return text_clean.strip()

def diff_strings(a, b):
    matcher = difflib.SequenceMatcher(None, a, b)
    eid = 0
    outs = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'insert':
            text = clean_unused(b[b0:b1])
            if len(text) > 0:
                eid += 1
                outs.append(text)
        elif opcode == 'delete':
            text = clean_unused(a[a0:a1])
            if len(text) > 0:
                eid += 1
                outs.append(text)
        elif opcode == 'replace':
            textb = clean_unused(b[b0:b1])
            texta = clean_unused(a[a0:a1])
            if len(texta) > 0 or len(textb) > 0:
                eid += 1
                outs.append(texta)
                outs.append(textb)
    return eid, outs

def get_time_range():
    my_date = datetime.now()
    end = my_date.isoformat()
    start_date = my_date.replace(year=my_date.year - 10)
    start = start_date.isoformat()
    return start, end

def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    return num

def remove_duplicates(l):
    l.sort(key=lambda x: x['timestamp'])
    seen = set()
    new_l = []
    for d in l:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    return new_l

def clean_text(text, domain='wiki'): 
    try:
        if domain == 'wiki': eid = re.search('\n See also', text).start()
        if domain == 'news': eid = re.search('Sources', text).start()
    except:
        eid = -1
    text = text[0: eid].strip('[ \n]')
    text = text.replace('thumb|', '')
    text = re.sub('right\|+[\d\.]+px\|', '',  text)
    text = re.sub('left\|+[\d\.]+px\|', '',  text)
    text = re.sub('upright=+[\d\.]\|', '', text)
    text = text.replace('left|', '')
    text = text.replace('right|', '')
    text = text.replace('https : //', 'https://')
    text = text.replace('http : //', 'https://')
    text = re.sub('<a.*?>|</a>', '', text, flags=re.MULTILINE) # remove <a href=''></a>
    text = re.sub(r"\S*https?:\S*", "URL", text) # replace https://xxx.com with URL
    return text.strip('\n').strip()

def clean_abstract(text): 
    text = text.replace('MATH', '')
    text = text.replace('$', '')
    text = text.replace('https : //', 'https://')
    text = text.replace('http : //', 'https://')
    text = re.sub('<a.*?>|</a>', '', text, flags=re.MULTILINE) # remove <a href=''></a>
    text = re.sub(r"\S*https?:\S*", "URL", text) # replace https://xxx.com with URL
    return text.strip('\n').strip()