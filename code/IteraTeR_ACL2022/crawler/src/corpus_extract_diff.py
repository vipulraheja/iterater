import os
import re
import json
import argparse
import pylatexenc
from pylatexenc.latexwalker import LatexWalker
from nltk.tokenize import word_tokenize
from utils import clean_unused, clean_text, clean_abstract
import numpy as np


def remove_empty(add_list):
    while '' in add_list:
        add_list.remove('')
    return add_list

def write_to_latex(temp_path, source_abs, preprint_v1):
    with open(f'{temp_path}/{preprint_v1}.tex', 'w') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\begin{document}\n')
        f.write('\\begin{abstract}\n')
        f.write(source_abs)
        f.write('\\end{abstract}\n')
        f.write('\\end{document}\n')
    return f'{temp_path}/{preprint_v1}.tex'

def generate_latex_file(ID, ver_id, before_revision, after_revision, temp_path = '../../temp_wiki'):
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
        
    preprint_v1 = f'{ID}v{ver_id}'
    preprint_v2 = f'{ID}v{ver_id+1}'
    source_abs = before_revision.replace('%', '\%')
    target_abs = after_revision.replace('%', '\%')
    source_file = write_to_latex(temp_path, source_abs, preprint_v1)
    target_file = write_to_latex(temp_path, target_abs, preprint_v2)
    latexdiff_command_tex = "latexdiff "
    latexdiff_command_tex += "--ignore-warnings "
    latexdiff_command_tex += "--math-markup=0 "
    latexdiff_command_tex += source_file + " " + target_file + " > " + f'{temp_path}/{ID}_diff_v{ver_id}v{ver_id+1}.tex'
    print(latexdiff_command_tex)
    try:
        os.system(latexdiff_command_tex)
    except:
        print('Skip!!!')
    # delete non diff files
    os.remove(f'{temp_path}/{ID}v{ver_id}.tex')
    os.remove(f'{temp_path}/{ID}v{ver_id+1}.tex')
                
def extract_abstract(file_path):
    with open(file_path, 'r') as f:
        latex = f.read()
    searchObj = re.search('begin{abstract}', latex)
    try:
        sid = searchObj.end()
    except:
        sid = 0
    eid = len(latex)
    abstract = latex[sid: sid+eid-1].strip('[ \n]')
    abstract = abstract.replace('begin{abstract}', '')
    abstract = re.sub(r'\$', 'MATH', abstract)
    abstract = re.sub(r'\\newcite{.*}', 'CITATION', abstract)
    
    paras = abstract.split('\n\n')
    outs = []
    for para in paras:
        if 'DIFdel' in para or '\DIFadd' in para:
            outs.append(para)
    abstract = '\nNEW_PARAGRAPH\n'.join(outs)
    return abstract
    
def extract_abstract_raw(diff_out, mode='DIFdel'):
    abstract = []
    pos_arr = np.zeros(len(diff_out))
    for i, diff in enumerate(diff_out):
        if diff in ['DIFdel', 'DIFadd']:
            if i+1 == len(diff_out): continue
            if diff == mode:
                abstract.append(diff_out[i+1])
                pos_arr[i+1] = 1
            else:
                pos_arr[i+1] = -1
        else:
            if pos_arr[i] == 0:
                abstract.append(diff)
    return ' '.join(abstract)

def parse_diffs(ID, ver_id, temp_path = '../data/temp'):
    tmp = {}
    diff_file_path = f'{temp_path}/{ID}_diff_v{ver_id}{ver_id+1}.tex'
    diff_abs = extract_abstract(diff_file_path)
    (nodes, pos, len_) = LatexWalker(diff_abs).get_latex_nodes(pos=0)
    
    all_nodes = []
    for i, node in enumerate(nodes):
        if node.isNodeType(pylatexenc.latexwalker.LatexGroupNode):
            prev_node = nodes[i-1]
            if prev_node.isNodeType(pylatexenc.latexwalker.LatexMacroNode) and prev_node.macroname in ['DIFdel', 'DIFadd', 'newcite']:
                for j, char_node in enumerate(node.nodelist):
                    if char_node.isNodeType(pylatexenc.latexwalker.LatexCharsNode):
                        all_nodes += [prev_node, char_node]
                    elif char_node.isNodeType(pylatexenc.latexwalker.LatexGroupNode):
                        for k, cchar_node in enumerate(char_node.nodelist):
                            if cchar_node.isNodeType(pylatexenc.latexwalker.LatexCharsNode):
                                prev_char_node = char_node.nodelist[k-1]
                                if prev_char_node.isNodeType(pylatexenc.latexwalker.LatexMacroNode) and prev_char_node.macroname in ['DIFdel', 'DIFadd']:
                                    all_nodes += [prev_char_node, cchar_node]
                                else:
                                    all_nodes += [prev_node, cchar_node]
                                    
        if node.isNodeType(pylatexenc.latexwalker.LatexCharsNode):
            all_nodes += [node]
            
    diff_out, delete_list, add_list = [], [], []
    for i, node in enumerate(all_nodes):
        if node.isNodeType(pylatexenc.latexwalker.LatexCharsNode):
            chars = node.chars
            chars = chars.replace('MATH', '')
            chars = chars.replace('DIFdelbegin', '')
            char_str = ' '.join(word_tokenize(chars))
            if len(char_str) > 0:
                diff_out.append(char_str.replace('NEW_PARAGRAPH', '\n**NEW_PARAGRAPH**\n'))
        elif node.isNodeType(pylatexenc.latexwalker.LatexMacroNode):
            if node.macroname == 'DIFdel':
                diff_out.append(node.macroname)
                chars = all_nodes[i+1].chars
                chars = chars.replace('MATH', '')
                chars = chars.replace('DIFdelbegin', '')
                char_str = ' '.join(word_tokenize(chars))
                delete_list.append(clean_unused(char_str))
            if node.macroname == 'DIFadd':
                diff_out.append(node.macroname)
                chars = all_nodes[i+1].chars
                chars = chars.replace('MATH', '')
                chars = chars.replace('DIFdelbegin', '')
                char_str = ' '.join(word_tokenize(chars))
                add_list.append(clean_unused(char_str))
              
    delete_list = remove_empty(delete_list)
    add_list = remove_empty(add_list)
    
    source_abs = extract_abstract_raw(diff_out, mode='DIFdel')
    target_abs = extract_abstract_raw(diff_out, mode='DIFadd')
    
    if len(diff_out) < 1: 
        return tmp
    
    tmp = {
            "before_raw_txt": source_abs, 
            "after_raw_txt": target_abs, 
            "diff_out": diff_out,
            "delete_list": delete_list,
            "add_list": add_list
            }
    return tmp


def parse_diff(fpath='../data/31K_revisions', domain='wiki'):
    temp_path = f'{fpath}/temp_{domain}'
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
        
    files = os.listdir(fpath)
    for fname in files:
        if domain in fname:
            with open(f'{fpath}/{fname}', 'r') as f:
                all_docs = f.read().strip().split('\n')
            break
        
    for i, line in enumerate(all_docs):
        line = json.loads(line)
        doc_id = line['doc_id']
        version_depth = line['version_depth']
        if domain == 'arxiv':
            before_revision = clean_abstract(line['before_revision'])
            after_revision = clean_abstract(line['after_revision'])
        else:
            before_revision = clean_text(line['before_revision'])
            after_revision = clean_text(line['after_revision'])
        generate_latex_file(doc_id, version_depth, before_revision, after_revision, temp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default='wiki', type=str,
                        help="Specify document domain, please select from: arxiv, wiki, news")
    args = parser.parse_args()
        
    parse_diff(fpath='../data/31K_revisions', domain=args.domain)
    