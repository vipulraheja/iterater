import json
import os
import re
from copy import deepcopy
from pprint import pprint
from typing import Dict, List

from sentsplit.config import en_config
from sentsplit.segment import SentSplit


def split_and_align(samples: List[Dict], outfile_path: str):
    def _handle_each_action(whole_sentence: str, sentences: List[str], action: Dict, char_to_sent: Dict, sent_to_char: Dict, doc_id: str, revision_depth: int) -> Dict:
        start_char_pos = action['start_char_pos']
        end_char_pos = action['end_char_pos']
        start_sent_index = char_to_sent[start_char_pos]
        end_sent_index = char_to_sent[end_char_pos - 1] if end_char_pos > 0 else 0
        # if start_sent_index != end_sent_index:
        #     print(f'Sentences are merged, {start_sent_index}:{end_sent_index}')
        before_sentence = ''.join(sentences[start_sent_index:end_sent_index + 1])

        start_char_pos_sent_level = start_char_pos - sent_to_char[start_sent_index]
        end_char_pos_sent_level = end_char_pos - sent_to_char[start_sent_index]
        before_revision_part = before_sentence[start_char_pos_sent_level:end_char_pos_sent_level]
        assert before_revision_part == whole_sentence[start_char_pos:end_char_pos]
        action['doc_id'] = doc_id
        action['revision_depth'] = revision_depth
        action['start_char_pos_sent_level'] = start_char_pos_sent_level
        action['end_char_pos_sent_level'] = end_char_pos_sent_level
        action['start_sent_index'] = start_sent_index
        action['end_sent_index'] = end_sent_index
        after_sentence = get_after_revision(before_sentence, action)
        action['before_sent'] = before_sentence
        action['after_sent'] = after_sentence
        action['sent_start_char_pos'] = sent_to_char[start_sent_index]
        # print(before_sentence)
        # print(after_sentence)
        # pprint(action)
        # input('next')
        return action

    def get_after_revision(before_revision: str, edit_action: Dict) -> str:
        """Construct after_revision from before_revision and edit_action"""
        chunks = [before_revision[:edit_action['start_char_pos_sent_level']]]
        if edit_action['type'] == 'R':
            chunks.append(edit_action['after'])
        elif edit_action['type'] == 'D':
            pass
        else:
            chunks.append(edit_action['after'])
        chunks.append(before_revision[edit_action['end_char_pos_sent_level']:])
        after_revision = ''.join(chunks)
        after_revision = re.sub(r' {2,}', ' ', ''.join(chunks))
        return after_revision

    def _split_and_align(sample: Dict) -> List[Dict]:
        before = sample['before_revision']
        before_sents = splitter.segment(before)
        char_to_sent = {}
        sent_to_char = {}
        cuml_char_index = 0
        for sent_index, sent in enumerate(before_sents):
            sent_to_char[sent_index] = cuml_char_index
            for _ in range(len(sent)):
                char_to_sent[cuml_char_index] = sent_index
                cuml_char_index += 1
        char_to_sent[cuml_char_index] = sent_index

        edit_actions = sample['edit_actions']
        doc_id = sample['doc_id']
        revision_depth = sample['revision_depth']
        sent_level_actions = []
        for action in edit_actions:
            sent_level_action = _handle_each_action(before, before_sents, action, char_to_sent, sent_to_char, doc_id, revision_depth)
            sent_level_actions.append(sent_level_action)
        return sent_level_actions

    all_sent_level_actions = 0
    with open(outfile_path, 'w') as f:
        for sample in samples:
            sent_level_actions = _split_and_align(sample)
            for sent_act in sent_level_actions:
                f.write(json.dumps(sent_act)+'\n')
                all_sent_level_actions += 1
    print(all_sent_level_actions)
    return all_sent_level_actions


if __name__ == '__main__':
    def process_dir(bname_path: str) -> None:
        print(bname_path)

        samples = []
        with open(bname_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            samples.append(json.loads(line))
        
        len_edit_actions = 0
        for sample in samples:
            len_edit_actions += len(sample['edit_actions'])
        # print(len(samples), len_edit_actions)
        
        outfile_path = os.path.join(base_dir, 'sent' + os.path.basename(bname_path)[3:])
        all_sent_level_actions = split_and_align(samples, outfile_path)
        

    my_config = deepcopy(en_config)
    my_config['mincut'] = 10
    splitter = SentSplit('en', **my_config)


    base_dir = '../data/200K_revisions'
    base_names = ['doc-level_arxiv.train.json', 'doc-level_arxiv.dev.json', 'doc-level_arxiv.test.json',
                  'doc-level_wiki.train.json', 'doc-level_wiki.dev.json', 'doc-level_wiki.test.json',
                  'doc-level_news.train.json', 'doc-level_news.dev.json', 'doc-level_news.test.json']
    for bname in base_names:
        bname_path = os.path.join(base_dir, bname)
        process_dir(bname_path)