import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from re import I
from typing import Any, Dict

import regex as re
from sentsplit.config import en_config
from sentsplit.segment import SentSplit

import streamlit as st
from annotated_text import annotated_text

INTENT_TO_COLOR = {
    'clarity': '#affae0',
    'coherence': '#fcdea7',
    'fluency': '#c2ffc2',
    'meaning-changed': '#d1d1de',
    'others': '#9accfc',
    'style': '#ffcfe9',
    # 'no_intent': '#9370DB'
}

TO_EDIT_TYPE = {
    'A': 'Addition',
    'R': 'Replacement',
    'D': 'Deletion'
}


class DummyModel:
    def __init__(self):
        self.dummy_revisions = []
        with open(DATA_PATH, 'r') as f:
            for line in f:
                self.dummy_revisions.append(json.loads(line))

    def revise(self, document: Dict) -> Dict:
        '''
        Given a document (Dict), model adds `after_revision` and `edit_actions`:
            {
                "doc_id": "65581384",
                "before_revision": "before document",
                "after_revision": "revised document",
                "edit_actions": [
                    {
                        "type": "R",
                        "before": "has",
                        "after": "had",
                        "start_char_pos": 214,
                        "end_char_pos": 217,
                        "major_intent": "fluency",
                    },
                    ..
                ]
            }
        '''
        for dummy_rev in self.dummy_revisions:
            if dummy_rev['doc_id'] == document['doc_id']:
                assert document['before_revision'] == dummy_rev['before_revision']
                return dummy_rev


@st.cache(allow_output_mutation=True)
def load_sentsplit() -> SentSplit:
    my_config = deepcopy(en_config)
    my_config['maxcut'] = 800
    return SentSplit('en', **my_config)


@st.cache(allow_output_mutation=True)
def load_model(model_path: str = 'dummy') -> Any:
    if model_path == 'dummy':
        return DummyModel()
    else:
        from baseline_model import BaselineModel
        return BaselineModel(model_path)


def do_human_evaluation(doc_id: str, curr_rev_depth: int) -> None:
    def _uncolor_all_edit_actions() -> None:
        '''Remove additional coloring of the text'''
        for i, chunk in enumerate(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks']):
            if isinstance(chunk, tuple):
                # the first three items represent: text, label, and background color
                if len(chunk) > 3:
                    st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][i] = chunk[:3]

    def _color_edit_action(edit_action_index: int) -> None:
        '''Color an edit action in RED so that it stands out for human evaluation'''
        before_repr, arrow_repr, after_repr = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'][edit_action_index]['representation']
        doc_chunk_index = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['edit_idx_to_doc_chunk_idx'][edit_action_index]
        assert st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index] == before_repr
        assert st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index + 1] == arrow_repr
        assert st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index + 2] == after_repr
        # make the current edit action RED (#FF0000)
        st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index] = (*st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index], '#FF0000')
        st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index + 1] = (*st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index + 1], '#FF0000')
        st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index + 2] = (*st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'][doc_chunk_index + 2], '#FF0000')

    def _decrease_current_annotation_index() -> None:
        '''Callback function when "Prev" button is clicked'''
        if st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index'] - 1 >= 0:
            st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index'] -= 1
            _uncolor_all_edit_actions()
            _color_edit_action(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index'])

    def _increase_current_annotation_index() -> None:
        '''Callback function when "Next" button is clicked'''
        if st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index'] + 1 < len(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions']):
            st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index'] += 1
            _uncolor_all_edit_actions()
            _color_edit_action(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index'])

    def _confirm_annotation(acc_or_rej: str) -> None:
        '''Callback function when "Confirm" button is clicked'''
        if acc_or_rej is not None:
            curr_annot_index = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index']
            st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'][curr_annot_index]['accept_or_reject'] = acc_or_rej

    def _get_acc_or_rej_index() -> None:
        '''If Accept or Reject decision is made for current edit_action, return 0 for Reject and 1 for Accept'''
        curr_annot_index = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index']
        if 'accept_or_reject' not in st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'][curr_annot_index]:
            return 0
        if st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'][curr_annot_index]['accept_or_reject'] == 'Reject':
            return 0
        else:
            return 1

    def _save_human_evaluation() -> None:
        '''Callback function when "Submit" button is clicked after evaluation of one document'''
        evaluator_id = st.session_state['evaluator_id']
        curr_rev_depth = st.session_state[doc_id]['curr_rev_depth']
        now = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        output_path = os.path.join('./eval_results', st.session_state["evaluator_id"], f'{evaluator_id}-{eval_mode.upper()}-{doc_id}-cycle_{curr_rev_depth}-{now}.json')
        with open(output_path, 'w') as f:
            json.dump(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'], f, indent=2)
            st.success(f'Results are saved to `{output_path}` !')
            st.balloons()

    def _get_number_of_completed() -> int:
        no_completed = 0
        for i in range(len(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'])):
            if 'accept_or_reject' in st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'][i]:
                no_completed += 1
        return no_completed


    assert 'evaluator_id' in st.session_state
    assert st.session_state['curr_doc_id'] == doc_id
    assert doc_id in st.session_state

    st.session_state[doc_id]['in_human_eval'] = True

    if len(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks']) == 0:
        edit_actions = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions']
        edit_actions = sorted(edit_actions, key=lambda x: x['start_char_pos'])
        before_revision = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['before_revision']

        prev_char_pos = 0
        for index, edit_action in enumerate(edit_actions):
            before = edit_action['before'] if edit_action['before'] is not None else ''
            after = edit_action['after'] if edit_action['after'] is not None else ''
            start_char_pos = edit_action['start_char_pos']
            end_char_pos = edit_action['end_char_pos']
            major_intent = edit_action['major_intent']

            st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'].append(before_revision[prev_char_pos:start_char_pos])

            color = INTENT_TO_COLOR[major_intent]
            repr = [(before, 'before', color), ('→', major_intent, color), (after, 'after', color)]

            st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['edit_idx_to_doc_chunk_idx'][index] = len(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'])
            st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'].extend(repr)
            edit_action['representation'] = repr

            prev_char_pos = end_char_pos

        if prev_char_pos < len(before_revision) - 1:
            st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'].append(before_revision[prev_char_pos:])
        st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'] = edit_actions

    # main document
    doc_no = st.session_state[doc_id]['doc_no']
    st.subheader(f'{doc_no}-{doc_id}: {st.session_state[doc_id][f"rev_depth_{curr_rev_depth}"]["doc"]["before_revision"][:100]}...')
    _uncolor_all_edit_actions()
    _color_edit_action(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index'])

    doc_col, eval_col = st.columns([3, 1])
    with doc_col:
        annotated_text(*st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc_chunks'])
    with eval_col:
        edit_action = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'][st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index']]
        st.subheader(f'{st.session_state[doc_id][f"rev_depth_{curr_rev_depth}"]["curr_annot_index"] + 1}/{len(st.session_state[doc_id][f"rev_depth_{curr_rev_depth}"]["doc"]["edit_actions"])}')
        annotated_text(*edit_action['representation'])
        st.markdown(f'#### {TO_EDIT_TYPE[edit_action["type"]]} by {edit_action["major_intent"].upper()}')

        st.button('Prev', on_click=_decrease_current_annotation_index)

        with st.form('evaluation_form'):
            acc_or_rej = st.radio('Reject or Accept', ['Reject', 'Accept'], _get_acc_or_rej_index())
            confirmed = st.form_submit_button('Confirm')
            if confirmed:
                _confirm_annotation(acc_or_rej)

        if 'accept_or_reject' in st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'][st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['curr_annot_index']]:
            st.button('Next', on_click=_increase_current_annotation_index)
        else:
            st.write('Please confirm to proceed')
            # disabling button does not work sometimes
            # st.button('Next', on_click=_increase_current_annotation_index, disabled=True)

        # if annotator has finished annotation
        num_completed = _get_number_of_completed()
        if num_completed >= len(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions']):
            if st.button('Submit'):
                _save_human_evaluation()
                st.session_state[doc_id]['in_human_eval'] = False
                return True
        else:
            if len(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions']) - num_completed <= 5:
                not_done = [index + 1 for index in range(len(st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']["edit_actions"])) if 'accept_or_reject' not in st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']["edit_actions"][index]]
                st.info(f'Completed: {num_completed}/{len(st.session_state[doc_id][f"rev_depth_{curr_rev_depth}"]["doc"]["edit_actions"])}, Remaining: {not_done}')
            else:
                st.info(f'Completed: {num_completed}/{len(st.session_state[doc_id][f"rev_depth_{curr_rev_depth}"]["doc"]["edit_actions"])}')
    return False


def prepare_revision(doc_id: str) -> Dict:
    '''Given previous rev_depth_x, set before_revision of the current depth accordingly.'''
    curr_rev_depth = st.session_state[doc_id]['curr_rev_depth']
    curr_rev = {
        'curr_annot_index': 0,
        'doc_chunks': [],
        'edit_idx_to_doc_chunk_idx': {},
        'doc': deepcopy(st.session_state[doc_id][f'rev_depth_{curr_rev_depth - 1}']['doc'])
    }
    if curr_rev_depth <= 1:
        return curr_rev
    else:
        curr_rev['doc']['before_revision'] = curr_rev['doc']['after_human_evaluation']
        curr_rev['doc']['after_human_evaluation'] = None
        curr_rev['doc']['after_revision'] = None
        curr_rev['doc']['edit_actions'] = None
    return curr_rev


def reflect_human_evaluation(doc_id: str, curr_rev_depth: int) -> None:
    # go through each edit action and apply to before revision
    edit_actions = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions']
    edit_actions = sorted(edit_actions, key=lambda x: x['start_char_pos'])
    before_revision = st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['before_revision']
    curr_char_pos = 0
    text_chunks = []
    for edit_action in edit_actions:
        if edit_action['accept_or_reject'] != 'Accept':
            before = edit_action['before']
            after = edit_action['after']
            major_intent = edit_action['major_intent']
            edit_type = edit_action['type']
            st.session_state[doc_id]['rejected_edit_actions'].add((before, after, major_intent, edit_type))
            continue

        prior_text = before_revision[curr_char_pos:edit_action['start_char_pos']]
        if prior_text is not None:
            text_chunks.append(prior_text)
        if edit_action['after'] is not None:
            text_chunks.append(edit_action['after'])
        curr_char_pos = edit_action['end_char_pos']
    if curr_char_pos < len(before_revision) - 1 and before_revision[curr_char_pos:] is not None:
        text_chunks.append(before_revision[curr_char_pos:])

    st.session_state[doc_id][f'rev_depth_{curr_rev_depth}']['doc']['after_human_evaluation'] = ''.join(text_chunks)


def filter_already_rejected_edit_actions(sel_doc_id: str, curr_rev_depth: int) -> None:
    filtered_edit_actions = []
    for i, edit_action in enumerate(st.session_state[sel_doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions']):
        before = edit_action['before']
        after = edit_action['after']
        major_intent = edit_action['major_intent']
        edit_type = edit_action['type']
        if (before, after, major_intent, edit_type) not in st.session_state[sel_doc_id]['rejected_edit_actions']:
            filtered_edit_actions.append(edit_action)
    st.session_state[sel_doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions'] = filtered_edit_actions


def save_entire_states(doc_id: str) -> None:
    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError

    evaluator_id = st.session_state['evaluator_id']
    now = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    output_path = os.path.join('./eval_results', st.session_state["evaluator_id"], f'{evaluator_id}-{eval_mode.upper()}-{doc_id}-entire_doc-{now}.json')
    with open(output_path, 'w') as f:
        json.dump(st.session_state[doc_id], f, indent=2, default=set_default)
        st.success(f'All results for document {doc_id} are saved to `{output_path}` !')


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, help='pass either `model` or `human`')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    eval_mode = args.mode

    st.set_page_config(
        page_title='IteraTE',
        layout='wide',
    )

    MAX_REVISION_DEPTH = 3
    DATA_PATH = './doc-level.final.test.doc_id_combined.jsonl'
    # DATA_PATH = 'doc-level.final.test.json'
    if eval_mode == 'model':
        MODEL_PATH = '../models/2021-01-06-multi-intent/checkpoint-54500'
        # MODEL_PATH = '/projects/text-revision/models/baseline_generative/pegasus/2021-01-06-multi-intent/checkpoint-54500'
    else:
        MODEL_PATH = 'dummy'

    original_test_data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            original_sample = json.loads(line)
            if eval_mode == 'model' and original_sample['revision_depth'] != "1":
                continue
            sample = {
                'doc_id': original_sample['doc_id'],
                'before_revision': original_sample['before_revision']
            }
            original_test_data.append(sample)

    sentsplit = load_sentsplit()
    model = load_model(model_path=MODEL_PATH)

    if 'evaluator_id' not in st.session_state:
        with st.form('evaluator_id_form'):
            evaluator_id = st.text_input('Enter evaluator_id')
            st.info('You may have to click "Submit" twice.')
            submitted = st.form_submit_button('Submit')
            if submitted:
                evaluator_id = re.sub(r'\s+', r'_', evaluator_id)
                if len(evaluator_id) <= 2:
                    st.error('The length of "evaluator_id" must be greater than 2 characters')
                else:
                    st.session_state['evaluator_id'] = evaluator_id
                    os.makedirs(f'./eval_results/{evaluator_id}', exist_ok=True)
    else:
        st.success(f'Welcome! Your evaluator_id is: {st.session_state["evaluator_id"]}')
        st.header('Guidelines')
        st.write('In this evaluation, you will go through text revision suggestions, and decide to either `accept` or `reject` each of them.')
        st.write('The suggestions are shown in colored boxes, and the current suggestion to evaluate is further highlighted in red.')
        st.write('Please evaluate each suggestion carefully, and make your decision by clicking either `Accept` or `Reject` radio button and submitting it.')
        st.write('You can perform the evaluation in order and once a set of evaluation is completed, please proceed to the next set by clicking the navigation dropdown bar on the left.')
        annotated_text(*[(intent.upper(), '', color) for intent, color in INTENT_TO_COLOR.items()])

        # 'doc_no:first_sentence' for sidebar items
        doc_titles = [f'{i}:{doc["before_revision"][:50]}' for i, doc in enumerate(original_test_data)]
        # select a document from sidebar
        sel_doc_title = st.sidebar.selectbox('Document', doc_titles)
        # document index from the list
        sel_doc_no = int(sel_doc_title.split(':', 1)[0])
        sel_doc_id = original_test_data[sel_doc_no]['doc_id']

        if 'curr_doc_id' not in st.session_state:
            st.session_state['curr_doc_id'] = sel_doc_id

        st.session_state['curr_doc_id'] = sel_doc_id

        if sel_doc_id not in st.session_state:
            st.session_state[sel_doc_id] = {
                # document index from the list
                'doc_no': sel_doc_no,
                # current revision depth
                'curr_rev_depth': 1,
                'rejected_edit_actions': set(),
                'rev_depth_0': {
                    # current index of edit revision
                    'curr_annot_index': 0,
                    # document chunks to color annotations
                    'doc_chunks': [],
                    # dictionary to map edit action index to document chunk index
                    'edit_idx_to_doc_chunk_idx': {},
                    # document, a dict, containing "doc_id", "before_revision", "after_revision", and "edit_actions"
                    'doc': original_test_data[sel_doc_no]
                }
            }

        if 'in_human_eval' not in st.session_state[sel_doc_id]:
            st.session_state[sel_doc_id]['in_human_eval'] = False

        st.subheader(f'current revision cycle: {st.session_state[sel_doc_id]["curr_rev_depth"]}')

        if st.session_state[sel_doc_id]['curr_rev_depth'] <= MAX_REVISION_DEPTH:
            curr_rev_depth = st.session_state[sel_doc_id]['curr_rev_depth']
            if not st.session_state[sel_doc_id]['in_human_eval']:
                with st.spinner('In progress..'):
                    st.session_state[sel_doc_id][f'rev_depth_{curr_rev_depth}'] = prepare_revision(sel_doc_id)
                    curr_doc_for_rev = st.session_state[sel_doc_id][f'rev_depth_{curr_rev_depth}']['doc']
                    model_revised_doc = model.revise(curr_doc_for_rev)
                    st.session_state[sel_doc_id][f'rev_depth_{curr_rev_depth}']['doc'] = model_revised_doc
                    filter_already_rejected_edit_actions(sel_doc_id, curr_rev_depth)
            if len(st.session_state[sel_doc_id][f'rev_depth_{curr_rev_depth}']['doc']['edit_actions']) == 0:
                st.info(f'No more revisions needed for this document. Evaluation is complete for document-{sel_doc_id}. Please proceed to next document.')
                save_entire_states(sel_doc_id)
            else:
                if do_human_evaluation(sel_doc_id, curr_rev_depth):
                    reflect_human_evaluation(sel_doc_id, curr_rev_depth)
                    st.session_state[sel_doc_id]['curr_rev_depth'] += 1
                    _, eval_col = st.columns([3, 1])
                    with eval_col:
                        st.button('NEXT ITERATION!')
        else:
            st.info(f'Maximum revision cycles are reached. Evaluation is complete for document-{sel_doc_id}. Please proceed to next document.')
            save_entire_states(sel_doc_id)

        with st.expander('Show current state'):
            st.write(st.session_state)