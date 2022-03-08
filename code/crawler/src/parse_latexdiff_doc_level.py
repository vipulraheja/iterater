import json
import os
import re
import sys
from copy import deepcopy
from glob import glob
from typing import Dict, List, Tuple

from sentsplit.config import en_config
from sentsplit.segment import SentSplit

PUNCTUATIONS = set(['.', '?', '!'])


class LatexdiffParser:
    def __init__(self, latexdiff_dir: str, output_json_path: str):
        self.latexdiff_dir = latexdiff_dir
        self.output_json_path = output_json_path
        self.anchor_symbol = 'ൠ'

    def parse_and_save(self) -> int:
        diff_file_paths = sorted(glob(os.path.join(self.latexdiff_dir, '*_diff_*.tex')))
        parsed_samples = []
        skipped_files = []
        for diff_f_p in diff_file_paths:
            parsed_sample = self.parse_latex_diff(diff_f_p)
            if parsed_sample is None:
                skipped_files.append(diff_f_p)
            else:
                parsed_samples.append(parsed_sample)
        print(f'total no. of parsed samples: {len(parsed_samples)}')
        print(f'total no. of skipped samples: {len(skipped_files)}')
        if len(skipped_files) > 0:
            with open(self.output_json_path + '.skipped', 'w') as f:
                for skipped in skipped_files:
                    f.write(f'{skipped}\n')

        # save as json file
        with open(self.output_json_path, 'w') as f:
            json.dump(parsed_samples, f, indent=2)
        return len(parsed_samples)

    def parse_latex_diff(self, latex_diff_path: str) -> Dict:
        """Parse single latexdiff file"""
        print(f'Parsing {latex_diff_path}')
        with open(latex_diff_path) as f:
            raw_text = f.read()

        # parse doc_id and revision depth
        rgx_basename = r'(.+)_diff_v(\d+)v(\d+)\.tex'
        basename = os.path.basename(latex_diff_path)
        basename_match = re.match(rgx_basename,  basename)
        assert basename_match
        doc_id = basename_match.group(1)
        rev_no = basename_match.group(2)
        revision_depth = rev_no

        # extract abstract
        rgx_abstract = r'\\begin\{abstract\}(.+)\\end\{abstract\}'
        abstract_match = re.search(rgx_abstract, raw_text, re.DOTALL)
        if not abstract_match:
            print(f'ERROR: abstract not extracted for {latex_diff_path}')
            return
        abstract = abstract_match.group(1)

        # drop paragraphs where there is no latexdiff commands
        abstract = self.drop_irrelevant_parts(abstract)

        # strip latex commands like \emph{..} etc.
        abstract = self.strip_latex_command(abstract)

        # standardize latexdiff commands
        abstract = self.standardize_latexdiff_commands(abstract)

        # remove space between punctuations when it is followed by latexdiff commands
        abstract = self.adjust_punctuations(abstract)

        before_revision, after_revision, edit_actions = self.parse_abstract(abstract)
        if not before_revision:
            print('ERROR: sanity checks are not passed')
            return

        # add sentence segmentation info.
        before_sents = splitter.segment(before_revision)
        sents_positions = []
        cuml_char_index = 0
        for sent in before_sents:
            sents_positions.append(cuml_char_index)
            for _ in range(len(sent)):
                cuml_char_index += 1

        parsed_sample = {
            'doc_id': doc_id,
            'revision_depth': revision_depth,
            'before_revision': before_revision,
            'after_revision': after_revision,
            'edit_actions': edit_actions,
            'sents_char_pos': sents_positions
        }
        return parsed_sample

    def drop_irrelevant_parts(self, text: str) -> str:
        """Drop paragraphs that do not contain latexdiff commands"""
        filtered_paragraphs = []
        delbegin_count = addbegin_count = 0
        paragraphs = text.split('\n\n')
        excludes = set()
        for i, para in enumerate(paragraphs):
            if all(command not in para for command in ['\DIFdelbegin', '\DIFaddbegin', '\DIFdelend', '\DIFaddend']):
                # need to check the opened commands as a command may span across paragraphs
                if delbegin_count <= 0 and addbegin_count <= 0:
                    excludes.add(i)
            else:
                delbegin_count += para.count('\DIFdelbegin')
                addbegin_count += para.count('\DIFaddbegin')
                delbegin_count -= para.count('\DIFdelend')
                addbegin_count -= para.count('\DIFaddend')

        filtered_paragraphs = [para for i, para in enumerate(paragraphs) if i not in excludes]
        return '\n\n'.join(filtered_paragraphs)

    def strip_latex_command(self, text: str) -> str:
        """Strip latex commands like `\emph{content}`"""
        def _strip(matched: re.Match) -> str:
            excludes = set(['DIFdel', 'DIFadd'])
            command_name = matched.group(1)
            content = matched.group(2)
            if command_name in excludes:
                return matched.group(0)
            return content

        rgx_command = r'\\([\w\d]+)\{(.*?)\}'
        stripped_text = re.sub(rgx_command, _strip, text, re.DOTALL)
        return stripped_text

    def standardize_latexdiff_commands(self, text: str) -> str:
        """Remove unwanted giberish that is sometimes present in between latexdiff commands like, `%DIFDELCMD <%DIFDELCMD < %%%`
        and strip the outer latexdiff commands like DIFdel(begin|end) and DIFadd(begin|end)"""
        rgx_delbegin = r'\\DIFdelbegin\s*%DIFDELCMD < } %%%\s*\\DIFdelend'
        rgx_addbegin = r'\\DIFaddbegin\s*%DIFDELCMD < } %%%\s*\\DIFaddend'
        text = re.sub(rgx_delbegin, r'', text, re.DOTALL)
        text = re.sub(rgx_addbegin, r'', text, re.DOTALL)

        rgx_delbegin = r'(\\DIFdelbegin)[^\\]+?(?=\\DIFdel)'
        rgx_addbegin = r'(\\DIFaddbegin)[^\\]+?(?=\\DIFadd)'
        text = re.sub(rgx_delbegin, r'\1 ', text, re.DOTALL)
        text = re.sub(rgx_addbegin, r'\1 ', text, re.DOTALL)

        rgx_delend = r'(\\DIFdel\{(?:\\}|[^\}])*\})([^\\]+)(\\DIFdelend)'
        rgx_addend = r'(\\DIFadd\{(?:\\}|[^\}])*\})([^\\]+)(\\DIFaddend)'
        text = re.sub(rgx_delend, r'\1 \3', text, re.DOTALL)
        text = re.sub(rgx_addend, r'\1 \3', text, re.DOTALL)

        rgx_delend = r'(\\DIFdel\{(?:\\}|[^\}])*\})([^\\]+)(\\DIFdel)'
        rgx_addend = r'(\\DIFadd\{(?:\\}|[^\}])*\})([^\\]+)(\\DIFadd)'
        text = re.sub(rgx_delend, r'\1 \3', text, re.DOTALL)
        text = re.sub(rgx_addend, r'\1 \3', text, re.DOTALL)

        # Strip outer latexdiff commands
        text = text.replace('\DIFdelbegin', '')
        text = text.replace('\DIFdelend', '')
        text = text.replace('\DIFaddbegin', '')
        text = text.replace('\DIFaddend', '')
        return text

    def adjust_punctuations(self, text: str) -> str:
        """Remove any space before a punctuation that is present alone"""

        punct_str = f"[{''.join(PUNCTUATIONS)}]"
        rgx_punct = rf'(\\DIF(del|add)end) +(' + punct_str + r')'
        return re.sub(rgx_punct, r'\1\3', text)

    def parse_abstract(self, abstract: str) -> Tuple[str, str, Dict]:
        """Parse a cleaned raw abstract to produce `before_revision`, `after_revision`, and `edit_actions`"""
        def _parse_abstract(matched: re.Match, action_type: str) -> str:
            """Process a matched regex by stripping the latexdiff commands, anchoring the contents, and defining an `edit_action`"""
            anchor = f'{self.anchor_symbol}{action_type}{self.anchor_symbol}'
            start_position = matched.start()

            if action_type == 'R':
                before = ' '.join(matched.group(1).strip().split())
                after = ' '.join(matched.group(2).strip().split())
            elif action_type == 'D':
                before = ' '.join(matched.group(1).strip().split())
                after = None
            elif action_type == 'A':
                before = None
                after = ' '.join(matched.group(1).strip().split())
            else:
                print(f'ERROR: unknown action_type, {action_type}')
                sys.exit()

            action = {
                'type': action_type,
                'before': before,
                'after': after
            }
            if action_type == 'R':
                replaced_string = f'{anchor}{before}'
                replace_actions.append((start_position, action))
            elif action_type == 'D':
                replaced_string = f'{anchor}{before}'
                delete_actions.append((start_position, action))
            else:
                replaced_string = f'{anchor}'
                add_actions.append((start_position, action))
            return replaced_string

        # replace action is when one delbegin is immediately followed by one addbegin
        rgx_replace = r'\\DIFdel\{((?:\\}|[^\}])*)\} *\\DIFadd\{((?:\\}|[^\}])*)\}'
        rgx_delete = r'\\DIFdel\{((?:\\}|[^\}])*)\}'
        rgx_add = r'\\DIFadd\{((?:\\}|[^\}])*)\}'


        # need to store and order the actions separately,
        # otherwise, parsing previous (long) actions may distort the relative order
        replace_actions = []
        delete_actions = []
        add_actions = []

        abstract = re.sub(rgx_replace, lambda m: _parse_abstract(m, 'R'), abstract)
        # print(abstract)
        # input('after R')
        abstract = re.sub(rgx_delete, lambda m: _parse_abstract(m, 'D'), abstract)
        # print('after D')
        # input(abstract)
        abstract = re.sub(rgx_add, lambda m: _parse_abstract(m, 'A'), abstract)
        # print('after A')
        # input(abstract)
        replace_actions = sorted(replace_actions)
        delete_actions = sorted(delete_actions)
        add_actions = sorted(add_actions)

        # normalize whitespaces
        abstract = ' '.join(abstract.split())
        # print(f'\n{abstract}')

        cursor = 0
        action_type = None
        before_revision = ''
        edit_actions = []
        while cursor < len(abstract):
            char = abstract[cursor]
            if char == self.anchor_symbol:
                assert abstract[cursor + 2] == self.anchor_symbol
                action_type = abstract[cursor + 1]
                assert action_type in set(['R', 'D', 'A'])

                if action_type in ['R', 'D']:
                    if action_type == 'R':
                        _, action = replace_actions.pop(0)
                    else:
                        _, action = delete_actions.pop(0)
                    before_part = action['before']
                    # assert before_part == abstract[cursor + 3:len(before_part)]
                    if before_part != abstract[cursor + 3:cursor + 3 + len(before_part)]:
                        print(f'ERROR: "{before_part}" != "{abstract[cursor + 3:cursor + 3 + len(before_part)]}"')
                    action['start_char_pos'] = len(before_revision)
                    before_revision = before_revision + before_part
                    action['end_char_pos'] = len(before_revision)
                    cursor = cursor + 3 + len(before_part)
                else:  # 'A'
                    _, action = add_actions.pop(0)
                    action['start_char_pos'] = len(before_revision)
                    action['end_char_pos'] = action['start_char_pos']
                    cursor = cursor + 3
                edit_actions.append(action)
            else:
                before_revision = before_revision + char
                cursor = cursor + 1

        if not self.sanity_check_before_revision(before_revision, edit_actions):
            print('ERROR: before_revision sanity check not passed!')
            return False, False, False

        # print(f'\n{before_revision}\n')

        after_revision = self.get_after_revision(before_revision, edit_actions)

        if not self.sanity_check_after_revision(after_revision):
            print('ERROR: after_revision sanity check not passed!')
            return False, False, False

        # print(f'\n{after_revision}\n')
        # print(edit_actions)

        return before_revision, after_revision, edit_actions

    def _sanity_check_remaining_latexdiff_command(self, text: str) -> bool:
        if 'DIFadd' in text or 'DIFdel' in text:
            print(f'ERROR: DIFadd or DIFdel in text:\n{text}')
            return False
        return True

    def _sanity_check_remaining_anchor(self, text: str) -> bool:
        if self.anchor_symbol in text:
            print(f'ERROR: anchor symbol {self.anchor_symbol} in text:\n{text}')
            return False
        return True

    def sanity_check_before_revision(self, text: str, edit_actions: List[Dict]) -> bool:
        if not self._sanity_check_remaining_latexdiff_command(text):
            return False
        if not self._sanity_check_remaining_anchor(text):
            return False
        is_passed = True
        for action in edit_actions:
            if action['before'] is not None:
                char_level_before = text[action['start_char_pos']:action['end_char_pos']]
                if char_level_before != action['before']:
                    print(f'ERROR: char-level `before` not matched!\n"{char_level_before}" != "{action["before"]}"\n{action}')
                    is_passed = False
        return is_passed

    def sanity_check_after_revision(self, text: str) -> bool:
        if not self._sanity_check_remaining_latexdiff_command(text):
            return False
        if not self._sanity_check_remaining_anchor(text):
            return False
        return True

    def get_after_revision(self, before_revision: str, edit_actions: List[Dict]) -> str:
        """Construct after_revision from before_revision and edit_actions"""
        chunks = []
        edit_actions = sorted(edit_actions, key=lambda d: d['start_char_pos'])
        index = 0
        for action in edit_actions:
            chunk = before_revision[index:action['start_char_pos']]
            chunks.append(chunk)
            index = action['end_char_pos']
            if action['type'] == 'R':
                chunks.append(action['after'])
            elif action['type'] == 'D':
                pass
            else:
                chunks.append(action['after'])
        # add remaining chunk if any
        if index < len(before_revision) - 1:
            chunks.append(before_revision[index:])
        after_revision = ''.join(chunks)
        after_revision = re.sub(r' {2,}', ' ', ''.join(chunks))
        return after_revision


if __name__ == '__main__':
    def process_dir(base_dir: str) -> None:
        latexdiff_dir_names = ['arxiv', 'news', 'wiki']
        len_samples = {}
        for dir_name in latexdiff_dir_names:
            dir_path = os.path.join(base_dir, 'temp_'+dir_name)
            assert os.path.isdir(dir_path)
            output_json_path = os.path.join(base_dir, f'doc-level_{dir_name}.json')
            parser = LatexdiffParser(dir_path, output_json_path)
            len_parsed_samples = parser.parse_and_save()
            len_samples[dir_name] = len_parsed_samples

        from pprint import pprint
        pprint(len_samples)

    my_config = deepcopy(en_config)
    my_config['mincut'] = 10
    splitter = SentSplit('en', **my_config)

    # process 4K
    base_dir_4k = '../data/31K_revisions'
    process_dir(base_dir_4k)
