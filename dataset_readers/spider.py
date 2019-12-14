import json
import logging
import os
import functools
import time
from typing import List, Dict
from multiprocessing import Pool, TimeoutError
from pathlib import Path

import re
import dill
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Field, Instance
from allennlp.data.dataset_readers import MultiprocessDatasetReader
from allennlp.data.fields import TextField, ProductionRuleField, ListField, IndexField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from overrides import overrides
from spacy.symbols import ORTH, LEMMA

from dataset_readers.dataset_util.spider_utils import fix_number_value, disambiguate_items, sql_tokenize
from dataset_readers.fields.knowledge_graph_field import SpiderKnowledgeGraphField
from semparse.contexts.spider_db_context import SpiderDBContext
from semparse.worlds.spider_world import SpiderWorld

logger = logging.getLogger(__name__)


@DatasetReader.register("sparc")
class SpiderDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 keep_if_unparsable: bool = True,
                 tables_file: str = None,
                 dataset_path: str = 'dataset/database',
                 load_cache: bool = True,
                 save_cache: bool = True,
                 loading_limit = -1):
        super().__init__(lazy=lazy)

        # default spacy tokenizer splits the common token 'id' to ['i', 'd'], we here write a manual fix for that
        spacy_tokenizer = SpacyWordSplitter(pos_tags=True)
        spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
        self._tokenizer = WordTokenizer(spacy_tokenizer)

        self._utterance_token_indexers = question_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        self._load_cache = load_cache
        self._save_cache = save_cache
        self._loading_limit = loading_limit

    @overrides
    def _read(self, file_path: str):
        cache_dir = 'cache'
        if self._load_cache:
            logger.info(f'Trying to load cache from {cache_dir}')
        cache_dir = os.path.join('cache', file_path.split("/")[-1])
        if self._save_cache:
            os.makedirs(cache_dir, exist_ok=True)

        print(file_path)
        if not file_path.endswith('.json'):
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

        cnt = 0
        with open(file_path, "r") as data_file:
            json_obj = json.load(data_file)
            # This code segment is for debugging the preprocessing, otherwise we use multiprocessing to speed up.
            # for total_cnt, ex in enumerate(json_obj):
            #     if self._loading_limit == cnt:
            #         break
            #     ins = self.process_ex_wrap(total_cnt, ex, cache_dir) 
            #     if ins is not None:
            #         cnt += 1
            #         yield ins
            with Pool(processes=6) as pool:
                ret = list(pool.starmap(functools.partial(self.process_ex_wrap, cache_dir=cache_dir), enumerate(json_obj)))
            ret = [i for i in ret if i is not None]
            print("Size:", len(ret))
            return ret
    
    def process_ex_wrap(self, total_cnt, ex, cache_dir):
        cache_filename = f'instance-{total_cnt}.pt'
        cache_filepath = os.path.join(cache_dir, cache_filename)
        print(cache_filepath)
        ins = None
        if self._load_cache:
            try:
                ins = dill.load(open(cache_filepath, 'rb'))
                return ins
            except Exception as e:
                # could not load from cache - keep loading without cache
                pass
        try:
            ins = self.process_ex(total_cnt, ex, cache_filepath)
            if len(ins.fields['action_sequences'].field_list) == 0:
                ins = None
        except Exception as e:
            pass
        return ins

    def process_ex(self, total_cnt, ex, cache_filepath):
        utterances = []
        sql = []

        for step in ex['interaction']:
            utterances.append(step['utterance'])

            step['query_toks_no_value'] = sql_tokenize(re.sub(r"\'([^\']*)\'|\"([^\"]*)\"", r'value', step['query'].lower()))
            step['query_toks'] = sql_tokenize(step['query'].lower())
            query_tokens = []
            for tok in step['query_toks_no_value']:
                query_tokens += tok.split(' ')
            step['query_toks_no_value'] = query_tokens
            try:
                fix_number_value(step)
            except Exception:
                pass
            
            step['query_toks_no_value'] = disambiguate_items(ex['database_id'], step['query_toks_no_value'],
                self._tables_file, allow_aliases=False)

            sql.append(step['query_toks_no_value'])

        ins = self.text_to_instance(
                utterances=utterances,
                db_id=ex['database_id'],
                sql=sql)

        if self._save_cache:
            dill.dump(ins, open(cache_filepath, 'wb'))

        return ins

    def text_to_instance(self,
                         utterances: List[str],
                         db_id: str,
                         sql: List[List[str]] = None):
        fields: Dict[str, Field] = {}



        ctxts = [SpiderDBContext(db_id, utterance, tokenizer=self._tokenizer,
                                     tables_file=self._tables_file, dataset_path=self._dataset_path)
                                     for utterance in utterances ]


        super_utterance = ' '.join(utterances)
        hack_ctxt = SpiderDBContext(db_id, super_utterance, tokenizer=self._tokenizer,
                                     tables_file=self._tables_file, dataset_path=self._dataset_path)

        kg = SpiderKnowledgeGraphField(hack_ctxt.knowledge_graph,
                                        hack_ctxt.tokenized_utterance,
                                        self._utterance_token_indexers,
                                        entity_tokens=hack_ctxt.entity_tokens,
                                        include_in_vocab=False,  # TODO: self._use_table_for_vocab,
                                        max_table_tokens=None)
        '''
        kgs = [SpiderKnowledgeGraphField(db_context.knowledge_graph,
                                        db_context.tokenized_utterance,
                                        self._utterance_token_indexers,
                                        entity_tokens=db_context.entity_tokens,
                                        include_in_vocab=False,  # TODO: self._use_table_for_vocab,
                                        max_table_tokens=None)  # self._max_table_tokens)
                        for db_context in ctxts]
        '''
        worlds = []

        for i in range(len(sql)):
            sqli = sql[i]
            db_context = ctxts[i]
            world = SpiderWorld(db_context, query=sqli)
            worlds.append(world)

        fields["utterances"] = ListField([TextField(db_context.tokenized_utterance, self._utterance_token_indexers)
                for db_context in ctxts])

        #action_sequence, all_actions = world.get_action_sequence_and_all_actions()

        action_tups = [world.get_action_sequence_and_all_actions() for world in worlds]
        action_sequences = [tup[0] for tup in action_tups]
        all_actions = [tup[1] for tup in action_tups]

        for i in range(len(action_sequences)):
            action_sequence = action_sequences[i]

            if action_sequence is None and self._keep_if_unparsable:
                # print("Parse error")
                action_sequence = []
            elif action_sequence is None:
                return None

            action_sequences[i] = action_sequence

        all_valid_actions_fields = []
        all_action_sequence_fields = []

        for i in range(len(all_actions)):
            index_fields: List[Field] = []
            production_rule_fields: List[Field] = []

            all_actionsi = all_actions[i]
            for production_rule in all_actionsi:
                nonterminal, rhs = production_rule.split(' -> ')
                production_rule = ' '.join(production_rule.split(' '))
                field = ProductionRuleField(production_rule,
                                            world.is_global_rule(rhs),
                                            nonterminal=nonterminal)
                production_rule_fields.append(field)

            valid_actions_field = ListField(production_rule_fields)

            all_valid_actions_fields.append(valid_actions_field)

            action_map = {action.rule: i  # type: ignore
                        for i, action in enumerate(valid_actions_field.field_list)}

            index_fields: List[Field] = []

            action_sequence = action_sequences[i]

            for production_rule in action_sequence:
                index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
            if not action_sequence:
                index_fields = [IndexField(-1, valid_actions_field)]

            action_sequence_field = ListField(index_fields)

            all_action_sequence_fields.append(action_sequence_field)


        fields["valid_actions"] = ListField(all_valid_actions_fields)
        fields["action_sequences"] = ListField(all_action_sequence_fields)
        fields["worlds"] = ListField([MetadataField(world) for world in worlds])
        fields["schema"] = kg

        '''
        fields['utterances'] = ListField[TextField]
        fields['valid_actions'] = ListField[ListField[ProductionRuleField]]
        fields['action_sequences'] = ListField[ListField[IndexField]]
        fields['worlds'] = ListField[MetadataField[SpiderWorld]]
        fields['schemas'] = ListField[SpiderKnowledgeGraphField]
        '''

        return Instance(fields)
