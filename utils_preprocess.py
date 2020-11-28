import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)


def print_squad_sample(train_data, line_length=14, separator_length=120):
  sample = train_data.sample(frac=1).head(1)
  context = sample.doc_tokens.values
  print('='*separator_length)
  print('CONTEXT: ')
  print('='*separator_length)
  lines = [' '.join(context[0][idx:idx+line_length]) for idx in range(0, len(context[0]), line_length)]
  for l in lines:
      print(l)
  print('='*separator_length)
  questions = train_data[train_data.doc_tokens.values==context]
  print('QUESTION:', ' '*(3*separator_length//4), 'ANSWER:')
  for idx, row in questions.iterrows():
    question = row.question_text
    answer = row.orig_answer_text
    print(question, ' '*(3*separator_length//4-len(question)+9), (answer if answer else 'No awnser found'))


def create_features(cached_features_file, examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    if not os.path.exists(cached_features_file):
      features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            is_training=is_training)
      torch.save(features, cached_features_file)
    else:
      features = torch.load(cached_features_file)
    return features


def generate_bert_loader_train(features, batch_size, drop_last):
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
  all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

  all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
  all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
  dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                          all_start_positions, all_end_positions,
                          all_cls_index, all_p_mask)
  train_sampler = RandomSampler(dataset)
  train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)
  return train_dataloader, len(dataset)


def generate_bert_loader_validation(features, batch_size, drop_last):
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
  all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

  all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
  dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                          all_example_index, all_cls_index, all_p_mask)
  validation_sampler = SequentialSampler(dataset)
  validation_dataloader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch_size, drop_last=True)
  return validation_dataloader, len(dataset)