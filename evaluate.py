import torch
from tqdm import tqdm, trange
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad, plot_pr_curve
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(model, validation_dataloader, features_val, examples_val, tokenizer, predict_file, len_val, batch_size, device):
  print("***** Running evaluation *****")
  print("  Num examples = %d" % len_val)
  print("  Batch size = %d" % batch_size)
  all_results = []
  for batch in tqdm(validation_dataloader, desc="Evaluating", miniters=100, mininterval=5.0):
    model.eval()
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
      inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
                }
      example_indices = batch[3]
      outputs = model(**inputs)

    for i, example_index in enumerate(example_indices):
      eval_feature = features_val[example_index.item()]
      unique_id = int(eval_feature.unique_id)

      result = RawResult(unique_id    = unique_id,
                         start_logits = to_list(outputs[0][i]),
                         end_logits   = to_list(outputs[1][i]))
      all_results.append(result)

  # Compute predictions
  output_prediction_file = "/predictions/predictions.json"
  output_nbest_file = "/predictions/nbest_predictions.json"
  output_null_log_odds_file = "/predictions/null_odds.json"
  output_dir = "/predictions/predict_results"

  write_predictions(examples_val, features_val, all_results, 10,
                  30, True, output_prediction_file,
                  output_nbest_file, output_null_log_odds_file, False,
                  True, 0.0)

  # Evaluate with the official SQuAD script
  evaluate_options = EVAL_OPTS(data_file=predict_file,
                               pred_file=output_prediction_file,
                               na_prob_file=output_null_log_odds_file,
                               out_image_dir=None)
  results = evaluate_on_squad(evaluate_options)
  return results