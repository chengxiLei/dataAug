from datasets import load_from_disk,load_metric,load_dataset
reloaded_dataset = load_from_disk("./new_data/1")
shuffled = reloaded_dataset
# shuffled = reloaded_dataset.shuffle(seed=1)


import re
#tokenizer
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch
timit = load_dataset("timit_asr", data_dir='./');
timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
timit = timit.map(remove_special_characters)



# from transformers import Wav2Vec2CTCTokenizer

# tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")



# from transformers import Wav2Vec2FeatureExtractor

# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

# from transformers import Wav2Vec2Processor

# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)



# def prepare_dataset(batch):
#     audio = batch["audio"]

#     # batched output is "un-batched" to ensure mapping is correct
#     batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
#     with processor.as_target_processor():
#         batch["labels"] = processor(batch["text"]).input_ids
#     return batch

# timit_prepared = shuffled.select([11,12,13]).map(prepare_dataset, num_proc=1)
# # timit_prepared = timit_prepared.train_test_split(train_size=0.1)

# tst_prepared = timit["train"].select([11,12,13]).map(prepare_dataset, num_proc=1)

# print("############ generated")
# print(timit_prepared.features)
# print(shuffled[100]["text"])
# print("############ original")
# print(tst_prepared.features)
# print(timit["train"][100]["text"])

import IPython.display as ipd
import numpy as np


# import torch

# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional, Union

# @dataclass
# class DataCollatorCTCWithPadding:
#     """
#     Data collator that will dynamically pad the inputs received.
#     Args:
#         processor (:class:`~transformers.Wav2Vec2Processor`)
#             The processor used for proccessing the data.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:
#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence if provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
#         max_length_labels (:obj:`int`, `optional`):
#             Maximum length of the ``labels`` returned list and optionally padding length (see above).
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.
#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#     """

#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True
#     max_length: Optional[int] = None
#     max_length_labels: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     pad_to_multiple_of_labels: Optional[int] = None

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lengths and need
#         # different padding methods
#         input_features = [{"input_values": feature["input_values"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=self.padding,
#                 max_length=self.max_length_labels,
#                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
#                 return_tensors="pt",
#             )

#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels

#         return batch

# data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# wer_metric = load_metric("wer")
# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)

#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#     pred_str = processor.batch_decode(pred_ids)
#     # we do not want to group tokens when computing the metrics
#     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#     wer = wer_metric.compute(predictions=pred_str, references=label_str)

#     return {"wer": wer}


# from transformers import Wav2Vec2ForCTC

# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-base", 
#     ctc_loss_reduction="mean", 
#     pad_token_id=processor.tokenizer.pad_token_id,
# )
# model.freeze_feature_extractor()

# from transformers import TrainingArguments

# training_args = TrainingArguments(
#   output_dir="./wav2vec",
#   group_by_length=True,
#   per_device_train_batch_size=32,
#   evaluation_strategy="steps",
#   num_train_epochs=30,
#   fp16=True,
#   gradient_checkpointing=True, 
#   save_steps=500,
#   eval_steps=500,
#   logging_steps=500,
#   learning_rate=1e-4,
#   weight_decay=0.005,
#   warmup_steps=1000,
#   save_total_limit=2,
# )

# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     data_collator=data_collator,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=timit_prepared["train"],
#     eval_dataset=timit_prepared["test"],
#     tokenizer=processor.feature_extractor,
# )



print("all good")
lengtharr = len(np.asarray(shuffled[100]["audio"]["array"]))
inverse_audio = np.asarray(shuffled[100]["audio"]["array"])[::-1]

print(inverse_audio[1])
print(np.asarray(shuffled[100]["audio"]["array"])[lengtharr - 2])
# ipd.Audio(data=np.asarray(shuffled[20]["audio"]["array"]), autoplay=True, rate=16000)
ipd.Audio(data=np.asarray(timit["train"][20]["audio"]["array"]), autoplay=True, rate=16000)

