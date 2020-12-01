import torch
from pytorch_transformers import BertForQuestionAnswering


class bert_mod(object):
  def __init__(self, base='bert-base-uncased'):
    self.model = BertForQuestionAnswering.from_pretrained(base)

  def freeze_param(self):
    for param in self.model.bert.parameters():
      param.requires_grad = False

  def bert_linear(self):
    return self.model
    
  def bert_linear_linear(self):
    self.model.qa_outputs=torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=768, bias=True),
                                         torch.nn.Linear(in_features=768, out_features=2, bias=True))
    return self.model

  def bert_linear_act_drop(self):
    self.model.qa_outputs=torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=768, bias=True),
                                         torch.nn.ReLU(inplace=False),
                                         torch.nn.Dropout(0.2),
                                         torch.nn.Linear(in_features=768, out_features=2, bias=True))
    return self.model

  def bert_linear_more_hidden(self):
    self.model.qa_outputs=torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=1536, bias=True),
                                         torch.nn.Linear(in_features=1536, out_features=2, bias=True))
    return self.model

  def bert_linear_linear_linear(self):
    self.model.qa_outputs=torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=768, bias=True),
                                         torch.nn.Linear(in_features=768, out_features=768, bias=True),
                                         torch.nn.Linear(in_features=768, out_features=2, bias=True))
    return self.model

  def bert_linear_full(self):
    self.model.qa_outputs=torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=1536, bias=True),
                                         torch.nn.ReLU(inplace=False),
                                         torch.nn.Dropout(0.2),
                                         torch.nn.Linear(in_features=1536, out_features=768, bias=True),
                                         torch.nn.ReLU(inplace=False),
                                         torch.nn.Dropout(0.2),
                                         torch.nn.Linear(in_features=768, out_features=2, bias=True))
    return self.model

  def bert_linear_no_act(self):
    self.model.qa_outputs=torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=1536, bias=True),
                                         torch.nn.Dropout(0.2),
                                         torch.nn.Linear(in_features=1536, out_features=768, bias=True),
                                         torch.nn.Dropout(0.2),
                                         torch.nn.Linear(in_features=768, out_features=2, bias=True))
    return self.model

  def bert_conv_linear(self):
    self.model.bert.add_module('CNN',torch.nn.Conv2d(in_channels=32*256,out_channels=768,kernel_size=3,stride=1))
    return self.model
