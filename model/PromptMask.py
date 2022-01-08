
import torch
import torch.nn as nn
from pytorch_transformers import RobertaForMaskedLM
from pytorch_transformers.modeling_bert import BertLayerNorm
from config.cfg import cfg, path, hyper_roberta
from torch.autograd import Variable
import os

if cfg['device'] != 'TPU':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu_id'])
    device = torch.device(cfg['device'])
else:
    # for TPU
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()

class PromptMask(nn.Module):
    def __init__(self):
        super(PromptMask, self).__init__()
        self.roberta = RobertaForMaskedLM.from_pretrained(path['roberta_path'])

    def forward(self, input_x):
        mask0 = (input_x == 50264)
        mask1 = (input_x != 1).type(torch.long)

        input_x = self.roberta(input_x, attention_mask=mask1)
        x = input_x[0]
        x = x[mask0]

        return x






