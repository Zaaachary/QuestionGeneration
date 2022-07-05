"""
ref
https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/text-transformers.ipynb#scrollTo=ddfafe98
https://pytorch-lightning.readthedocs.io/en/latest/
"""
import logging

import torch
from pytorch_lightning import LightningModule

from transformers import (
    AdamW, get_linear_schedule_with_warmup,
    GPT2Config, GPT2Tokenizer,
    BartConfig, BartForConditionalGeneration, BartTokenizer
)
from transformers import BertTokenizer  # CPT

from models.GPT2_Model import GPT2LMHeadModel
from models.modeling_cpt import CPTForConditionalGeneration # CPT


# logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
    'cpt': (BartConfig, CPTForConditionalGeneration, BertTokenizer)
}

# class ProtoQA_GPT2_Baseline(LightningModule):
class Generation_Model(LightningModule):
    def __init__(
        self,
        PTM_name_or_path: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_proportion: int = 0,
        weight_decay: float = 0.0,
        train_batch_size_pre_device: int = 32,
        model_type = 'cpt',
        args_str: str = '',
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()
        
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type.lower()]

        self.model = model_class.from_pretrained(PTM_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(self.hparams.PTM_name_or_path)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, **inputs):
        # inference & prediction
        return self.model(**inputs)

    def generate(self, **inputs):
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.generation_tf_utils.TFGenerationMixin.generate
        return self.model.generate(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)

        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        self.log("val_loss", val_loss)
        return {'loss': val_loss}

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        perplexity = torch.exp(torch.tensor(loss))
        self.log("val_loss", loss, prog_bar=True)
        self.log("perplexity", perplexity, prog_bar=True)
        logging.info(f"at step {self.global_step} - val_loss:{round(loss.item(), 5)}; perplexity:{round(perplexity.item(), 5)}")
        return loss

    def set_example_num(self, example_num):
        self.example_num = example_num

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=configure_optimizers#configure-optimizers"""
        # Calculate total steps
        if isinstance(self.trainer.gpus, list):
            # specific device like [6], [6,7]
            gpus = len(self.trainer.gpus)
        else:
            gpus = self.trainer.gpus
        
        tb_size = self.hparams.train_batch_size_pre_device * max(1, gpus)
        ab_size = self.trainer.accumulate_grad_batches

        # (len_dataset // batch_size) * epoch if len_dataset % batch_size = 0 else (len_dataset // batch_size + 1) * epoch # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch

        total_steps = int((self.example_num * int(self.trainer.max_epochs) // tb_size) // ab_size )
        warmup_steps = int(self.hparams.warmup_proportion * total_steps)
        logging.info(f'example_num: {self.example_num}; max_epochs: {self.trainer.max_epochs}')
        logging.info(f'total_steps: {total_steps}; warmup_steps: {warmup_steps}')

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


