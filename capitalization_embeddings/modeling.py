"""BERT variants with factorized capitalization embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
    TokenClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertOnlyMLMHead,
    BertPooler,
    BertPreTrainedModel,
)


class CapitalizedBertConfig(BertConfig):
    """BERT config extended with a tiny capitalization feature vocabulary."""

    model_type = "capitalized-bert"

    def __init__(
        self,
        capitalization_vocab_size: int = 3,
        capitalization_pad_token_id: int = 0,
        capitalization_loss_weight: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.capitalization_vocab_size = capitalization_vocab_size
        self.capitalization_pad_token_id = capitalization_pad_token_id
        self.capitalization_loss_weight = capitalization_loss_weight


class CapitalizedBertEmbeddings(nn.Module):
    """BERT embeddings plus a learned capitalization feature embedding."""

    def __init__(self, config: CapitalizedBertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.capitalization_embeddings = nn.Embedding(
            config.capitalization_vocab_size,
            config.hidden_size,
            padding_idx=config.capitalization_pad_token_id,
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        capitalization_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = self.position_ids[
                :,
                past_key_values_length : seq_length + past_key_values_length,
            ]

        if token_type_ids is None:
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            token_type_ids = buffered_token_type_ids.expand(input_shape[0], seq_length)

        if capitalization_ids is None:
            capitalization_ids = torch.zeros(
                input_shape,
                dtype=torch.long,
                device=device,
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        embeddings = embeddings + self.capitalization_embeddings(capitalization_ids)

        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CapitalizedBertModel(BertPreTrainedModel):
    """BERT encoder that accepts `capitalization_ids` next to `input_ids`."""

    config_class = CapitalizedBertConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["BertEmbeddings", "BertLayer", "CapitalizedBertEmbeddings"]

    def __init__(
        self,
        config: CapitalizedBertConfig,
        add_pooling_layer: bool = True,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = CapitalizedBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.attn_implementation = getattr(config, "_attn_implementation", "eager")
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        capitalization_ids: torch.LongTensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions | tuple[torch.Tensor, ...]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if input_ids is not None:
            if hasattr(self, "warn_if_padding_and_no_attention_mask"):
                self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Specify either input_ids or inputs_embeds.")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                device=device,
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            input_shape,
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask,
            )
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            capitalization_ids=capitalization_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@dataclass
class CapitalizedMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    capitalization_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class CapitalizedBertForMaskedLM(BertPreTrainedModel):
    """Masked LM with an auxiliary capitalization prediction head."""

    config_class = CapitalizedBertConfig
    _tied_weights_keys = {
        "cls.predictions.decoder.bias": "cls.predictions.bias",
        "cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
    }

    def __init__(self, config: CapitalizedBertConfig) -> None:
        super().__init__(config)
        self.bert = CapitalizedBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.capitalization_classifier = nn.Linear(
            config.hidden_size,
            config.capitalization_vocab_size,
        )
        self.post_init()

    def get_output_embeddings(self) -> nn.Module:
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.cls.predictions.decoder = new_embeddings

    @classmethod
    def from_uncased_pretrained(
        cls,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        **kwargs: Any,
    ) -> "CapitalizedBertForMaskedLM":
        config_kwargs = kwargs.pop("config_kwargs", {})
        config = kwargs.pop(
            "config",
            CapitalizedBertConfig.from_pretrained(
                pretrained_model_name_or_path,
                **config_kwargs,
            ),
        )
        return cls.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        capitalization_ids: torch.LongTensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        capitalization_labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CapitalizedMaskedLMOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            capitalization_ids=capitalization_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        capitalization_scores = self.capitalization_classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            token_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            loss = token_loss

        if capitalization_labels is not None:
            cap_loss_fct = CrossEntropyLoss(ignore_index=-100)
            cap_loss = cap_loss_fct(
                capitalization_scores.view(-1, self.config.capitalization_vocab_size),
                capitalization_labels.view(-1),
            )
            loss = (
                cap_loss * self.config.capitalization_loss_weight
                if loss is None
                else loss + cap_loss * self.config.capitalization_loss_weight
            )

        if not return_dict:
            output = (prediction_scores, capitalization_scores) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CapitalizedMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            capitalization_logits=capitalization_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CapitalizedBertForTokenClassification(BertPreTrainedModel):
    """Token classifier for downstream tasks such as CoNLL-2003 NER."""

    config_class = CapitalizedBertConfig

    def __init__(self, config: CapitalizedBertConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = CapitalizedBertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @classmethod
    def from_uncased_pretrained(
        cls,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        **kwargs: Any,
    ) -> "CapitalizedBertForTokenClassification":
        config_kwargs = kwargs.pop("config_kwargs", {})
        config = kwargs.pop(
            "config",
            CapitalizedBertConfig.from_pretrained(
                pretrained_model_name_or_path,
                **config_kwargs,
            ),
        )
        return cls.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        capitalization_ids: torch.LongTensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> TokenClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            capitalization_ids=capitalization_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
