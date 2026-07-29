#    Copyright 2023 Haotian Liu & Qinghao Ye (Modified from LLaVA)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mplug_owl2 import (
    MPLUGOwl2Config,
    MplugOwlVisionConfig,
    MplugOwlVisualAbstractorConfig,
)
from .constants import IMAGE_TOKEN_INDEX
from .modeling_llama2 import LlamaModel, LlamaPreTrainedModel
from .visual_encoder import MplugOwlVisionModel, MplugOwlVisualAbstractorModel


class MPLUGOwl2MetaModel:
    def __init__(self, config):
        super().__init__(config)
        self.vision_model = MplugOwlVisionModel(
            MplugOwlVisionConfig(**config.visual_config["visual_model"])
        )
        self.visual_abstractor = MplugOwlVisualAbstractorModel(
            MplugOwlVisualAbstractorConfig(
                **config.visual_config["visual_abstractor"]
            ),
            config.hidden_size,
        )

    def get_vision_tower(self):
        return self.vision_model

    def get_visual_abstractor(self):
        return self.visual_abstractor


class MPLUGOwl2MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        image_features = self.get_model().vision_model(images).last_hidden_state
        return self.get_model().visual_abstractor(
            encoder_hidden_states=image_features
        ).last_hidden_state

    def prepare_inputs_for_multimodal(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        images,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required when images are provided")

        if isinstance(images, list) or images.ndim == 5:
            concat_images = torch.cat(images, dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [features.flatten(0, 1) for features in image_features]
        else:
            image_features = self.encode_images(images)

        input_embeddings = []
        modality_indicators = []
        image_index = 0

        for current_input_ids in input_ids:
            image_token_indices = torch.where(
                current_input_ids == IMAGE_TOKEN_INDEX
            )[0]
            if image_token_indices.numel() == 0:
                current_embeddings = self.get_model().embed_tokens(
                    current_input_ids
                )
                input_embeddings.append(current_embeddings)
                modality_indicators.append(
                    torch.zeros(
                        current_embeddings.shape[0],
                        dtype=torch.long,
                        device=current_embeddings.device,
                    )
                )
                image_index += 1
                continue

            current_embeddings = []
            current_modalities = []
            while image_token_indices.numel() > 0:
                current_image_features = image_features[image_index]
                image_token_start = image_token_indices[0]
                text_embeddings = self.get_model().embed_tokens(
                    current_input_ids[:image_token_start]
                )
                current_embeddings.extend(
                    [text_embeddings, current_image_features]
                )
                current_modalities.extend(
                    [
                        torch.zeros(
                            text_embeddings.shape[0],
                            dtype=torch.long,
                            device=text_embeddings.device,
                        ),
                        torch.ones(
                            current_image_features.shape[0],
                            dtype=torch.long,
                            device=current_image_features.device,
                        ),
                    ]
                )
                image_index += 1
                current_input_ids = current_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(
                    current_input_ids == IMAGE_TOKEN_INDEX
                )[0]

            if current_input_ids.numel() > 0:
                text_embeddings = self.get_model().embed_tokens(
                    current_input_ids
                )
                current_embeddings.append(text_embeddings)
                current_modalities.append(
                    torch.zeros(
                        text_embeddings.shape[0],
                        dtype=torch.long,
                        device=text_embeddings.device,
                    )
                )

            input_embeddings.append(torch.cat(current_embeddings, dim=0))
            modality_indicators.append(torch.cat(current_modalities, dim=0))

        sequence_lengths = [
            embeddings.shape[0] for embeddings in input_embeddings
        ]
        max_length = max(sequence_lengths)
        padded_embeddings = []
        padded_modalities = []
        for embeddings, modalities in zip(
            input_embeddings,
            modality_indicators,
        ):
            padding_length = max_length - embeddings.shape[0]
            padded_embeddings.append(
                torch.cat(
                    [
                        embeddings,
                        torch.zeros(
                            padding_length,
                            embeddings.shape[1],
                            dtype=embeddings.dtype,
                            device=embeddings.device,
                        ),
                    ],
                    dim=0,
                )
            )
            padded_modalities.append(
                torch.cat(
                    [
                        modalities,
                        torch.zeros(
                            padding_length,
                            dtype=modalities.dtype,
                            device=modalities.device,
                        ),
                    ],
                    dim=0,
                )
            )

        input_embeddings = torch.stack(padded_embeddings, dim=0)
        modality_indicators = torch.stack(padded_modalities, dim=0)

        if attention_mask is not None:
            expanded_attention_masks = []
            for current_attention_mask, sequence_length in zip(
                attention_mask,
                sequence_lengths,
            ):
                inserted_tokens = sequence_length - current_attention_mask.shape[0]
                expanded_attention_masks.append(
                    torch.cat(
                        [
                            torch.ones(
                                inserted_tokens,
                                dtype=current_attention_mask.dtype,
                                device=current_attention_mask.device,
                            ),
                            current_attention_mask,
                            torch.zeros(
                                max_length - sequence_length,
                                dtype=current_attention_mask.dtype,
                                device=current_attention_mask.device,
                            ),
                        ],
                        dim=0,
                    )
                )
            attention_mask = torch.stack(expanded_attention_masks, dim=0)

        return (
            None,
            modality_indicators,
            attention_mask,
            past_key_values,
            input_embeddings,
        )


class MPLUGOwl2LlamaModel(MPLUGOwl2MetaModel, LlamaModel):
    config_class = MPLUGOwl2Config


class MPLUGOwl2LlamaForCausalLM(
    LlamaPreTrainedModel,
    MPLUGOwl2MetaForCausalLM,
):
    config_class = MPLUGOwl2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = MPLUGOwl2LlamaModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.post_init()

    def get_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        modality_indicators = None
        if images is not None:
            (
                input_ids,
                modality_indicators,
                attention_mask,
                past_key_values,
                inputs_embeds,
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                images,
            )

        outputs = self.model(
            input_ids=input_ids,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.lm_head(outputs[0])

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoConfig.register("mplug_owl2", MPLUGOwl2Config, exist_ok=True)
AutoModelForCausalLM.register(
    MPLUGOwl2Config,
    MPLUGOwl2LlamaForCausalLM,
    exist_ok=True,
)
