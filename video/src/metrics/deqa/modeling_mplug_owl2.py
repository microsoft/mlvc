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


import warnings
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.utils\.generic"
)

from .visual_encoder import MplugOwlVisionModel, MplugOwlVisualAbstractorModel  # noqa: E402
from .utils import extend_list, find_prefix  # noqa: E402
from .modeling_llama2 import replace_llama_modality_adaptive  # noqa: E402
from .configuration_mplug_owl2 import (MPLUGOwl2Config, MplugOwlVisionConfig,  # noqa: E402
                                       MplugOwlVisualAbstractorConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: E402
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaForCausalLM,  # noqa: E402
                          LlamaModel)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<|image|>"


class MPLUGOwl2MetaModel:
    def __init__(self, config):
        super(MPLUGOwl2MetaModel, self).__init__(config)
        self.vision_model = MplugOwlVisionModel(
            MplugOwlVisionConfig(**config.visual_config["visual_model"])
        )
        self.visual_abstractor = MplugOwlVisualAbstractorModel(
            MplugOwlVisualAbstractorConfig(**config.visual_config["visual_abstractor"]),
            config.hidden_size,
        )

    def get_vision_tower(self):
        vision_model = getattr(self, "vision_model", None)
        if type(vision_model) is list:
            vision_model = vision_model[0]
        return vision_model

    def get_visual_abstractor(self):
        visual_abstractor = getattr(self, "visual_abstractor", None)
        if type(visual_abstractor) is list:
            visual_abstractor = visual_abstractor[0]
        return visual_abstractor


class MPLUGOwl2MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        image_features = self.get_model().vision_model(images).last_hidden_state
        image_features = (
            self.get_model()
            .visual_abstractor(encoder_hidden_states=image_features)
            .last_hidden_state
        )
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        if images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            multiway_indices = torch.zeros_like(input_ids).long().to(self.device)
            return (
                input_ids,
                multiway_indices,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_modality_indicators = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids[:half_len]
                )
                cur_input_embeds_2 = self.get_model().embed_tokens(
                    cur_input_ids[half_len:]
                )
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2],
                    dim=0,
                )
                new_input_embeds.append(cur_input_embeds)

                cur_modality_indicators = (
                    torch.zeros(len(cur_input_embeds)).long().to(self.device)
                )
                new_modality_indicators.append(cur_modality_indicators)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_modality_indicators = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                )
                cur_new_input_embeds.append(cur_image_features)

                # Add modality indicator
                assert image_token_start == len(cur_input_ids[:image_token_start])
                cur_modality_indicators.append(
                    torch.zeros(len(cur_input_ids[:image_token_start])).long()
                )
                cur_modality_indicators.append(
                    torch.ones(len(cur_image_features)).long()
                )

                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_labels = cur_labels[image_token_start + 1:]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids)
                )
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids)).long())
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

            # Modality
            cur_modality_indicators = [
                x.to(device=self.device) for x in cur_modality_indicators
            ]
            cur_modality_indicators = torch.cat(cur_modality_indicators, dim=0)
            new_modality_indicators.append(cur_modality_indicators)

            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            # Embedding
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # Modality
            new_modality_indicators_align = []
            for cur_modality_indicator in new_modality_indicators:
                cur_new_embed = torch.cat(
                    (
                        cur_modality_indicator,
                        torch.zeros(
                            max_len - cur_modality_indicator.shape[0],
                            dtype=cur_modality_indicator.dtype,
                            device=cur_modality_indicator.device,
                        ),
                    ),
                    dim=0,
                )
                new_modality_indicators_align.append(cur_new_embed)
            new_modality_indicators = torch.stack(new_modality_indicators_align, dim=0)

            # Label
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # Attention Mask
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_modality_indicators = torch.stack(new_modality_indicators, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return (
            None,
            new_modality_indicators,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )


class MPLUGOwl2LlamaModel(MPLUGOwl2MetaModel, LlamaModel):
    config_class = MPLUGOwl2Config

    def __init__(self, config: MPLUGOwl2Config):
        super(MPLUGOwl2LlamaModel, self).__init__(config)


class MPLUGOwl2LlamaForCausalLM(LlamaForCausalLM, MPLUGOwl2MetaForCausalLM):
    config_class = MPLUGOwl2Config

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MPLUGOwl2LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(self, input_type=None, **kwargs):
        if input_type is None:
            return self.forward_single(**kwargs)
        elif input_type == "single":
            kwargs_desp = self.get_subitem(kwargs, task_type="description")
            kwargs_score = self.get_subitem(kwargs, task_type="score")
            loss_desp = 0
            if len(kwargs_desp["task_types"]) > 0:
                del kwargs_desp["task_types"]
                output_desp = self.forward_single(**kwargs_desp)
                loss_desp = output_desp.loss
            loss_score = 0
            if len(kwargs_score["task_types"]) > 0:
                del kwargs_score["task_types"]
                output_score = self.forward_single(
                    use_softkl_loss=self.config.softkl_loss,
                    **kwargs_score,
                )
                loss_score = output_score.loss
            if dist.get_rank() == 0:
                loss_desp_item = loss_desp if isinstance(loss_desp, int) else loss_desp.item()
                loss_score_item = loss_score if isinstance(loss_score, int) else loss_score.item()
                print(
                    f"[loss (w/o weight) | "
                    f"description loss: {round(loss_desp_item, 6)}, "
                    f"score loss: {round(loss_score_item, 6)}]"
                )
            loss = self.config.weight_desp * loss_desp + self.config.weight_next_token * loss_score
            return CausalLMOutputWithPast(loss=loss)
        elif input_type == "pair":
            return self.forward_pair(**kwargs)
        else:
            raise ValueError

    def softkl_loss(self, logits, labels, level_probs):
        batch_size = logits.shape[0]
        level_prefix = torch.tensor(self.config.level_prefix).to(labels.device)
        idx_prefix_label = find_prefix(labels, level_prefix)  # B
        idx_level_label = idx_prefix_label + level_prefix.shape[0]

        level_ids_label = labels[torch.arange(batch_size), idx_level_label]
        for level_id in level_ids_label:
            assert level_id in self.config.level_ids

        # After padding in prepare_inputs_labels_for_multimodal(), the length of labels will be the same as logits
        assert logits.shape[1] == labels.shape[1]
        idx_level_logit = idx_level_label - 1
        logits_level_ids = logits[
            torch.arange(batch_size), idx_level_logit
        ].contiguous()  # [B, V]

        preds = torch.softmax(logits_level_ids, dim=1)  # [B, V]
        target = torch.zeros_like(preds)  # [B, V]
        target[:, self.config.level_ids] = level_probs
        target = target.detach()

        pred_log = torch.log(preds)
        loss_kl = F.kl_div(pred_log, target, reduction="batchmean")
        return loss_kl, idx_level_label, idx_level_logit

    def forward_single(
        self,
        input_ids: torch.LongTensor = None,
        # modality_indicators: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        use_softkl_loss: Optional[bool] = None,
        level_probs: Optional[torch.Tensor] = None,
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
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        (
            input_ids,
            modality_indicators,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss_kl = None
        if use_softkl_loss and labels is not None:
            loss_kl, idx_level_label, idx_level_logit = self.softkl_loss(logits, labels, level_probs)

            def del_elements(source, idx):
                """source: [B, N] / [B, N, V],
                idx: [B, ] with the value range [0, N-1]"""
                mask = torch.ones([*source.shape[:2]], dtype=torch.bool)
                for idx_1, idx_del in enumerate(idx):
                    mask[idx_1, idx_del] = False
                if len(source.shape) == 2:
                    source_del = source[mask].view(source.size(0), source.size(1) - 1)
                else:
                    assert len(source.shape) == 3
                    source_del = source[mask].view(source.size(0), source.size(1) - 1, source.size(2))
                return source_del

            labels_del = del_elements(labels, idx_level_label)
            logits_del = del_elements(logits, idx_level_logit)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if loss_kl is None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            else:
                shift_logits = logits_del[..., :-1, :].contiguous()
                shift_labels = labels_del[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if loss is not None and loss_kl is not None:
            loss = loss + self.config.weight_softkl * loss_kl

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_score(self, item):
        outputs = self.forward_single(
            input_ids=item["input_ids"],
            attention_mask=item["attention_mask"],
            labels=item["labels"],
            images=item["images"],
            return_dict=True,
            use_softkl_loss=self.config.softkl_loss,
            level_probs=item["level_probs"],
        )

        batch_size = outputs.logits.shape[0]
        level_prefix = torch.tensor(self.config.level_prefix).to(item["labels"].device)
        idx_prefix_label = find_prefix(item["labels"], level_prefix)  # B
        idx_level_label = idx_prefix_label + level_prefix.shape[0]

        level_ids_label = item["labels"][torch.arange(batch_size), idx_level_label]
        for level_id in level_ids_label:
            assert level_id in self.config.level_ids

        num_vision_tokens = outputs.logits.shape[1] - item["labels"].shape[1]
        idx_level_logit = idx_level_label + num_vision_tokens - 1
        logits_level_ids = outputs.logits[
            torch.arange(batch_size), idx_level_logit
        ].contiguous()  # [B, V]

        probs_org = torch.softmax(logits_level_ids, dim=1)  # [B, V]
        loss_in_level = 1 - probs_org[:, self.config.level_ids].contiguous().sum(dim=1)  # [B, 5] -> [B, ]
        bound = torch.tensor(1e-2).to(loss_in_level)
        loss_in_level = torch.max(bound, loss_in_level.mean())  # level prob > 0.99

        if self.config.closeset_rating_loss:
            logits_levels = logits_level_ids[:, self.config.level_ids].contiguous()
            probs = torch.softmax(logits_levels, dim=1)
        else:
            probs = probs_org[:, self.config.level_ids].contiguous()

        weights = torch.tensor([5, 4, 3, 2, 1]).to(probs)
        scores = torch.matmul(probs, weights)

        variances = (weights.repeat(batch_size, 1) - scores.unsqueeze(1)) ** 2
        stds = torch.sqrt(torch.sum(probs * variances, dim=1))
        return scores, stds, outputs.loss, loss_in_level

    def get_subitem(self, item, task_type):
        for key in list(item.keys()):
            if item[key] is None:
                del item[key]

        subitem = {}
        for key in item:
            subitem[key] = []
        for idx in range(len(item["task_types"])):
            if item["task_types"][idx] == task_type:
                for key in item:
                    subitem[key].append(item[key][idx])

        batch_size = torch.tensor(len(subitem["task_types"])).cuda()
        world_size = dist.get_world_size()
        batch_size_allrank = [torch.tensor(0).cuda() for _ in range(world_size)]
        dist.barrier()
        dist.all_gather(batch_size_allrank, batch_size)
        batch_size_max = torch.stack(batch_size_allrank, dim=0).max().item()
        batch_size_min = torch.stack(batch_size_allrank, dim=0).min().item()

        for key in item:
            subitem[key] = extend_list(subitem[key], batch_size_max, batch_size_min)
            if torch.is_tensor(item[key]) and len(subitem[key]):
                subitem[key] = torch.stack(subitem[key], dim=0)
        return subitem

    def forward_pair(self, item_A, item_B, **kwargs):
        item_A_desp = self.get_subitem(item_A, task_type="description")
        item_B_desp = self.get_subitem(item_B, task_type="description")
        assert item_A_desp["task_types"] == item_B_desp["task_types"]
        item_A_score = self.get_subitem(item_A, task_type="score")
        item_B_score = self.get_subitem(item_B, task_type="score")
        assert item_A_score["task_types"] == item_B_score["task_types"]

        # calculate loss_desp for description tasks
        loss_desp = 0
        if len(item_A_desp["task_types"]) > 0:
            outputs = self.forward_single(
                input_ids=item_A_desp["input_ids"],
                attention_mask=item_A_desp["attention_mask"],
                labels=item_A_desp["labels"],
                images=item_A_desp["images"],
                return_dict=True,
                use_softkl_loss=False,
            )
            loss_desp = outputs.loss

        # calculate loss_score for score tasks
        loss_score = 0
        if len(item_A_score["task_types"]) > 0:
            gt_scores_A = item_A_score["gt_scores"]
            pred_scores_A, pred_stds_A, loss_next_token_A, loss_in_level_A = self.get_score(item_A_score)
            gt_scores_B = item_B_score["gt_scores"]
            pred_scores_B, pred_stds_B, loss_next_token_B, loss_in_level_B = self.get_score(item_B_score)

            if not self.config.continuous_rating_loss:
                loss_rank = self.binary_rating_loss(pred_scores_A, gt_scores_A, pred_scores_B, gt_scores_B)
            else:
                gt_stds_A = item_A_score["stds"]
                gt_stds_B = item_B_score["stds"]
                assert (gt_stds_A >= 0).all() and (gt_stds_B >= 0).all()
                loss_rank = self.rating_loss(
                    pred_scores_A,
                    pred_stds_A,
                    gt_scores_A,
                    gt_stds_A,
                    pred_scores_B,
                    pred_stds_B,
                    gt_scores_B,
                    gt_stds_B,
                )

            loss_next_token = loss_next_token_A + loss_next_token_B
            loss_in_level = loss_in_level_A + loss_in_level_B
            if dist.get_rank() == 0:
                print(
                    f"[score loss (w/o weight) | "
                    f"ranking loss: {round(loss_rank.item(), 6)}, "
                    f"next token loss: {round(loss_next_token.item(), 6)}, "
                    f"in level loss: {round(loss_in_level.item(), 6)}]"
                )

            loss_rank = self.config.weight_rank * loss_rank

            if self.config.weight_next_token:
                assert self.config.weight_next_token > 0
                loss_next_token = self.config.weight_next_token * loss_next_token
            else:
                loss_next_token = 0

            if self.config.weight_in_level:
                assert self.config.weight_in_level > 0
                loss_in_level = self.config.weight_in_level * loss_in_level
            else:
                loss_in_level = 0

            loss_score = loss_rank + loss_next_token + loss_in_level

        if dist.get_rank() == 0:
            loss_desp_item = loss_desp if isinstance(loss_desp, int) else loss_desp.item()
            loss_score_item = loss_score if isinstance(loss_score, int) else loss_score.item()
            print(
                f"[loss (w/o weight) | "
                f"description loss: {round(loss_desp_item, 6)}, "
                f"score loss: {round(loss_score_item, 6)}]"
            )

        loss = self.config.weight_desp * loss_desp + loss_score
        return CausalLMOutputWithPast(loss=loss)

    def rating_loss(
        self,
        pred_scores_A,
        pred_stds_A,
        gt_scores_A,
        gt_stds_A,
        pred_scores_B,
        pred_stds_B,
        gt_scores_B,
        gt_stds_B,
    ):
        # eps=1e-8 is important. eps=0 is unable to step, and lr keeps unchanged.
        eps = 1e-8
        if self.config.use_fix_std:
            pred = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / 2))  # 2 -> sqrt(2 * (1**2 + 1**2))
        else:
            pred_var = pred_stds_A * pred_stds_A + pred_stds_B * pred_stds_B + eps
            if self.config.detach_pred_std:
                pred_var = pred_var.detach()
            pred = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / torch.sqrt(2 * pred_var)))
        gt_var = gt_stds_A * gt_stds_A + gt_stds_B * gt_stds_B + eps
        gt = 0.5 * (1 + torch.erf((gt_scores_A - gt_scores_B) / torch.sqrt(2 * gt_var))).to(pred.device)
        gt = gt.detach()
        loss = (1 - (pred * gt + eps).sqrt() - ((1 - pred) * (1 - gt) + eps).sqrt()).mean()
        return loss

    def binary_rating_loss(self, pred_scores_A, gt_scores_A, pred_scores_B, gt_scores_B):
        pred = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / 2))  # 2 -> sqrt(2 * (1**2 + 1**2))
        gt = (gt_scores_A > gt_scores_B).to(pred.dtype).to(pred.device)
        gt = gt.detach()
        if self.config.binary_rating_loss == "bce":
            loss = F.binary_cross_entropy(pred, gt)
        elif self.config.binary_rating_loss == "fidelity":
            loss_1 = 1 - pred[gt == 1].sqrt()
            loss_2 = 1 - (1 - pred[gt == 0]).sqrt()
            loss = (loss_1.sum() + loss_2.sum()) / pred_scores_A.shape[0]
        else:
            raise NotImplementedError(f"Wrong type of binary_rating_loss: {self.config.binary_rating_loss}")
        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("mplug_owl2", MPLUGOwl2Config)
AutoModelForCausalLM.register(MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM)

replace_llama_modality_adaptive()
