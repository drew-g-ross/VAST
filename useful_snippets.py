# below code is in batch_get in VAST class. I looked at vast27m_ret method of VAST class to find them
# CAPTION EMBEDDINGS (feat_t_vision_caption)
elif key == 'vision_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.vision_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")
caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens

elif key == 'feat_t_vision_caption':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

# VIDEO EMBEDDINGS
elif key == 'feat_v':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            feat_v = self.contra_head_v(vision_output_pooled)
            feat_v = F.normalize(feat_v,dim=-1)
            batch[key] = feat_v
elif key == 'vision_output':
            vision_output = self.forward_vision_encoder(batch.vision_pixels)
            batch[key] = vision_output

# read pixels in vision_mapper read(_id)