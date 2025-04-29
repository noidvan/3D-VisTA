import random

import torch

from pipeline.registry import registry

def random_point_cloud(pc, pc_mask, mask_ratio):
    output_mask = []
    for i in range(len(pc)):
        if pc_mask[i] == 0:
            output_mask.append(0)
        else:
            prob = random.random()
            if prob < mask_ratio:
                output_mask.append(0)
            else:
                output_mask.append(1)
    
    output_mask = torch.tensor(output_mask, dtype=torch.bool)
    return output_mask

def random_caption_word(tokens, tokens_mask, tokenizer, vocab, mask_ratio):
    output_label = []
    output_tokens = tokens.clone()
    for i, token in enumerate(tokens): # 101 cls 102 sep use them as SOS and EOS token
        if tokens_mask[i] == 0 or token == 101:
            output_label.append(-1)
        elif token == 102:
            output_tokens[i] = tokenizer.mask_token_id
            output_label.append(vocab.token_to_id('[EOS]'))
        else:
            prob = random.random()
            # mask token with 15% probability
            if prob < mask_ratio:
                output_tokens[i] = tokenizer.mask_token_id
                output_label.append(vocab.token_to_id(tokenizer.decode([tokens[i]])))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
    output_label = torch.Tensor(output_label).long()
    return output_tokens, output_label

def random_word(tokens, tokens_mask, tokenizer, mask_ratio):
    output_label = []
    output_tokens = tokens.clone()
    for i, token in enumerate(tokens):
        if tokens_mask[i] == 0:
            output_label.append(-1)
        else:
            prob = random.random()
            # mask token with 15% probability
            if prob < mask_ratio:
                prob /= mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    output_tokens[i] = tokenizer.mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    output_tokens[i] = random.choice(list(tokenizer.vocab.items()))[1]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                output_label.append(token.item())
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
    output_label = torch.Tensor(output_label).long()
    return output_tokens, output_label

def pad_tensors(tensors, lens=None, pad=0):
    assert tensors.shape[0] <= lens
    if tensors.shape[0] == lens:
        return tensors
    shape = list(tensors.shape)
    shape[0] = lens - shape[0]
    res = torch.ones(shape, dtype=tensors.dtype) * pad
    res = torch.cat((tensors, res), dim=0)
    return res