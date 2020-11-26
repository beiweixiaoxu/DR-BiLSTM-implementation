import torch
import torch.nn as nn
def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask

def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)

def weighted_sum(tensor, weights, mask):
    weighted_sum = weights.bmm(tensor)
    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()
    return weighted_sum * mask


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    # 按照每个句子的长度进行排序，假如 sequences_lengths=[10, 2, 8, 20]
    # sorted_seq_lens = [2, 8, 10 ,20]; sorting_index = [1, 2, 0, 3]
    sorted_seq_lens, sorting_index =\
        sequences_lengths.sort(0, descending=descending)

    # 根据sorting_index对句子进行重排序
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range =\
        sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    # reverse_mapping = [2, 0, 1, 3]
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    # print("reverse_mapping:", reverse_mapping)
    # restoration_index = [2, 0, 1, 3]  真不知道这个到底有什么意义
    restoration_index = idx_range.index_select(0, reverse_mapping)
    # print("restoration_index:", restoration_index)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index