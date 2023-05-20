import torch

def pad_sublists(list_of_lists, target_length):
    padded_lists = []
    for sublist in list_of_lists:
        sublist_len = len(sublist)
        num_copies = target_length // sublist_len
        padding_len = target_length % sublist_len
        padded_sublist = sublist * num_copies + sublist[:padding_len]
        padded_lists.append(padded_sublist)
    return torch.tensor(padded_lists)

# Example usage
original_list = [[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)], [(7.0, 8.0), (9.0, 10.0), (11.0, 12.0)]]
target_length = 50

tensor = pad_sublists(original_list, target_length)
print(tensor.shape)