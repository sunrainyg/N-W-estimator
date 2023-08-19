import torch

def create_shift_matrix(input_matrix):
    shift_matrix = torch.zeros_like(input_matrix)
    num_rows, num_cols = input_matrix.size()

    for i in range(num_rows):
        shift_amount = i + 1  # Shift amount for each row
        shift_matrix[i] = torch.cat((input_matrix[i, -shift_amount:], input_matrix[i, :-shift_amount]))

    return shift_matrix

# 创建一个示例矩阵
input_matrix = torch.tensor([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12]])

shifted_matrix = create_shift_matrix(input_matrix)
print("原始矩阵:\n", input_matrix)
print("移位矩阵:\n", shifted_matrix)
