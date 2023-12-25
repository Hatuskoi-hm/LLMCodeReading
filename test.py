import torch

logits = torch.tensor([
                        [[1.0, 6.0, 6.0], [4.0, 5.0, 6.0]],
                        [[1.0, 9.0, 3.0], [4.0, 5.0, 6.0]]
                        ]
                        ,dtype=float
                        )
# print("logits shape1", logits.shape)

# # 使用 softmax 函数将 logits 转换为概率分布
# probs = torch.nn.functional.softmax(logits)

# # 计算 softmax 函数的输入
# inputs = logits[:, :-1, :]

# print(inputs.shape, inputs)
# # [[1.0, 2.0], [4.0, 5.0]]
num = torch.tensor(6.0,dtype=float)
res = torch.eq(logits, num).long()
# print(res.shape, res)
res = res.argmax(-1) - 1
print("============")
print(logits, logits.shape)
print("============")
print(torch.arange(2))
print("============")
print(res, res.shape)
pooled_logits = logits[torch.arange(2), res]
print("============")
print(pooled_logits)