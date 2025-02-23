### scGPT example
## Đơn giản hóa mô hình
Để tiện cho quá trình hiểu được cách thức hoạt động của scGPT, ta sẽ đơn giản hóa mô hình
# Từ điển gene
Ta sử dụng từ điển 7 gene sau, với 3 token đặc biệt đã được thêm vào và mang giá trị nhỏ để ko ảnh hưởng tới suy diễn
```
"pad": 0,
    "cls": 1,
    "eoc": 2,
    "EPN2-IT1": 3,
    "CD82": 4,
    "KIAA1614": 5,
    "ANKDD1B": 6,
    "BMIQ5": 7,
    "GASK1B": 8,
    "RANBP3": 9
```
# Các kích cỡ tinh chỉnh
- Số block transformer: 2
- Chiều input: 9
- Chiều embedding: 8
## Tạo đầu vào
Ví dụ cell được cho như sau: 
```
Cell 1:
  Gene IDs:          ['BMIQ5', 'KIAA1614', 'GASK1B', 'ANKDD1B', 'CD82']
  Expression Values: [19.821866528065534, 90.30081342059579, 60.03385151918508, 92.66545675338, 87.37395838194259]
  Condition Values:  [5, 4, 1, 0, 7]
```
Thêm cls, eoc, pad token và thực hiện binning cho expression value:
```
Cell 1:
  Gene Tokens:          [1, 7, 5, 8, 6, 4, 0, 0, 2]
  Expression Tokens: [0, 0, 3, 1, 4, 2, 0, 0, 0]
  Condition Tokens:  [0, 5, 4, 1, 0, 7, 0.0, 0, 0]
```
## Embedding
Lớp embedding được định nghĩa như sau:
```
gene_embedding = nn.Embedding(len(gene_dict), embedding_dim)
condition_embedding = nn.Embedding(11, embedding_dim)
expression_fc = nn.Linear(1, embedding_dim) # FCL for expression
```
Cho token qua lớp embedding và tổng lại thu được
```
Cell 1:
  Final embedding:          tensor([[-0.7203,  0.5364,  0.4968, -1.0081,  1.7731,  1.3066,  0.0176,  1.3956],
        [-1.1351,  3.7490, -1.6256,  1.6209,  1.3269,  0.0096,  2.0012,  1.2817],
        [ 2.8275,  3.0115,  2.7899, -2.6210,  1.5789, -0.8304, -1.9447,  1.0870],
        [ 0.4802,  2.3293,  0.6298,  0.5248, -0.6761,  2.9275, -1.6134,  1.4951],
        [ 2.1639,  1.9644,  1.7440, -0.4937,  0.6969, -0.1040, -0.4468,  1.6210],
        [ 1.6501, -1.2376,  3.5985, -1.4598, -0.9352, -0.8912, -3.0465, -0.0788],
        [-1.3100,  3.1064, -0.4155, -1.2653,  1.4944, -0.5014,  4.2071, -0.4269],
        [-1.3100,  3.1064, -0.4155, -1.2653,  1.4944, -0.5014,  4.2071, -0.4269],
        [-1.2796,  0.0909,  1.4498, -0.0412, -0.2909, -0.2754,  2.6565,  2.1460]],
       grad_fn=<AddBackward0>)
  Shape:          torch.Size([9, 8])
```

## Forward
Cho cell chạy qua 2 block transformer và xong, chứ cái block này siêu hành

## Special Masking Technique
Model sẽ thực hiện chọn gene để đoán, mask nó thành thấy được và cho nó thấy các gene đã biết (ground truth)
```
def create_scgpt_attention_mask(self, gene_tokens, unknown_gene_mask, guessing_gene_mask):
        """
        Create specialized attention mask for scGPT.
        Args:
            gene_tokens (torch.Tensor): Input tokens of shape (batch_size, seq_len)
            unknown_gene_mask (torch.Tensor): Mask indicating unknown genes (1 for unknown, 0 for known) 
            guessing_gene_mask (torch.Tensor): Mask indicating the current guessing gene (1 for guessing, 0 otherwise)
        Returns:
            torch.Tensor: Attention mask of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len = gene_tokens.size()

        # Base mask for padding and EOC tokens
        pad_mask = (gene_tokens == self.pad_token) | (gene_tokens == self.eoc_token)
        cls_mask = (gene_tokens == self.cls_token)
        known_mask = ~pad_mask & ~cls_mask

        # Expand dimensions for broadcasting
        pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
        cls_mask = cls_mask.unsqueeze(1).expand(-1, seq_len, -1)
        known_mask = known_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # Mask for unknown genes
        unknown_mask = unknown_gene_mask.unsqueeze(1).expand(-1, seq_len, -1)
        unknown_self_mask = unknown_gene_mask.unsqueeze(2).expand(-1, -1, seq_len)

        # Mask for the current guessing gene
        guessing_mask = guessing_gene_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # Attention Mask Construction
        # 1. Unknown genes can see known genes but not other unknown genes
        unknown_attention = unknown_mask & (known_mask | cls_mask)

        # 2. The current guessing gene can see known genes, CLS, and previously guessed genes (left context)
        guessing_attention = guessing_mask & (
            known_mask | cls_mask | (torch.tril(torch.ones(seq_len, seq_len, device=gene_tokens.device)).bool())
        )

        # 3. CLS token sees all
        cls_attention = cls_mask

        # 4. Combine all masks
        combined_mask = pad_mask | (~(unknown_attention | guessing_attention | cls_attention))

        return combined_mask
```
