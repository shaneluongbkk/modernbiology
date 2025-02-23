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
Cho cell chạy qua 2 block transformer và xong
