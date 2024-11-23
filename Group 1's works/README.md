
# POSITIONAL EMBEDING AND WORD EMBEDING
## Giới thiệu
- Đây là chương trình xử lý đầu vào cho transformer sử dụng kỹ thuật Positional Embedding và Word Embedding và được viết bằng ngôn ngữ c++. Chương trình thực hiện các nhiệm vụ: 
   + Nhập một câu từ người dùng và thực hiện tách từ (tokens)
   + Kết hợp word embedding và positional encoding để tạo một vector tổng hợp cho mỗi từ.
   + Đội bọn em không test chất lượng vì đây là phần decoder.

## Quy trình hoạt động
**INPUT:** 
- File embedding (mini_test.txt: Đây là file sử dụng tạm thời để thực hiện unit test) chứa các từ và vector nhúng tương ứng.
- Câu nhập từ người dùng.
- Size vector từ: 1x300.

**MAIN:** 
- Load embedding: Đọc các vector từ file.
- Split sentence: Tách câu nhập vào thành danh sách các từ (tokens).
- Positional encoding: Tạo vector vị trí cho mỗi từ dựa trên vị trí của nó trong câu.
- Combine embedding: Cộng Word embedding của từ và positional encoding để tạo vector tổng hợp.

**OUTPUT:**
- Ma trận embedding cho các từ trong câu (Token_numbers x 300).
- Các vector tổng hợp của Word embedding và Positional embedding.

**CODE:**
- Sử dụng ngôn ngữ lập trình C++.
- File CSDL thử nghiệm: mini_test.txt.
- Các hàm được sử dụng: 
    + load_vector_position: Lấy vị trí của từng từ trong câu.
    + load_embedding: Đọc file nhúng đầu vào.
    + get_embedding: Lấy vector của từ.
    + split_senquence: Chia câu thành từng từ.
- README.md: Hướng dẫn sử dụng.

## TÁC GIẢ:
**TEAM:** 1.
**THÀNH VIÊN**:
- Nguyễn Vũ Bách.
- Trịnh Hữu Trí.

