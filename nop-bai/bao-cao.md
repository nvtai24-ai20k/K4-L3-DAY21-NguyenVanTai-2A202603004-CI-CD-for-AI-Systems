# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Văn Tài |
| MSSV | 2A202603004 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/nvtai24-ai20k/K4-L3-DAY21-NguyenVanTai-2A202603004-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 4 | 200 | 0.05 | 3 | 0.7014 | 0.8740 |
| 5 | 100 | 0.2 | 5 | 0.7207 | 0.8760 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.2`, `max_depth=5`.

**Lý do:** Bộ này có F1 cao nhất (0.7207). Lần có accuracy cao nhất (lần 1) không phải lần có F1
cao nhất: accuracy chỉ dao động 0.846 - 0.878 trong khi F1 dao động 0.605 - 0.721, nên accuracy gần
như không phân biệt được mô hình tốt và mô hình bỏ sót lớp thu nhập cao. Về đánh đổi, hạ
`learning_rate` từ 0.1 xuống 0.05 phải tăng lên 200 cây mới đạt F1 xấp xỉ (0.7014 so với 0.7109).

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu thuộc lớp thu nhập > 50K, nên mô hình luôn đoán "thu nhập thấp" vẫn đạt accuracy
0,752 dù không phát hiện được người thu nhập cao nào. F1 của lớp dương kết hợp precision và recall
trên chính lớp thu nhập cao, nên mô hình đó có F1 = 0. Không dùng `average="weighted"` hay
`"macro"` vì chúng trộn F1 của lớp đa số (khoảng 0,92) vào: với mô hình đã chọn, F1 weighted là
0,871 trong khi F1 lớp dương chỉ 0,721, khiến ngưỡng 0,65 mất tác dụng.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `pip install` bị timeout. | Mạng chậm khi tải từ PyPI. | Thêm `--default-timeout=120 --retries 10`. |
| Test lỗi `ImportError` từ mlflow. | mlflow 2.13 không ghim SQLAlchemy, pip cài bản 2.1 không tương thích. | Ghim `sqlalchemy==2.0.36`. |
| Không tạo được VM trên GCP. | Project chưa gắn billing. | Chuyển sang AWS (S3 + EC2), ghim phiên bản `dvc-s3` / `aiobotocore` để pip không dò ngược. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7207 | 0.8760 |
| Bước 3 (thêm `train_batch2`) | 0.7297 | 0.8800 |

**Nhận xét:** Gấp đôi dữ liệu chỉ làm F1 tăng 0,009, tương đương một dự đoán đúng thêm trên 124 mẫu
dương của holdout, vì `train_batch2` cùng phân phối với dữ liệu cũ nên không mang thêm thông tin
mới. Điều Bước 3 chứng minh là quy trình tự động: commit dữ liệu kích hoạt cả bốn job và VM tự
phục vụ model mới mà không cần thao tác thủ công.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 2 - Ngưỡng quyết định: ngưỡng 0,30 cho F1 0,7463, cao hơn 0,7207 ở ngưỡng 0,5.
- [x] Bonus 3 - Precision / recall: `outputs/detail.txt` cho recall lớp cao 0,645 (44 FN, 18 FP); bỏ sót người thu nhập cao là lỗi phổ biến hơn.
- [x] Bonus 4 - Hoàn trả: Release chỉ promote model và restart VM khi F1 mới >= F1 model đang chạy.
- [x] Bonus 5 - Lệch dữ liệu: cảnh báo khi tỷ lệ lớp dương lệch > 5 điểm so với 24,8%, ghi `positive_rate` vào `report.json`.
