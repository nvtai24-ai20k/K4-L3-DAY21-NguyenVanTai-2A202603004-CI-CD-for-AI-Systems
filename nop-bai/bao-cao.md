# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Văn Tài |
| MSSV | 2A202603004 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/nvtai24-ai20k/K4-L3-DAY21-NguyenVanTai-2A202603004-CI-CD-for-AI-Systems |
| Ngày nộp | ___ |

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

**Lý do:** Bộ này cho `f1_score` cao nhất (0.7207) trên tập holdout, vượt ngưỡng 0.65 với
biên an toàn rộng. Lần chạy có accuracy cao nhất (lần 1, 0.8780) lại không phải lần có F1 cao
nhất: accuracy chỉ dao động trong khoảng 0.846 - 0.878, trong khi F1 dao động 0.605 - 0.721,
nghĩa là accuracy gần như không phân biệt được mô hình tốt và mô hình bỏ sót lớp thu nhập cao.
Lần 2 (cây nông, ít cây, learning_rate thấp) là ví dụ rõ nhất: accuracy vẫn 0.846 nhưng F1 tụt
xuống 0.605, dưới ngưỡng. Về đánh đổi, giảm `learning_rate` từ 0.1 xuống 0.05 với cùng độ sâu 3
đòi hỏi tăng `n_estimators` lên 200 mới đạt F1 xấp xỉ (0.7014 so với 0.7109), còn tăng
`learning_rate` lên 0.2 cho phép giữ 100 cây mà vẫn đạt F1 cao nhất khi cây đủ sâu.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult chỉ có 24,8% mẫu thuộc lớp thu nhập > 50K. Vì vậy một mô hình vô dụng luôn trả lời
"thu nhập thấp" vẫn đạt accuracy 0,752, một con số trông khá tốt nhưng mô hình đó không phát
hiện được bất kỳ người thu nhập cao nào. Accuracy bị lớp đa số chi phối nên không phản ánh
năng lực trên lớp mà ta thực sự quan tâm. F1 của lớp dương là trung bình điều hòa của precision
và recall trên chính lớp thu nhập cao: nó chỉ cao khi mô hình vừa tìm ra được phần lớn người thu
nhập cao (recall) vừa không gán nhầm quá nhiều (precision); mô hình "luôn đoán thấp" có F1 = 0.
Không dùng `average="weighted"` hay `average="macro"` vì hai cách này trộn F1 của lớp đa số
(khoảng 0,92) vào kết quả, kéo con số lên cao và làm ngưỡng 0,65 mất ý nghĩa: với mô hình đã
chọn, F1 weighted là 0,871 trong khi F1 lớp dương chỉ là 0,721.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `pip install -r requirements.txt` thất bại giữa chừng. | Mạng chậm, pip hết thời gian chờ khi tải gói từ PyPI. | Chạy lại với `--default-timeout=120 --retries 10`. |
| Unit test lỗi `ImportError: FallbackAsyncAdaptedQueuePool`. | mlflow 2.13 không ghim SQLAlchemy nên pip cài bản 2.1, bản này đã bỏ class mlflow cần. | Ghim `sqlalchemy==2.0.36` trong `requirements.txt` (cả máy cá nhân lẫn CI). |
| Không tạo được VM trên GCP, sau đó `pip install dvc[s3]` chạy hơn 10 phút không xong. | Project GCP chưa gắn billing; khi chuyển sang AWS, `dvc[s3]` không ghim phiên bản nên pip dò ngược qua hàng chục bản `aiobotocore`. | Chuyển sang AWS (S3 + EC2, IAM user chỉ có quyền trên đúng bucket) và ghim `dvc-s3`, `s3fs`, `fsspec`, `aiobotocore[boto3]`. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7207 | 0.8760 |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: workflow đã đọc secret `MLFLOW_TRACKING_URI` / `USERNAME` / `PASSWORD`, cần tạo tài khoản DagsHub và thêm secrets.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: ngưỡng 0,30 cho F1 0,7463 so với 0,7207 ở ngưỡng 0,5, vì mô hình thận trọng với lớp thiểu số nên hạ ngưỡng giúp tăng recall.
- [x] Bonus 3 - Báo cáo precision / recall tự động: `src/evaluate.py` ghi `outputs/detail.txt`; recall lớp cao chỉ 0,645 (44 FN so với 18 FP), nên bỏ sót người thu nhập cao là lỗi phổ biến và tốn kém hơn.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: job Release so F1 mới với `artifacts/current/report.json`, chỉ promote và restart VM khi F1 mới >= F1 cũ.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: `train.py` cảnh báo khi tỷ lệ lớp dương lệch > 5 điểm so với 24,8% và ghi `positive_rate` vào `report.json`.
