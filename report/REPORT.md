# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (T4)

**Python / PyTorch / Ultralytics:** `3.13.16` / `2.11.0+cu130` / `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Record nói về **cả bức ảnh**. Nó không có `bbox_xyxy` hay `polygon_xy` nên không chỉ ra vị trí nào trong ảnh. Model xếp hạng các lớp ImageNet-1K cho toàn ảnh `traffic`, và JSON lưu lại 5 lớp có score cao nhất.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Danh sách lớp **đi kèm taxonomy/checkpoint**. `yolo11n-cls.pt` dùng ImageNet-1K nên chỉ chọn được trong các lớp của taxonomy này, không tự nghĩ ra tên lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Vì `class_id` chỉ có nghĩa trong một taxonomy cụ thể, và tên lớp cũng có thể viết khác nhau giữa các taxonomy. Ví dụ trong JSON, bàn ăn là `532 / dining_table` ở ImageNet-1K nhưng là `60 / dining table` ở COCO-80. Ghi đủ ba trường thì nhãn mới tra lại được và không bị lẫn khi gộp dữ liệu.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Guideline cần nói rõ tiêu chí chọn lớp: gán theo chủ thể chính (vật nổi bật hoặc chiếm diện tích lớn nhất) hay theo bối cảnh chung của ảnh. Nó cũng cần nói phải làm gì khi hai lớp ngang nhau. Ví dụ ảnh `traffic` có nhiều xe buýt, ô tô và người đi bộ, nên nếu không có tiêu chí thì mỗi annotator có thể chọn một lớp khác nhau. Gặp ca khó thì cầu cứu Lab Coach.

6. Vì sao không được lấy `score` của model làm ground truth?

   `score` chỉ là độ tin của model vào dự đoán của chính nó và không thể chứng minh dự đoán là đúng. Ảnh `traffic` có `cab` = 0.510915 nhưng trên ảnh chủ yếu là xe buýt và ô tô. Trong khi đó Ảnh `kitchen` còn bị đoán là `gong` = 0.420191 mặc dù đó là nhà bếp. Một lý do nữa là ảnh mẫu là cảnh COCO có nhiều vật, còn taxonomy là ImageNet-1K, nên top-1 có thể trông khá lạ (dị thường). Ground truth phải do người gán theo guideline và được kiểm tra lại bởi người có thẩm quyền.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Gốc tọa độ là góc trên-trái của ảnh (ảnh rộng 640, cao 427 pixel). Góc trên-trái của box ở x = 385.33, y = 69.24, tức là phía trên đầu người đầu bếp. Góc dưới-phải ở x = 498.92, y = 348.92, tức là dưới chân. Box rộng khoảng 113.58 pixel và cao khoảng 279.68 pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Trên ảnh `kitchen`, ngưỡng **0.20 có 17 prediction** và ngưỡng **0.60 có 6 prediction**. Ngưỡng 0.35 dùng cho JSON có 11 prediction.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Reviewer phải xem nhiều hộp có điểm thấp hơn: loại hộp sai hoặc trùng, sửa lớp sai, chỉnh box lệch. Từ 0.60 xuống 0.20, ảnh `kitchen` có thêm 11 hộp, gồm `bowl`, `cup`, `spoon`, `potted plant`, `dining table` và `bottle`. Đổi lại, hạ ngưỡng giúp ít bỏ sót vật hơn: `dining table` chỉ xuất hiện ở 0.20. Ngưỡng chỉ lọc prediction, không phải quy tắc bỏ vật khi tạo ground truth, nên vật thật có score thấp vẫn phải được gán nhãn.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Mỗi vật thể có một box riêng, bốn cạnh ôm sát phần **có thể nhìn thấy được** của vật, không chừa thêm nền và không đoán phần bị che hoặc nằm ngoài ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Vật bị cắt mép là cánh tay ở mép trái ảnh: `person` 0.610931, `bbox_xyxy` = [0.12, 263.18, 61.55, 310.78], chỉ thấy một phần người. Mình sẽ **viết thành quy tắc guideline**, vì vật cắt mép gặp thường xuyên và có thể xử lý thống nhất: vẫn gán lớp `person` và vẽ box theo phần nhìn thấy. Chỉ escalation nếu phần nhìn thấy nhỏ đến mức không chắc đó là vật gì.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Box chỉ cho một hình chữ nhật bao ngoài, nên có cả nền bên trong. Polygon đi theo đường viền thật của vật: với `kitchen-001` là 348 điểm bám theo đầu, vai, tay và chân. Nhờ đó biết được hình dạng và pixel nào thuộc vật.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   `instance_id` dùng để phân biệt từng đối tượng riêng trong một ảnh. Nó không phải class ID vì `kitchen-002` và `kitchen-003` khác `instance_id` nhưng cùng `class_id` = 45 (`bowl`). Nó không phải tracking ID vì chỉ là số thứ tự trong ảnh đó, không dùng để theo dõi cùng một vật qua nhiều ảnh hay nhiều frame video.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Vẽ mask bám sát biên phần nhìn thấy của vật. Khi hai vật chồng nhau, pixel chỗ đó thuộc về vật nằm phía trước, và không vẽ đoán phần bị che.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Hai vật chạm nhau là cái bát `kitchen-006` đặt trên bàn `kitchen-005`. Hai polygon chồng lên nhau khoảng 907 pixel (mình tính bằng cách tô hai polygon lên trên lưới ảnh), trong vùng x từ 1 đến 84, y từ 304 đến 370. Mình **dùng guideline**, vì đây là tình huống lặp lại và đã có quy tắc rõ: bát nằm phía trước nên pixel đó thuộc về bát, mask bàn phải chừa chỗ này. Chỉ escalation nếu không xác định được vật nào nằm trước.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh: `class_id`, `class_name`, `taxonomy_name` | Ảnh `traffic` có `cab` 0.510915 đứng đầu dù chủ yếu là xe buýt và ô tô; ảnh `kitchen` bị đoán `gong` 0.420191 | Xem cả ảnh, chọn một lớp trong taxonomy theo tiêu chí của guideline | Lớp có đúng taxonomy và đúng tiêu chí không; ca nhiều chủ đề có xử lý thống nhất không |
| Phát hiện vật thể | Mỗi vật một box `xyxy` (pixel) kèm lớp | Cánh tay cắt mép `person` 0.610931 tại [0.12, 263.18, 61.55, 310.78]; `oven` 0.6343 tại [489.09, 200.73, 617.81, 344.88] trông giống tủ có chậu kim loại hơn là lò; ảnh có bàn lớn nhưng ở ngưỡng 0.35 không có box `dining table` (chỉ xuất hiện ở 0.20) | Vẽ box ôm sát phần nhìn thấy cho từng vật và gán lớp | Có sót vật, box trùng, sai lớp hoặc box lệch không; hộp điểm thấp có đúng không |
| Instance segmentation | Mỗi đối tượng một polygon/mask kèm lớp và `instance_id` riêng | Bát `kitchen-006` và bàn `kitchen-005` chồng nhau khoảng 907 pixel; `kitchen-006` còn sát mép trái ảnh | Vẽ biên theo phần nhìn thấy, tách riêng từng đối tượng, xử lý chỗ chồng theo guideline | Biên có sát vật không; mask có lấn sang vật khác không; mỗi đối tượng có đúng một instance không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Chỉ dùng các ảnh COCO do notebook cung cấp, không tải thêm ảnh hay dữ liệu cá nhân lên, và không ghi họ tên, MSSV hay thông tin nhạy cảm vào báo cáo/output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Mình sẽ dừng xử lý ngay, không chia sẻ dữ liệu đó, và báo cho Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
