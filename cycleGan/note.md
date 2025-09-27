# CycleGan
## 1. Giới thiệu chung

**CycleGAN (Cycle-Consistent Generative Adversarial Network)** là một kiến trúc **học sinh ảnh không giám sát** dùng để chuyển đổi hình ảnh từ một miền (domain) này sang miền khác (ví dụ: ảnh thường → tranh phong cách Monet).  
Nguyên lý hoạt động cốt lõi:

- Dùng **hai bộ sinh (Generators)** để chuyển đổi qua lại giữa hai miền ảnh (X→Y và Y→X).
- Dùng **hai bộ phân biệt (Discriminators)** để đánh giá tính chân thực của ảnh sinh ra.
- Ràng buộc thêm **chu kỳ nhất quán (Cycle Consistency Loss)** để đảm bảo ảnh khi biến đổi qua lại X→Y→X vẫn giữ nguyên nội dung gốc.

---

## 2. Kiến trúc mạng trong `model.py`
### 2.1 ResidualBlock
- Mỗi block chứa 2 lớp **Conv2D**, chuẩn hóa (**GroupNorm, InstanceNorm**) và kích hoạt **ReLU**.
- Cơ chế **skip connection** giúp giữ thông tin gốc và giảm gradient vanishing.

### 2.2 Generator
- Gồm ba phần:
    1. **Downsampling**: giảm kích thước ảnh, trích xuất đặc trưng.
    2. **Bottleneck**: nhiều Residual Blocks để học đặc trưng sâu.
    3. **Upsampling**: khôi phục lại kích thước ảnh, tạo ảnh đầu ra.
- Lớp cuối dùng **Tanh** để chuẩn hóa giá trị pixel về [-1, 1].

### 2.3 Discriminator
- Mạng CNN với nhiều lớp Conv2D và LeakyReLU.
- Đầu ra là bản đồ đặc trưng (feature map) thay vì một giá trị duy nhất → **PatchGAN discriminator**. Điều này giúp mô hình đánh giá tính chân thực ở mức **cục bộ (patch)** chứ không phải toàn ảnh.

---

## 3. Cơ chế huấn luyện trong `train.py`
### 3.1 Thành phần chính
- **2 Generators**: G_XtoY và G_YtoX.
- **2 Discriminators**: D_X và D_Y.
- Hàm mất mát chính:
    - **Adversarial Loss (BCEWithLogits)**: ép ảnh sinh ra trông giống ảnh thật.
    - **Cycle Consistency Loss (L1 Loss)**: đảm bảo X→Y→X ≈ X và Y→X→Y ≈ Y.
    - Tổng loss:  
        $\mathcal{L}_G = \mathcal{L}_{GAN} + \lambda \mathcal{L}_{cycle}$  
        với ($\lambda$ = 10) để nhấn mạnh tính nhất quán chu kỳ.
### 3.2 Quy trình huấn luyện

- **Bước 1**: Huấn luyện Discriminators:
    - So sánh ảnh thật với ảnh giả (fake) từ Generator.
    - Loss = trung bình loss của D_X và D_Y.
- **Bước 2**: Huấn luyện Generators:
    - Tạo ảnh giả để đánh lừa D.
    - Tính thêm cycle loss để đảm bảo tính nhất quán.
- **Bước 3**: Cập nhật tham số bằng **Adam optimizer** với β=(0.5, 0.99).

---

## 4. Nguyên lý hoạt động tổng quát

- **Discriminator** học cách phân biệt ảnh thật và ảnh sinh.
- **Generator** học cách đánh lừa Discriminator bằng ảnh giả ngày càng giống thật.
- **Cycle consistency** giữ cho ảnh không bị mất nội dung khi dịch qua lại hai miền.
- Cơ chế **đấu tay đôi (minimax game)** giữa G và D giúp mô hình tiến dần đến trạng thái cân bằng:
    - Ảnh sinh ra có chất lượng cao, gần giống ảnh thật.
    - Discriminator khó có thể phân biệt.

---

# 5. Ví dụ
- **Miền X**: ảnh phong cảnh chụp bằng máy ảnh.
- **Miền Y**: tranh của họa sĩ Monet.
- **Ứng dụng**: Generator G_XtoY sẽ biến ảnh chụp thành bức tranh giống Monet (giữ lại hình dáng núi, hồ, cây nhưng thay đổi màu sắc và bút pháp).
- **Ngược lại**: Generator G_YtoX biến tranh Monet thành ảnh "thật" gần giống cảnh ngoài đời.