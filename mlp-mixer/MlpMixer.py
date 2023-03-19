from torch import nn
from torch import einsum
from torch import Tensor
from torch import mean

def patches(X : Tensor, P : int, H : int, W : int, C : int) -> Tensor:
    '''
        ---------------------------------------------------------------
        Description:
        X có số chiều là Numbers * Channels * Height * Width. \n
        P là kích thước của chiều cao, chiều rộng của một phần ảnh (P). \n
        H (Height) là chiều cao của bức ảnh nguyên. \n
        W (Width) là chiều rộng của bức ảnh nguyên. \n
        C (Channels) là số kênh của bức ảnh nguyên. \n
        ---------------------------------------------------------------
        Returns:
        X_new với số chiều (Numbers * S * O) \n
        O (channels) = C * P * P
    '''
    # Kiểm tra thử việc lựa chọn giá trị có chuẩn hay không
    # (tránh trường hợp giá trị chọn không chia hết gây chồng lắp bước tính)
    S = H * W / (P * P)
    assert H * W % (P * P) == 0, "Kích thước P không chuẩn. H = {}. W = {}. P = {}. S = {}. Dư : {}" \
        .format(H, W, P, S, H * W % (P * P))
    # Lần lượt chia bức ảnh đầu vào thành từng phần tương ứng với chia patch dựa trên từng chiều cao và chiều rộng
    # Đầu ra với số chiều (Numbers * Channels * Sh * Sw * P * P)
    # Với Sh, Sw lần lượt là giá trị patch ứng với từng chiều cao và chiều rộng
    X = X.unfold(2, P, P) \
        .unfold(3, P, P)
    # Tiến hành đổi lại số chiều của X thành (Numbers * Sh * Sw * Channels * P * P)
    # với mục tiêu gộp lại thành số chiều nhỏ hơn
    X = einsum("nchwab->nhwcab", X)

    # Tiến hành sửa lại kích thước của đầu vào thành 
    # (Numbers * S * O) với O = C * P * P
    size = X.size() 
    X = X.reshape(size[0], int(S), P * P * C)
    return X

'''
    Là một khối mạng đa tầng gồm hai lớp fully-connected và một hàm phi tuyến GELU
    Tùy vào vị trí lắp đặt và đầu vào mà khối này có nhiệm vụ khác nhau 
'''
class MLP(nn.Module):
    def __init__(self, sinp : int, hidden : int ,sout : int):
        # sinp viết tắt của standard input nghĩa là đầu vào chuẩn
        # hidden là số đơn vị tính toán ẩn
        # sout viết tắt của standard out nghĩa là đầu ra chuẩn
        super().__init__()
        self.l1 = nn.Linear(sinp, hidden)
        self.l2 = nn.Linear(hidden, sout)
        self.gelu = nn.GELU()
    
    def forward(self, X : Tensor) -> Tensor:
        X = self.l1(X)
        X = self.gelu(X)
        X = self.l2(X)
        # print(X.shape)
        return X

'''
    Là bộ phận kết hợp hai khối MLP với hai nhiệm vụ khác nhau
    Được gọi tên lần lượt là Token-Mixing và Channel-Mixing
'''
class MLP_Layer(nn.Module):
    def __init__(self, patches : int, channels : int, sout : int, DS : int, DC : int):
        # Giải thích 
        # Dựa vào việc tách một bức ảnh thành các bức nhỏ hơn thì ta có
        # patches là số phần mà bức ảnh lớn tạo thành bao nhiêu bức ảnh con
        # channels là biểu diễn ứng với một bức ảnh nhỏ đó (kết quả tính toán lan truyền
        # qua các khối)
        # DS là số đơn vị ẩn của khối Token-Mixing
        # DC là số đơn bị ẩn của Channel-Mixing
        super().__init__()
        # token-mixing có nhiệm vụ học những mối quan hệ giữa các bức ảnh nên
        # đầu vào là patches
        self.tm = MLP(patches, DS, patches)
        self.cm = MLP(channels, DC, channels)

        # Lớp chuẩn hóa cho khối Token-Mixing
        self.ln_tm = nn.LayerNorm((channels, patches))

        # Lớp chuẩn hóa cho khối Channel-Mixing
        self.ln_cm = nn.LayerNorm((patches, channels))
    
    def forward(self, X : Tensor):
        # Đầu vào mặc định là Patches * Channels (Hàng * Cột)
        # Token-Mixing thực hiện tính toán trên toàn bộ phần ảnh (các patch ảnh)
        X = self.transpose(X)
        # print(X.shape)
        u = X + self.tm(
            self.ln_tm(X)
        )
        u = self.transpose(u)
        y = u + self.cm(
            self.ln_cm(u)
        )
        return y

    def transpose(self, X : Tensor):
        # Đầu vào là Numbers * A * B
        return einsum("nab->nba", X)
'''
    Kiến trúc của mô hình
'''
class MLP_Mixer(nn.Module):
    def __init__(self, 
        blocks : int, patches : int, 
        DS : int, DC : int, 
        channels : int, height : int, width : int, channels_image : int, 
        cls : int, dropout : int = 0.1):
        # blocks là số lớp mlp-layer
        # Kiến trúc khởi tạo như này yêu cầu biết và có sẵn
        # patches -> Là kích thước bức ảnh con
        # DS, DC là số lớp ẩn giữa hai đơn vị MLP
        # main_hidden là số chiều tính toán chính
        # height là chiều cao của bức ảnh gốc
        # width là chiều rộng của bức ảnh gốc
        # channels_image là số kênh màu của bức ảnh
        # dropout dùng trong tắt quá trình học
        super().__init__()

        # Lưu trữ thông số
        self.patches = patches
        self.channels = channels
        self.height = height
        self.width = width
        self.channels_image = channels_image

        _channels = channels_image * patches * patches
        _num_imgs = int((height * width) / (patches * patches))
        # Linear-Embedding
        self.le = nn.Linear(_channels, channels)

        # Số lớp mlp-layers dành cho tính toán
        self.mlp_layers = nn.ModuleList(
            [MLP_Layer(_num_imgs, channels, channels, DS, DC) for _i in range(blocks)]
        )

        self.dropout = nn.Dropout(dropout)

        # Số lớp phân loại
        self.classifier = nn.Linear(channels, cls)
        self.softmax = nn.Softmax(dim=0)

    def __call__(self, X : Tensor):
        X = patches(X, self.patches, self.height, self.width, self.channels_image)
        X = self.le(X)
        # print(X.shape)

        for __, layer in enumerate(self.mlp_layers):
            # print(X.shape)
            X = layer.forward(X)
            # print(X.shape)

        X = mean(X, dim=1)
        # print(X.shape)
        X = self.dropout(X)
        X = self.classifier(X)
        X = self.softmax(X)

        return X
