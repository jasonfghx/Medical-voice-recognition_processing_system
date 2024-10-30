import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)

class AudioUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(1, 32)  # 輸入單聲道音訊
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        
        # Decoder
        self.dec4 = ConvBlock(256, 128)
        self.dec3 = ConvBlock(256, 64)  # 256 因為要考慮skip connection
        self.dec2 = ConvBlock(128, 32)
        self.dec1 = ConvBlock(64, 32)
        
        self.final = nn.Conv1d(32, 1, kernel_size=1)
        
        # Pooling and Upsampling
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool(enc3)
        
        enc4 = self.enc4(pool3)
        
        # Decoder path
        dec4 = self.dec4(enc4)
        up3 = self.upsample(dec4)
        
        # Skip connections
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(up3)
        up2 = self.upsample(dec3)
        
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(up2)
        up1 = self.upsample(dec2)
        
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(up1)
        
        return self.final(dec1)

# 訓練代碼
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    for noisy_audio, clean_audio in train_loader:
        noisy_audio = noisy_audio.to(device)
        clean_audio = clean_audio.to(device)
        
        optimizer.zero_grad()
        output = model(noisy_audio)
        loss = criterion(output, clean_audio)
        loss.backward()
        optimizer.step()
        
    return loss.item()

# 數據預處理
def prepare_audio_data(audio_file, segment_length=16384):
    """
    將音訊檔案切分成小片段並轉換為模型輸入格式
    """
    import librosa
    
    # 加載音訊
    audio, sr = librosa.load(audio_file, sr=None)
    
    # 將音訊切分成固定長度的片段
    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    
    # 轉換為PyTorch張量
    segments = torch.FloatTensor(segments)
    segments = segments.unsqueeze(1)  # 添加通道維度
    
    return segments, sr

# 使用示例
def denoise_audio(model_path, audio_file):
    # 加載模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioUNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 準備音訊數據
    segments, sr = prepare_audio_data(audio_file)
    
    # 去噪
    denoised_segments = []
    with torch.no_grad():
        for segment in segments:
            segment = segment.unsqueeze(0).to(device)  # 添加批次維度
            denoised = model(segment)
            denoised_segments.append(denoised.cpu().squeeze().numpy())
    
    # 重建完整音訊
    denoised_audio = np.concatenate(denoised_segments)
    
    return denoised_audio, sr
