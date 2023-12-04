import torch
import torch.nn as nn
import torch.fft as fft
class FFTLinearLayer(nn.Module):
    def __init__(self, in_features, out_features,threshold):
        super(FFTLinearLayer, self).__init__()

        self.in_features=in_features
        self.out_features=out_features
        self.threshold=threshold

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))
        self.register_buffer('threshold',torch.tensor(threshold))
        # Define the fixed Linear layer: v
        # self.OPT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        
        # self.fft_filt_shape = [out_features, in_features]
        self.fft_filt_height_width=2*threshold

        # 初始化上三角矩阵的参数
        self.amplitude_gain_upper_triangular_vector = nn.Parameter(torch.ones(self.fft_filt_height_width*(self.fft_filt_height_width+1)//2))
        # self.phase_offset = nn.Parameter(torch.tensor(0.0))

        # 创建上三角矩阵，但不在forward中使用
        self.upper_triangular_matrix = self.create_upper_triangular_matrix(n)

    def create_upper_triangular_matrix(self, n):
        # 创建上三角矩阵
        upper_triangular_matrix = torch.zeros(n, n)
        # upper_triangular_matrix = torch.triu(upper_triangular_matrix, diagonal=0) #包含对角线
        
        # 使用向量填充上三角矩阵的非对角线部分
        upper_triangular_matrix += torch.diag(self.upper_triangular_vector, 0)

        return upper_triangular_matrix


    def forward(self,attn,x):

        orig_dtype = x.dtype
        #fix filter
        fix_filt= attn.weight.data
        #fft_finetune
        filt=self.fft_finetune(fix_filt,self.amplitude_gain,self.phase_offset)
        #bias term
        bias_term = attn.bias.data if attn.bias is not None else None
        if bias_term is not None:
            bias_term = bias_term.to(orig_dtype)
        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias_term)
        return out
    def fft_finetune(self,input_matrix,amplitude_gain,phase_offset):
        """
        Perform Fast Fourier Transform (FFT) on a given input_matrix using PyTorch on GPU.

        Args:
            input_matrix (torch.Tensor): The input matrix for FFT on GPU.

        Returns:
            torch.Tensor: The iFFT (real part) of the FFT result on GPU.
        """
        # Ensure the input_matrix is on GPU.
        # input_matrix = input_matrix.cuda()

        # Calculate the FFT along the last two dimensions of the input_matrix on GPU.
        fft_result = torch.fft.fft2(input_matrix)
        fft_result = self.adjust_low_frequency(fft_result, amplitude_gain, phase_offset)
        # Calculate the inverse FFT (iFFT) on GPU.
        ifft_result = torch.fft.ifft2(fft_result)

        # Calculate the difference (error) between the original input_matrix and ifft_result
        # error = torch.abs(input_matrix - ifft_result)
        # print(f"Maximum Absolute Error between Original and iFFT Result: {torch.max(error)}")
        # print(f"error is {error}")
        return ifft_result

    def adjust_low_frequency(self,fft_result_gpu, amplitude_gain, phase_offset):
        """
        调整低频率分量的振幅和相位，同时保持频谱中心化。

        参数：
        - fft_result_gpu: 输入的2D DFT结果，复数矩阵（torch.Tensor）
        - amplitude_gain: 高频率分量的振幅增益因子（float）
        - phase_offset: 高频率分量的相位偏移值（float）

        返回值：
        - 调整后的FFT结果，复数矩阵（torch.Tensor）
        """

        # 获取 fft_result_gpu 的形状，通常是 (M, N)，其中 M 和 N 是矩阵的维度
        M, N = fft_result_gpu.shape

        # 使用 fftshift 将频域中心移到矩阵的中心
        fft_result_gpu_shifted = fft.fftshift(fft_result_gpu, dim=(-2, -1))

        # 创建一个与 fft_result_gpu 相同形状的振幅和相位调整矩阵
        amplitude_adjusted = torch.ones_like(fft_result_gpu_shifted)
        phase_adjusted = torch.zeros_like(fft_result_gpu_shifted)

        # 计算低频率分量的振幅和相位的调整
        crow, ccol = M // 2, N // 2
        amplitude_adjusted[..., crow - self.threshold:crow + self.threshold, ccol - self.threshold:ccol + self.threshold] = amplitude_gain
        phase_adjusted[..., crow - self.threshold:crow + self.threshold, ccol - self.threshold:ccol + self.threshold] = phase_offset

        # 将调整应用到频域结果
        amplitude_adjusted = amplitude_adjusted * fft_result_gpu_shifted.abs()
        phase_adjusted = phase_adjusted + fft_result_gpu_shifted.angle()

        # 构建新的复数值
        new_frequency_component = amplitude_adjusted * torch.exp(1j * phase_adjusted)

        # 使用 ifftshift 恢复频域中心位置
        new_frequency_component = fft.ifftshift(new_frequency_component, dim=(-2, -1))

        return new_frequency_component

class FFTAttnProcessor(nn.Module):
    def __init__(self, threshold,hidden_size, cross_attention_dim=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        
        self.to_q_fft = FFTLinearLayer(hidden_size, hidden_size,threshold)
        self.to_k_fft = FFTLinearLayer(cross_attention_dim or hidden_size, hidden_size,threshold)
        self.to_v_fft = FFTLinearLayer(cross_attention_dim or hidden_size, hidden_size,threshold)
        self.to_out_fft = FFTLinearLayer(hidden_size, hidden_size,threshold)

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        
        query = self.to_q_fft(attn.to_q, hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        key = self.to_k_fft(attn.to_k, encoder_hidden_states)
        # value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        value = self.to_v_fft(attn.to_v, encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        # hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        hidden_states = self.to_out_fft(attn.to_out[0], hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
