import torch
def get_ifft_result_real(fft_result):
    # Calculate the inverse FFT (iFFT)
    ifft_result = torch.fft.ifft2(fft_result)
    ifft_result = ifft_result+1j*torch.zeros_like(ifft_result)
    # Extract the real part and convert it to float
    ifft_result_real = ifft_result.real.float()
    return ifft_result_real
def adjust_fft_result(fft_result_gpu, amplitude_gain, phase_offset):
    """
    调整FFT结果中所有频率分量的振幅和相位。

    参数：
    - fft_result_gpu: 输入的2D DFT结果，复数矩阵（torch.Tensor）
    - amplitude_gain: 振幅的增益因子（float）
    - phase_offset: 相位的偏移值（float）

    返回值：
    - 调整后的FFT结果，复数矩阵（torch.Tensor）
    """

    # 获取 fft_result_gpu 的形状，通常是 (M, N)，其中 M 和 N 是矩阵的维度
    M, N = fft_result_gpu.shape

    # 使用广播操作同时调整所有频率分量的振幅和相位
    amplitude_adjusted = fft_result_gpu.abs() * amplitude_gain
    phase_adjusted = fft_result_gpu.angle() + phase_offset

    # 构建新的复数值
    new_frequency_component = amplitude_adjusted * torch.exp(1j * phase_adjusted)

    return new_frequency_component

# 示例用法
if __name__ == "__main__":
    # 假设 fft_result_gpu 是 2D DFT 的结果
    fft_result_gpu = torch.tensor([[1+2j, 2+3j], [3+4j, 4+5j]], dtype=torch.complex64)
    fft_result_gpu_inverse= get_ifft_result_real(fft_result_gpu)
    
    # 设置微调的增益因子和相位偏移值
    amplitude_gain = 1.5  # 增益因子
    phase_offset = 0.5  # 相位偏移值

    # 调用函数进行微调
    adjusted_result = adjust_fft_result(fft_result_gpu, amplitude_gain, phase_offset)
    adjust_fft_result_inverse = get_ifft_result_real(adjusted_result)
    # 输出原来的结果
    print("fft_result_gpu:")
    print(fft_result_gpu)
    # 输出调整后的结果
    print("adjust_fft_result:")
    print(adjusted_result)
    print("---------------------------------")
    # 输出原来的ifft
    print("fft_result_gpu_inverse:")
    print(fft_result_gpu_inverse)
    # 输出调整后的ifft
    print("adjust_fft_result_inverse:")
    print(adjust_fft_result_inverse)
