import torch
import time

def fft_matrix_gpu(input_matrix):
    """
    Perform Fast Fourier Transform (FFT) on a given input_matrix using PyTorch on GPU.

    Args:
        input_matrix (torch.Tensor): The input matrix for FFT on GPU.

    Returns:
        torch.Tensor: The FFT of the input_matrix on GPU.
        torch.Tensor: The iFFT (real part) of the FFT result on GPU.
    """
    # Ensure the input_matrix is on GPU.
    input_matrix = input_matrix.cuda()

    # Start the timer for FFT
    start_time_fft = time.time()

    # Calculate the FFT along the last two dimensions of the input_matrix on GPU.
    fft_result = torch.fft.rfft2(input_matrix)

    # End the timer for FFT
    end_time_fft = time.time()

    # Calculate the elapsed time for FFT
    elapsed_time_fft = end_time_fft - start_time_fft
    print(f"FFT Calculation Time on GPU: {elapsed_time_fft} seconds")

    # Start the timer for inverse FFT
    start_time_ifft = time.time()

    # Calculate the inverse FFT (iFFT) on GPU.
    ifft_result = torch.fft.irfft2(fft_result,input_matrix.size())
    
    # # Extract the real part and convert it to float
    # ifft_result_real = ifft_result.real.float()

    # End the timer for inverse FFT
    end_time_ifft = time.time()

    # Calculate the elapsed time for inverse FFT
    elapsed_time_ifft = end_time_ifft - start_time_ifft
    print(f"iFFT Calculation Time on GPU: {elapsed_time_ifft} seconds")

    # Calculate the difference (error) between the original input_matrix and ifft_result
    error = torch.abs(input_matrix - ifft_result)
    print(f"Maximum Absolute Error between Original and iFFT Result: {torch.max(error)}")
    print(f"error is {error}")
    return fft_result, ifft_result

# Example usage on GPU:
if __name__ == "__main__":
    # Create a sample input matrix (replace this with your actual data) on GPU.
    # input_matrix_gpu = torch.tensor([[1.0, 2.0, 3.0],
    #                                  [4.0, 5.0, 6.0],
    #                                  [7.0, 8.0, 9.0]], dtype=torch.float32).cuda()
    # input_matrix_gpu = torch.tensor([[1.0, 2.0, 3.0],
    #                                  [4.0, 5.0, 6.0],
    #                                 ], dtype=torch.float32).cuda()
    
    matrix_size = 512

    # 构建一个1到512的矩阵
    input_matrix = torch.arange(1.0, matrix_size * matrix_size + 1.0, dtype=torch.float32).view(matrix_size, matrix_size)

    # 将矩阵移到GPU上
    input_matrix_gpu = input_matrix.cuda()
    print(input_matrix_gpu)
    # Perform FFT and iFFT on the input matrix on GPU, measure the time and error.
    fft_result_gpu, ifft_result_real_gpu = fft_matrix_gpu(input_matrix_gpu)

    # Print the FFT and iFFT results on GPU.
    print("FFT Result on GPU:")
    print(fft_result_gpu)
    print("iFFT Result (Real Part) on GPU:")
    print(ifft_result_real_gpu)
