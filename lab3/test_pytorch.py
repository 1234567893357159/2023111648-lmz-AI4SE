import sys

def check_pytorch():
    """检查 PyTorch 及其相关库的安装状态"""
    print("=" * 60)
    print("🔍 开始检查 PyTorch 安装状态...")
    print("=" * 60)

    all_success = True

    # 1. 检查 torch
    try:
        import torch
        print(f"✅ torch 版本: {torch.__version__}")
        print(f"   安装位置: {torch.__file__}")
    except ImportError as e:
        print(f"❌ torch 导入失败: {e}")
        return False

    # 2. 检查 torchvision
    try:
        import torchvision
        print(f"✅ torchvision 版本: {torchvision.__version__}")
    except ImportError:
        print("⚠️ torchvision 未安装（可选）")
        all_success = False

    # 3. 检查 torchaudio
    try:
        import torchaudio
        print(f"✅ torchaudio 版本: {torchaudio.__version__}")
    except ImportError:
        print("⚠️ torchaudio 未安装（可选）")
        all_success = False

    # 4. 检查 CUDA 支持（仅当 torch 可用时）
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("ℹ️ CUDA 不可用（如果您使用的是 CPU 版本，这是正常的）")
        # 提示是否安装了 GPU 版本但驱动未配置
        if hasattr(torch, 'version') and hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
            print("   注意: 您安装的是 GPU 版本 PyTorch，但 CUDA 驱动或工具包未正确配置。")
            print("   请检查 NVIDIA 驱动及 CUDA 版本是否匹配。")

    # 5. 检查 MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"✅ MPS (Apple Silicon) 可用")

    # 6. 基本张量运算测试
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x + 1
        print(f"✅ 基本张量运算测试通过: {x} + 1 = {y}")
    except Exception as e:
        print(f"❌ 张量运算测试失败: {e}")
        all_success = False

    # 7. 尝试 GPU 运算（如果 CUDA 可用）
    if torch.cuda.is_available():
        try:
            a = torch.randn(3, 3).cuda()
            b = torch.mm(a, a.T)
            print("✅ GPU 运算测试通过")
        except Exception as e:
            print(f"❌ GPU 运算测试失败: {e}")
            all_success = False

    # 总结
    print("=" * 60)
    if all_success:
        print("🎉 所有检查通过！PyTorch 安装成功。")
        return True
    else:
        print("⚠️ 部分检查未通过，请根据上述信息排查。")
        return False

if __name__ == "__main__":
    success = check_pytorch()
    sys.exit(0 if success else 1)