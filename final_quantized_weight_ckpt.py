import torch

# .ckpt 파일 경로
file_path = "/workspace/3.MyModel/torch_lite_famous_first/checkpoint/mbv2_224_slicing130_cross_DECODER_22_1_NEW_QAT-epoch=35-val_acc=0.60.ckpt"

# .ckpt 파일 로드
checkpoint_data = torch.load(file_path, map_location='cpu')  # map_location='cpu'로 파일을 로드하여 CPU로 이동

for key, value in checkpoint_data.items():
    if key == "state_dict":
        print(f"Key : {key}")
        for param_name, param in value.items():
            print(f" param_name : {param_name}")

# .ckpt 파일에 포함된 키와 각 데이터 항목 출력
for key, value in checkpoint_data.items():
    if isinstance(value, dict):
        print(f"{key}: (Dictionary with {len(value)} keys)")
    else:
        print(f"{key}: {type(value)}")

    # 모델 가중치(state_dict)일 경우 가중치 정보 출력
    
    
    
    if key == "state_dict":
        print("yes state_dict?")
        for param_name, param in value.items():
            print(f" - Parameter: {param_name}, Shape: {param.shape}")
            # print(f" Data : {param}")
            
            # 양자화된 텐서라면 scale과 zero_point 출력
            if param.is_quantized:
                print("-"*10,"Quantized Tensor","-"*10)
                print(f"   Scale: {param.q_scale()}")
                print(f"   Zero Point: {param.q_zero_point()}")
            print("-" * 40)
