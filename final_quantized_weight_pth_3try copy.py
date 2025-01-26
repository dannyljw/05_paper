# %%
import os
import torch
os.system("clear")
#%%
q_int_checkpoint_path = "/workspace/3.MyModel/05_paper/final_quantized_model_1epoch_state_dict.pth"
# print(torch.load(q_int_checkpoint_path,map_location='cpu').keys())

print("torch.load(q_int_checkpoint_path,map_location='cpu')['mobilenet_v2_branch.model.features_0.0.0.weight']")
# print(torch.load(q_int_checkpoint_path,map_location='cpu')['mobilenet_v2_branch.model.features_0.0.0.weight'])
quant_weight = torch.load(q_int_checkpoint_path,map_location='cpu')['mobilenet_v2_branch.model.features_0.0.0.weight']
print(quant_weight.dtype)
quant_tensor_size_kb = quant_weight.element_size() * quant_weight.numel() /1024
print(f"quant_tensor size : {quant_tensor_size_kb:.2f}KB")
#%%
print("torch.load(q_int_checkpoint_path,map_location='cpu')['mobilenet_v2_branch.model.features_0.0.0.weight'].int_repr()")
quant_int_weight = torch.load(q_int_checkpoint_path,map_location='cpu')['mobilenet_v2_branch.model.features_0.0.0.weight'].int_repr()
quant_int_tensor_size_kb = quant_int_weight.element_size() * quant_int_weight.numel() /1024
print(f"quant_int_weight : {quant_int_weight.dtype}")
print(f"quant_int_tensor size : {quant_int_tensor_size_kb:.2f}KB\n")
#%%
quant_all_weight = torch.load(q_int_checkpoint_path, map_location = 'cpu')
# print(quant_all_weight.keys())
for key, value in quant_all_weight.items():
    if "zero_point" in key:
        value_memory_kb = value.element_size() * value.numel()
        # print(f"key : {key}\n{value}\n{value_memory_kb:.2f}\n")
        print(value.dtype)
        print("zero_point")
        if value.dtype == torch.int64:
            print(value.dtype)
        elif value.dtype == torch.int32:
            print(value.dtype)
        else:
            print(f"ITS NOT : {value.dtype}")
        print("\n\n")
    elif "scale" in key:
        value_memory_kb = value.element_size() * value.numel()
        # print(f"key : {key}\n{value}\n{value_memory_kb:.2f}")
        # print("scale")
        continue
        if value.dtype == torch.float32:
            # print(value.dtype,"\n")
            continue
        elif value.dtype == torch.float16:
            # print(value.dtype,"\n")
            raise ValueError(f"Unexpected dtype for scale: {value.dtype}")
    # elif "weight" in key:
    #     print("-"*40)
    #     print(f"key : {key}")
    #     print(value.dtype)
    #     print("-"*40)
    #     print("\n\n")

#%%
for key, value in quant_all_weight.items():
    if "weight" in key:
        if value.dtype == torch.float32 or value.dtype == torch.qint8:
            # print(f"weight\n{value.dtype}")
            pass
        else:
            # print(f"Unexpected dtype for weight: {value.dtype}")
            pass
            raise ValueError(f"Unexpected dtype for weight: {value.dtype}")
    elif "scale" in key:
        # print(f"scale\n{value.dtype}")        
        if value.dtype == torch.float32:
            # print(value.dtype)
            pass
        else:
            raise ValueError(f"Unexpected dtype for weight: {value.dtype}")
    elif "zero_point" in key:
        if value.dtype == torch.int32 or value.dtype == torch.int64:
            print(key)
            print(value.dtype,"\n", value,"\n\n")
            
            pass
        else:
            raise ValueError(f"Unexpected dtype for zero_point: {value.dtype}")
    
    
    
#%%
# compare shape of the "weights" between .pth file which is quantized and .ckpt file which is not quantized
q_int_weight_path = "/workspace/3.MyModel/05_paper/final_quantized_model_1epoch_state_dict.pth"
q_int_weight = torch.load(q_int_weight_path, map_location='cpu')
q_int_weight_keys = q_int_weight.keys()
print(len(q_int_weight_keys))
print(q_int_weight)

float_weight_path = "/workspace/3.MyModel/torch_lite_famous_first/checkpoint/mbv2_224_slicing130_cross_DECODER_22_1_NEW-epoch=209-val_acc=0.6230.ckpt"
float_weight = torch.load(float_weight_path, map_location='cpu')
float_weight_keys = float_weight['state_dict'].keys()
print(len(float_weight_keys))
#%%
# compare name of the keys in q_int_weight_keys and float_weight_keys only if key contains the word "weight"
q_int_weight_keys = [key for key in q_int_weight_keys if "weight" in key]
float_weight_keys = [key for key in float_weight_keys if "weight" in key]
# delete word "mobilenet_v2_branch." which is at the front of all the keys at q_int_weight_keys.
q_int_weight_keys = [key.replace("mobilenet_v2_branch.", "") for key in q_int_weight_keys]