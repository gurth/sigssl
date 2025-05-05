import torch

# from models.networks.detr_utils.transformer import TransformerEncoderLayer
#
#
# def test_encoder_eval_train():
#     model = TransformerEncoderLayer(d_model=512, nhead=8)
#     model.eval()
#
#     # 使用相同的输入进行测试
#     input_tensor = torch.randn(10, 32, 512)  # 示例输入
#     pos_encoding = torch.randn(10, 32, 512)  # 示例位置编码
#
#     with torch.no_grad():
#         output_eval = model(input_tensor, pos=pos_encoding)
#         print("Output in eval mode:", output_eval)
#
#     model.train()
#     with torch.no_grad():
#         output_train = model(input_tensor, pos=pos_encoding)
#         print("Output in train mode:", output_train)
#
# def test_self_attn_eval_train():
#     model = TransformerEncoderLayer(d_model=512, nhead=8)
#     model.eval()
#
#     # 使用相同的输入进行测试
#     input_tensor = torch.randn(10, 32, 512)  # 示例输入
#     pos_encoding = torch.randn(10, 32, 512)  # 示例位置编码
#
#     model.eval()
#     with torch.no_grad():
#         q = k = model.with_pos_embed(input_tensor, pos_encoding)
#         src2_eval, attn_weights_eval = model.self_attn(q, k, value=input_tensor)
#         # print("Output src2 in eval mode:", src2_eval)
#
#     model.train()
#     with torch.no_grad():
#         q = k = model.with_pos_embed(input_tensor, pos_encoding)
#         src2_train, attn_weights_train = model.self_attn(q, k, value=input_tensor)
#         # print("Output src2 in train mode:", src2_train)
#
#     # 检查输出是否一致
#     if torch.allclose(src2_eval, src2_train, atol=1e-6):
#         print("Outputs are consistent between train and eval modes.")
#     else:
#         print("Outputs are inconsistent between train and eval modes.")
#
#     model.eval()
#     with torch.no_grad():
#         for name, param in model.self_attn.named_parameters():
#             print(f"Parameter {name} in eval mode: {param.data}")
#
#     model.train()
#     with torch.no_grad():
#         for name, param in model.self_attn.named_parameters():
#             print(f"Parameter {name} in train mode: {param.data}")
#
# def test_transformer_main():
#     test_self_attn_eval_train()